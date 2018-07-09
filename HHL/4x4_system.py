# ======================================================================
# Copyright CERFACS (June 2018)
# Contributor: Adrien Suau (suau@cerfacs.fr)
#
# This software is governed by the CeCILL-B license under French law and
# abiding  by the  rules of  distribution of free software. You can use,
# modify  and/or  redistribute  the  software  under  the  terms  of the
# CeCILL-B license as circulated by CEA, CNRS and INRIA at the following
# URL "http://www.cecill.info".
#
# As a counterpart to the access to  the source code and rights to copy,
# modify and  redistribute granted  by the  license, users  are provided
# only with a limited warranty and  the software's author, the holder of
# the economic rights,  and the  successive licensors  have only limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using, modifying and/or  developing or reproducing  the
# software by the user in light of its specific status of free software,
# that  may mean  that it  is complicated  to manipulate,  and that also
# therefore  means that  it is reserved for  developers and  experienced
# professionals having in-depth  computer knowledge. Users are therefore
# encouraged  to load and  test  the software's  suitability as  regards
# their  requirements  in  conditions  enabling  the  security  of their
# systems  and/or  data to be  ensured and,  more generally,  to use and
# operate it in the same conditions as regards security.
#
# The fact that you  are presently reading this  means that you have had
# knowledge of the CeCILL-B license and that you accept its terms.
# ======================================================================

"""Implements the HHL algorithm https://arxiv.org/abs/1110.2232v2."""

import qiskit
from qiskit import available_backends, get_backend, execute,\
    QuantumRegister, ClassicalRegister, QuantumCircuit, CompositeGate
from sympy import pi
from utils.gates.qpe import qpe
import utils.gates.hamiltonian_4x4
from utils.endianness import QRegisterBE, CRegister
from copy import deepcopy
import utils.gates.crzz
import numpy as np
import scipy.linalg as la


def postselect(statevector, qubit_number, value: bool):
    mask = 1 << qubit_number
    if value:
        array_mask = np.arange(len(statevector)) & mask
    else:
        array_mask = not (np.arange(len(statevector)) & mask)

    def normalise(vec):
        from scipy.linalg import norm
        return vec / norm(vec)

    return normalise(statevector[array_mask != 0])

def round_to_zero(vec, tol=2e-15):
    vec.real[abs(vec.real) < tol] = 0.0
    vec.imag[abs(vec.imag) < tol] = 0.0
    return vec


Q_SPECS = {
    "name": "HHL",
    "circuits": [
        {
            "name": "4x4",
            "quantum_registers": [
                {
                    "name": "qb",
                    "size": 2
                },
                {
                    "name": "qC",
                    "size": 4
                },
                {
                    "name": "ancilla",
                    "size": 1
                },
            ],
            "classical_registers": [
                {
                    "name": "classicalx",
                    "size": 2
                },
                {
                    "name": "cc",
                    "size": 4
                },
                {
                    "name": "cr",
                    "size": 1
                },
            ]
        },
    ],
}
Q_program = qiskit.QuantumProgram(specs=Q_SPECS)

circuit = Q_program.get_circuit("4x4")
ancilla = QRegisterBE(Q_program.get_quantum_register("ancilla"))
qC = QRegisterBE(Q_program.get_quantum_register("qC"))
qb = QRegisterBE(Q_program.get_quantum_register("qb"))
cr = CRegister(Q_program.get_classical_register('cr'))
cc = CRegister(Q_program.get_classical_register('cc'))
classicalx = CRegister(Q_program.get_classical_register('classicalx'))

## 0. Initialise b
circuit.comment("[4x4] Initialising b.")
circuit.h(qb)
circuit.comment("[4x4] Initialisation done!")


## 1. Quantum Phase Estimation
def c_U_powers(n, circuit, control, target):
    # Previous method: just applying an optimized hamiltonian an exponential
    # number of times.
    # for i in range(2**n):
    #     #circuit.hamiltonian4x4(control, target).inverse()
    #     circuit.hamiltonian4x4(control, target)

    # Hard-coded values obtained thanks to scipy.optimize.minimize.
    # The error (2-norm of the difference between the unitary matrix of the
    # quantum circuit and the true matrix) is bounded by 1e-7, no matter the
    # value of n.
    power = 2**n
    if power == 1:
        params = [0.19634953,      0.37900987,   0.9817477,  1.87900984,  0.58904862 ]
    elif power == 2:
        params = [1.9634954,       1.11532058,   1.9634954,  2.61532069,  1.17809726 ]
    elif power == 4:
        params = [-0.78539816,     1.01714584,   3.92699082, 2.51714589,  2.35619449 ]
    elif power == 8:
        params = [-9.01416169e-09, -0.750000046, 1.57079632, 0.750000039, -1.57079633]
    else:
        raise NotImplementedError("You asked for a non-implemented power: {}".format(power))
    circuit.hamiltonian4x4(control, target, params)

circuit.comment("[4x4] 1. Quantum phase estimation.")
qpe_gate = circuit.qpe(qC, qb, c_U_powers)


## 2. Phase rotation controlled by the eigenvalue.
circuit.comment("[4x4] Inverting computed eigenvalues.")
circuit.swap(qC[1], qC[2])

# r is a parameter of the circuit.
# A good value is between 5 and 6 according to the article.
r = 6
circuit.comment("[4x4] 2. Phase rotation.")
def cry(circuit, theta, ctrl, target):
    circuit.comment("CRY")
    # Apply the supposed c-RY operation.
    circuit.cu3(theta, 0, 0, ctrl, target)

for i in range(len(qC)):
    cry(circuit, 2**(len(qC)-i-r)*pi, qC[len(qC)-1-i], ancilla[0])

circuit.comment("Inverting the inversion of eigenvalues.")
circuit.swap(qC[1], qC[2])

## 3. Uncompute the Quantum Phase Estimation.
circuit.comment("[4x4] 3. Inverting quantum phase estimation.")
circuit._attach(deepcopy(qpe_gate).inverse())

circuit_no_measure = deepcopy(circuit)

## 4. Measure the ancilla qubit to check.
circuit.comment("[4x4] 4. Measurement.")
circuit.measure(ancilla, cr)
circuit.measure(qC, cc)
circuit.measure(qb, classicalx)

with open('4x4.qasm', 'w') as f:
    f.write(circuit.qasm())

qasm_sim = get_backend('local_qasm_simulator')
state_sim = get_backend('local_statevector_simulator')
unitary_sim = get_backend('local_unitary_simulator')

# res_qasm = execute([circuit], qasm_sim, shots=10**5).result()
# counts = res_qasm.get_counts()
# filtered_counts = {key: counts[key] for key in counts if key[-1] == '1'}
# significant_counts = {key: counts[key] for key in counts if counts[key] > 100}
# significant_filtered_counts = {key: filtered_counts[key]
#                                for key in filtered_counts
#                                if filtered_counts[key] > 5000}
# print("Counts:", counts, sep='\n')

res_state = execute([circuit_no_measure], state_sim).result()
statevector = round_to_zero(postselect(res_state.get_statevector(), 6, True), 1e-3)
full_state = round_to_zero(res_state.get_statevector(), 1e-3)
amplitudes = np.absolute(full_state)**2

solution = np.sqrt(340) * statevector[:4]
x_exact = np.array([-1, 7, 11, 13])

print("Exact solution: {}".format(x_exact))
print("Experimental solution: {}".format(solution))
print("Error in found solution: {}".format(la.norm(solution - x_exact)))

# res_unitary = execute([circuit_no_measure], unitary_sim, skip_translation=skip).result()
# unitary = res_unitary.get_unitary()
# print("Unitary matrix:", unitary, sep='\n')

X = np.arange(len(full_state))
import matplotlib.pyplot as plt
plt.bar(X, np.real(full_state))
plt.xticks(np.arange(0, len(X)+1, 2),
           [' '.join([bin(n)[3], bin(n)[4:8], bin(n)[8:10]]) for n in np.arange(len(X),2*len(X),2)],
           usetex=False, rotation='vertical')
plt.grid(zorder=0)
plt.show()
# plot_histogram(counts)
