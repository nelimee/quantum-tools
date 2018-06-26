# ======================================================================
# Copyright CERFACS (February 2018)
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

"""This file compute the error between the simulated hamiltonian and the exact one.

The output of this script is the maximum error between the simulated hamiltonian and
the exact (up to some floating-point rounding during the computation of the matrix
exponential) hamiltonian matrix. The maximum is taken on the errors for different
powers of the Hamiltonian.
"""

from qiskit import get_backend, execute, QuantumProgram
from utils.endianness import QRegisterBE, CRegister
import utils.gates.comment
import numpy as np
import utils.gates.hamiltonian_4x4
import scipy.linalg as la

def swap(U):
    from copy import deepcopy
    cpy = deepcopy(U)
    cpy[[1,2],:] = cpy[[2,1],:]
    cpy[:,[1,2]] = cpy[:,[2,1]]
    return cpy

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


Q_SPECS = {
    "name": "Hamiltonian_error",
    "circuits": [
        {
            "name": "4x4",
            "quantum_registers": [
                {
                    "name": "ctrl",
                    "size": 1
                },
                {
                    "name": "qb",
                    "size": 2
                },
            ],
            "classical_registers": [
            {
                "name": "classicalX",
                "size": 2
            }]
        }
    ],
}

errors = list()
max_power = 4

for power in range(max_power):
    # Create the quantum program
    Q_program = QuantumProgram(specs=Q_SPECS)
    # Recover the circuit and the registers.
    circuit = Q_program.get_circuit("4x4")
    qb = QRegisterBE(Q_program.get_quantum_register("qb"))
    ctrl = QRegisterBE(Q_program.get_quantum_register("ctrl"))
    classicalX = CRegister(Q_program.get_classical_register('classicalX'))

    # Apply the controlled-Hamiltonian.
    c_U_powers(power, circuit, ctrl[0], qb)

    # Get the unitary simulator backend and compute the unitary matrix
    # associated with the controlled-Hamiltonian gate.
    unitary_sim = get_backend('local_unitary_simulator')
    res = execute([circuit], unitary_sim).result()
    unitary = res.get_unitary()

    # Compute the exact unitary matrix we want to approximate.
    A = .25 * np.array([[15, 9, 5, -3],
                        [9, 15, 3, -5],
                        [5, 3, 15, -9],
                        [-3, -5, -9, 15]])
    # QISKit uses a different ordering for the unitary matrix.
    # The following line change the matrix ordering of the ideal
    # unitary to make it match with QISKit's ordering.
    expA = swap(la.expm(1.j * (2**power) * A * 2 * np.pi / 16))
    # As the simulated matrix is controlled, there are ones and zeros
    # (for the control) at each even position. As they don't appear in
    # our ideal matrix, we want to erase them. This is done by taking
    # only the odds indices.
    unit = unitary[1::2, 1::2]

    errors.append(la.norm(unit - expA))

print("Maximum computed error between the ideal matrix and "
      f"the one implemented on the quantum circuit: {max(errors)}")
