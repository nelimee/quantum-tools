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

"""Testing the QISKit initialiser."""

import qiskit

statevector_backend = qiskit.get_backend('local_statevector_simulator')

###############################################################
# Make a quantum program for state initialization.
###############################################################
qubit_number = 5
Q_SPECS = {
    "name": "StatePreparation",
    "circuits": [
        {
            "name": "initializerCirc",
            "quantum_registers": [{
                "name": "qr",
                "size": qubit_number
            }],
            "classical_registers": [{
                "name": "cr",
                "size": qubit_number
            }]},
    ],
}
Q_program = qiskit.QuantumProgram(specs=Q_SPECS)

## State preparation
import numpy as np
from qiskit.extensions.quantum_initializer import _initializer

def psi_0_coefficients(qubit_number: int):
    T = 2**qubit_number
    tau = np.arange(T)
    return np.sqrt(2 / T) * np.sin(np.pi * (tau + 1/2) / T)

def get_coeffs(qubit_number: int):
    # Can be changed to anything, the initialize function will take
    # care of the initialisation.
    return np.ones((2**qubit_number,)) / np.sqrt(2**qubit_number)
    #return psi_0_coefficients(qubit_number)

circuit_prep = Q_program.get_circuit("initializerCirc")
qr = Q_program.get_quantum_register("qr")
cr = Q_program.get_classical_register('cr')
coeffs = get_coeffs(qubit_number)
_initializer.initialize(circuit_prep, coeffs, [qr[i] for i in range(len(qr))])

res = qiskit.execute(circuit_prep, statevector_backend).result()
statevector = res.get_statevector("initializerCirc")
print(statevector)


# ## Quantum phase estimation
# from utils.gates.qpe import qpe
# from utils.endianness import QRegisterBE, CRegister
# import sympy as sym
# def c_U_powers(n: int, circuit, control, target):
#     for i in range(2**n):
#         circuit.u1(-3 * sym.pi / 8, target[0])
#         circuit.cx(control, target[0])
#         circuit.u1(3 * sym.pi / 8, target[0])
#         circuit.cx(control, target[0])

# circuit_phase = Q_program.get_circuit("QuantumPhaseEstimation")
# qphase = QRegisterBE(Q_program.get_quantum_register("qphase"))
# qvect = QRegisterBE(Q_program.get_quantum_register("qvect"))
# cr = CRegister(Q_program.get_classical_register('cr'))

# qpe(circuit_phase, qphase, qvect, c_U_powers)


# ## Controlled qubit test
# from utils.gates.qpe import qpe
# from utils.endianness import QRegisterBE, CRegister
# import sympy as sym
# def c_U_powers(n: int, circuit, control, target):
#     for i in range(2**n):
#         circuit.u1(-3 * sym.pi / 8, target[0])
#         circuit.cx(control, target[0])
#         circuit.u1(3 * sym.pi / 8, target[0])
#         circuit.cx(control, target[0])

# circuit_test = Q_program.get_circuit("ControlTest")
# qphase = QRegisterBE(Q_program.get_quantum_register("qphase"))
# qvect = QRegisterBE(Q_program.get_quantum_register("qvect"))
# qcontrol = QRegisterBE(Q_program.get_quantum_register("qcontrol"))
# cr = CRegister(Q_program.get_classical_register('cr'))

# qpe_gate = qpe(circuit_test, qphase, qvect, c_U_powers)
# from pprint import pprint

# circuit_test._attach(qpe_gate.q_if(qcontrol[0]))
