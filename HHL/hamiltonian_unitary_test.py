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

"""This file compute the error between a simulated unitary matrix and an exact one.

Depending on what part of the file is commented, the circuit being simulated will
change.
"""

import qiskit
from qiskit import available_backends, get_backend, execute, QuantumRegister, ClassicalRegister, QuantumCircuit
from sympy import pi
from utils.endianness import QRegisterBE, CRegister
import utils.gates.comment
import numpy as np
import utils.gates.hamiltonian_4x4

Q_SPECS = {
    "name": "Hamiltonian_unitary",
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
                    "size": 1
                },
            ],
            "classical_registers": [
                # {
                #     "name": "classicalX",
                #     "size": 2
                # }
            ]
        }
    ],
}
Q_program = qiskit.QuantumProgram(specs=Q_SPECS)

circuit = Q_program.get_circuit("4x4")
qb = QRegisterBE(Q_program.get_quantum_register("qb"))
ctrl = QRegisterBE(Q_program.get_quantum_register("ctrl"))
# classicalX = CRegister(Q_program.get_classical_register('classicalX'))

def round_to_zero(vec, tol=2e-15):
    vec.real[abs(vec.real) < tol] = 0.0
    vec.imag[abs(vec.imag) < tol] = 0.0
    return vec


from sympy import pi

def cry(circuit, theta, ctrl, target):
    # Verified
    # Apply the supposed c-RY operation.
    circuit.cu3(theta, pi, pi, ctrl, target)
    # For the moment, QISKit adds a phase to the U-gate, so we
    # need to correct this phase with a controlled Rzz.
    circuit.crzz(pi, ctrl, target)
cry(circuit, 2*pi, ctrl[0], qb[0])

unitary_sim = get_backend('local_unitary_simulator')
res = execute([circuit], unitary_sim).result()
unitary = round_to_zero(res.get_unitary())

# def controlled(U):
#     n = int(U.shape[0]/2)
#     return np.vstack((np.hstack((np.identity(n),   np.zeros((n, 3*n)))),
#                       np.hstack((np.zeros((n, n)), U[:n,:n], np.zeros((n, n)), U[:n,n:])),
#                       np.hstack((np.zeros((n, 2*n)), np.identity(n), np.zeros((n, n)))),
#                       np.hstack((np.zeros((n, n)), U[n:,:n], np.zeros((n, n)), U[n:,n:])),
#     ))

# def U(theta, phi, lam):
#     return np.array([[np.cos(theta/2), -np.exp(1.j*lam)*np.sin(theta/2)], [np.exp(1.j*phi)*np.sin(theta/2), np.exp(1.j*(phi+lam))*np.cos(theta/2)]])

# def Rx(theta):
#     return np.array([[np.cos(theta/2), 1j*np.sin(theta/2)],
#                      [1j*np.sin(theta/2), np.cos(theta/2)]])

# def Rzz(theta):
#     return np.array([[np.exp(1j*theta), 0],
#                      [0, np.exp(1j*theta)]])



def swap(U):
    from copy import deepcopy
    cpy = deepcopy(U)
    cpy[[1,2],:] = cpy[[2,1],:]
    cpy[:,[1,2]] = cpy[:,[2,1]]
    return cpy


# from scipy.linalg import expm, sqrtm, norm
# X = np.array([0,1,1,0]).reshape((2,2))
# sqrtX = sqrtm(X)
# ctrl_sqrtX = controlled(sqrtX)
# Z = np.array([1,0,0,-1]).reshape((2,2))

# A = .25 * np.array([[15, 9, 5, -3], [9, 15, 3, -5], [5, 3, 15, -9], [-3, -5, -9, 15]])
# expA = swap(expm(-1.j * A * 2 * np.pi / 16))

# unit = unitary[1::2, 1::2]

# err = norm(unit - expA)

np.set_printoptions(linewidth=110)
print("Simulated matrix:\n", swap(unitary), sep='')
# print("Theoretical matrix:\n", expA, sep='')
# print("Error: ", err, sep='')
# print("Matrices are equal?\n", np.isclose(unitary, expected), sep='')
