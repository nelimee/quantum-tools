# ======================================================================
# Copyright CERFACS (May 2018)
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

"""[OUTDATED] See 4x4hamiltonian.py
"""
from typing import Tuple, Union, Callable, List
from qiskit import QuantumCircuit, QuantumRegister, CompositeGate, QuantumGate
from utils.endianness import QRegisterBE
from qiskit.mapper._compiling import euler_angles_1q
import numpy as np
from utils.gates.qpe import qpe

QubitType = Tuple[QuantumRegister, int] #pylint: disable=invalid-name

class HHL2x2Gate(CompositeGate):

    def __init__(self,
                 b_quantum_register: QRegisterBE,
                 clock_quantum_register: QRegisterBE,
                 ancilla_qubit: QubitType,
                 A,
                 b,
                 qcirc: QuantumCircuit = None):
        """Initialize the HHL2x2Gate class.

        Apply the HHL algorithm to estimate the solution of the linear
        system Ax = b.

        Parameters:
            b_quantum_register: Quantum register large enough to encode
               the vector 'b'. Here, 'b' must be 2x1 so b_quantum_register
               should have at least 1 qubit.
            clock_quantum_register: Quantum register representing the clock.
               An increase in the size of this register will increase the
               precision of the algorithm.
            ancilla_qubit: Ancilla qubit used to check if the result encoded
               in 'b_quantum_register' matches with 'x', the unknown of the
               problem.
            A: The 2x2 hermitian matrix of the system.
            b: The 2x1 right-hand side vector.
            qcirc: The associated quantum circuit.
        """

        assert(len(b_quantum_register) >= 1, "The quantum register to store 'b' should "
               "contains at least one qubit.")
        assert(len(clock_quantum_register) > 0, "The clock quantum register should "
               "contains at least one qubit.")
        assert(A.shape == (2,2), "A should be a 2x2 matrix.")
        assert(A == A.T.conj(), "A should be Hermitian.")
        assert(B.shape == (2,) or B.shape == (2,1), "B should be a 2x1 vector.")

        n, m = len(b_quantum_register), len(clock_quantum_register)
        b_qubits = [b_quantum_register[i] for i in range(n)]
        clock_qubits = [clock_quantum_register[i] for i in range(m)]
        used_qubits = b_qubits + clock_qubits + [ancilla_qubit]

        super().__init__(self.__class__.__name__, # name
                         [A, b],                  # parameters
                         used_qubits,             # qubits
                         qcirc)                   # circuit

        ## 0. Preliminary steps.
        # 0.1. Initialise the 'b' register.
        #     This is done with a very bad algorithm (exponential) but for the moment
        #     it is sufficient.
        self._initialise_b(b, b_quantum_register)

        ## 1. Phase estimation procedure
        def c_U_powers(power: int, circuit, control, target):
            pass
        qpe(self, clock_quantum_register, b_quantum_register, c_U_powers)


        for i in range(n):
            c_U_powers(i, self, phase_quantum_register[n-1-i], eigenvector_quantum_register)

        iqft_be(self, phase_quantum_register, qcirc)

    def _A_simulation_parameters(self, A) -> QuantumGate:
        from qiskit.mapper._compiling import euler_angles_1q
        return euler_angles_1q(A)[:3]

    def _initialise_b(self, b: np.ndarray, b_qreg: QuantumRegister) -> None:
        def starting_state_coefficients(qubit_number: int):
            T = 2**qubit_number
            tau = np.arange(T)
            return np.sqrt(2 / T) * np.sin(np.pi * (tau + 1/2) / T)

        coeffs = starting_state_coefficients(len(b_qreg))
        initialize(self, coeffs, b_qreg)


def qpe(self,
        phase_quantum_register: QRegisterBE,
        eigenvector_quantum_register: QRegisterBE,
        c_U_powers: Callable[[int,
                              CompositeGate,
                              QubitType,
                              QRegisterBE], None],
        precision: float = 0.1) -> QuantumPhaseEstimationGate:
    self._check_qreg(phase_quantum_register)
    self._check_qreg(eigenvector_quantum_register)
    self._check_dups([phase_quantum_register, eigenvector_quantum_register])
    return self._attach(QuantumPhaseEstimationGate(phase_quantum_register,
                                                   eigenvector_quantum_register,
                                                   c_U_powers,
                                                   precision,
                                                   self))
