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

"""This module contains functions to add 2 quantum registers.

For the moment the module provides 2 addition function:
1) A circuit that will add 2 quantum registers and store the result in a third one.
   This implementation use many qubits (3N+1) and so is really inefficient for
   simulations
2) A circuit that perform |a>|b> -> |a>|b+a>.
   The order of the operands around the "+" is important because we can implement
   a substractor by inverting the circuit and the substractor will perform the
   operation |a>|b> -> |a>|b-a> (and not |a>|b> -> |a>|a-b>).
"""
import math
from typing import Tuple, Union
import sympy as sym
from qiskit import QuantumCircuit, QuantumRegister, CompositeGate
from utils.endianness import QRegisterPhaseLE

QubitType = Tuple[QuantumRegister, int] #pylint: disable=invalid-name

class _CarryGate(CompositeGate):

    def __init__(self,
                 input_carry: QubitType,
                 lhs: QubitType,
                 rhs: QubitType,
                 output_carry: QubitType,
                 qcirc: QuantumCircuit = None):
        """Initialize the _CarryGate class.

        Compute the carry bit for the given inputs.

        Parameters:
            input_carry  (QuantumRegister, int) the carry bit of the previous operation.
            lhs          (QuantumRegister, int) left-hand side.
            rhs          (QuantumRegister, int) right-hand side.
            output_carry (QuantumRegister, int) the computed carry bit.
            qcirc        (QuantumCircuit)       the associated quantum circuit.
        """
        super().__init__(self.__class__.__name__, # name
                         [], # parameters
                         [input_carry, lhs, rhs, output_carry], # qubits
                         qcirc)                                 # circuit
        self.ccx(lhs, rhs, output_carry)
        self.cx(lhs, rhs)
        self.ccx(input_carry, rhs, output_carry)

def _carry(self,
           input_carry: QubitType,
           lhs: QubitType,
           rhs: QubitType,
           output_carry: QubitType,
           qcirc: QuantumCircuit = None) -> _CarryGate:
    self._check_qubit(input_carry)
    self._check_qubit(lhs)
    self._check_qubit(rhs)
    self._check_qubit(output_carry)
    self._check_dups([input_carry, lhs, rhs, output_carry])
    return self._attach(_CarryGate(input_carry, lhs, rhs, output_carry, qcirc))

def _icarry(self,
            input_carry: QubitType,
            lhs: QubitType,
            rhs: QubitType,
            output_carry: QubitType,
            qcirc: QuantumCircuit = None) -> _CarryGate:
    self._check_qubit(input_carry)
    self._check_qubit(lhs)
    self._check_qubit(rhs)
    self._check_qubit(output_carry)
    self._check_dups([input_carry, lhs, rhs, output_carry])
    return self._attach(_CarryGate(input_carry, lhs, rhs, output_carry, qcirc).inverse())





class _BitAddWithoutCarryGate(CompositeGate):
    def __init__(self,
                 input_carry: QubitType,
                 lhs: QubitType,
                 rhs: QubitType,
                 qcirc: QuantumCircuit = None):
        """Initialize the _BitAddWithoutCarryGate class.

        Compute result: = lhs + rhs + carry (mod 2).

        Parameters:
            input_carry (QuantumRegister, int) the carry bit of the previous operation.
            lhs         (QuantumRegister, int) left-hand side.
            rhs         (QuantumRegister, int) right-hand side.
            qcirc       (QuantumCircuit)       the associated quantum circuit.
        """
        super().__init__(self.__class__.__name__, # name
                         [],                      # parameters
                         [input_carry, lhs, rhs], # qubits
                         qcirc)                   # circuit
        self.cx(lhs, rhs)
        self.cx(input_carry, rhs)


def _bit_add_without_carry(self,
                           input_carry: QubitType,
                           lhs: QubitType,
                           rhs: QubitType,
                           qcirc: QuantumCircuit = None) -> _BitAddWithoutCarryGate:
    self._check_qubit(input_carry)
    self._check_qubit(lhs)
    self._check_qubit(rhs)
    self._check_dups([input_carry, lhs, rhs])
    return self._attach(_BitAddWithoutCarryGate(input_carry, lhs, rhs, qcirc))

def _ibit_add_without_carry(self,
                            input_carry: QubitType,
                            lhs: QubitType,
                            rhs: QubitType,
                            qcirc: QuantumCircuit = None) -> _BitAddWithoutCarryGate:
    self._check_qubit(input_carry)
    self._check_qubit(lhs)
    self._check_qubit(rhs)
    self._check_dups([input_carry, lhs, rhs])
    return self._attach(_BitAddWithoutCarryGate(input_carry, lhs, rhs, qcirc).inverse())




class AddCQPGate(CompositeGate):
    """Implement the Conventional Quantum Plain adder.

    Implements the CQP adder presented in "Quantum Plain and Carry Look-Ahead Adders"
    written by Kai-Wen Cheng and Chien-Cheng Tseng in 2002.
    The implementation is FAR from optimal, this algorithm is a naive algorithm to
    add 2 quantum registers.
    """
    def __init__(self,
                 lhs: QuantumRegister,
                 rhs: QuantumRegister,
                 output_carry: QubitType,
                 ancilla: QuantumRegister,
                 qcirc: QuantumCircuit = None):
        """Initialize the AddCQP class.

        Implements the CQP adder presented in "Quantum Plain and Carry Look-Ahead Adders"
        written by Kai-Wen Cheng and Chien-Cheng Tseng in 2002.
        The implementation is FAR from optimal, this algorithm is a naive algorithm to
        add 2 quantum registers.

        Parameters:
            lhs          (QuantumRegister)      left-hand side.
            rhs          (QuantumRegister)      right-hand side.
            output_carry (QuantumRegister, int) set to 1 if the addition overflowed.
            ancilla      (QuantumRegister)      ancilla register: should contain at least N qubits.
            qcirc        (QuantumCircuit)       the associated circuit.
        """

        used_qubits = [qubit[i]
                       for qubit in [lhs, rhs, ancilla]
                       for i in range(len(qubit))] + [output_carry]

        super().__init__(self.__class__.__name__, # name
                         [],                      # parameters
                         used_qubits,             # qubits
                         qcirc)                   # circuit

        qubit_number = min([len(lhs), len(rhs), len(ancilla)])
        # qubit_number is the number of qubits we will use, so it cost nothing to check.
        lhs.check_range(qubit_number-1)
        rhs.check_range(qubit_number-1)
        ancilla.check_range(qubit_number-1)

        # 1. Compute the final carry
        for i in range(qubit_number-1):
            _carry(self, ancilla[i], lhs[i], rhs[i], ancilla[i+1], qcirc)
        _carry(self,
               ancilla[qubit_number-1],
               lhs[qubit_number-1],
               rhs[qubit_number-1],
               output_carry,
               qcirc)

        self.cx(lhs[qubit_number-1], rhs[qubit_number-1])

        # 2. Perform the additions with the computed carry bits and reverse
        #    the carry operation
        for i in range(qubit_number-1, 0, -1):
            _bit_add_without_carry(self, ancilla[i], lhs[i], rhs[i], qcirc)
            _icarry(self, ancilla[i-1], lhs[i-1], rhs[i-1], ancilla[i], qcirc)
        _bit_add_without_carry(self, ancilla[0], lhs[0], rhs[0], qcirc)


def add_cqp(self,
            lhs: QuantumRegister,
            rhs: QuantumRegister,
            output_carry: QubitType,
            ancilla: QuantumRegister,
            qcirc: QuantumCircuit = None) -> AddCQPGate:
    """Add to self the gates to perform |lhs>|rhs> -> |lhs>|rhs+lhs>."""
    self._check_qreg(lhs)
    self._check_qreg(rhs)
    self._check_qubit(output_carry)
    self._check_qreg(ancilla)
    self._check_dups([lhs, rhs, output_carry[0], ancilla])
    return self._attach(AddCQPGate(lhs, rhs, output_carry, ancilla, qcirc))

# TODO: does not substract for the moment, need checking.
# def iadd_cqp(self,
#             lhs: QuantumRegister,
#             rhs: QuantumRegister,
#             result: QuantumRegister,
#             output_carry: QubitType,
#             ancilla: QuantumRegister,
#             qcirc: QuantumCircuit = None) -> AddCQPGate:
#     self._check_qreg(lhs)
#     self._check_qreg(rhs)
#     self._check_qreg(result)
#     self._check_qubit(output_carry)
#     self._check_qreg(ancilla)
#     self._check_dups([lhs, rhs, result, output_carry[0], ancilla])
#     return self._attach(AddCQPGate(lhs, rhs, result, output_carry, ancilla, qcirc).inverse())


class _MAJGate(CompositeGate):

    def __init__(self,
                 carry: QubitType,
                 rhs: QubitType,
                 lhs: QubitType,
                 qcirc: QuantumCircuit = None):
        """Initialize the _MAJ (MAJority) class.

        This gate is used to perform an addition in "A new quantum ripple-carry addition circuit"
        written by Steven A. Cuccaro, Thomas G. Draper, Samuel A. Kutin and David Petrie Moulton
        in 2008.

        Parameters:
            carry (QuantumRegister, int) the carry bit of the previous operation.
            rhs   (QuantumRegister, int) right-hand side.
            lhs   (QuantumRegister, int) left-hand side.
            qcirc (QuantumCircuit)       the associated quantum circuit.
        """
        super().__init__(self.__class__.__name__, # name
                         [],                      # parameters
                         [carry, lhs, rhs],       # qubits
                         qcirc)                   # circuit
        self.cx(lhs, rhs)
        self.cx(lhs, carry)
        self.ccx(carry, rhs, lhs)

def _maj(self,
         carry: QubitType,
         rhs: QubitType,
         lhs: QubitType,
         qcirc: QuantumCircuit = None) -> _MAJGate:
    self._check_qubit(lhs)
    self._check_qubit(rhs)
    self._check_qubit(carry)
    return self._attach(_MAJGate(carry, rhs, lhs, qcirc))

def _imaj(self,
          carry: QubitType,
          rhs: QubitType,
          lhs: QubitType,
          qcirc: QuantumCircuit = None) -> _MAJGate:
    self._check_qubit(lhs)
    self._check_qubit(rhs)
    self._check_qubit(carry)
    return self._attach(_MAJGate(carry, rhs, lhs, qcirc).inverse())



class _UMAGate(CompositeGate):

    def __init__(self,
                 carry: QubitType,
                 rhs: QubitType,
                 lhs: QubitType,
                 qcirc: QuantumCircuit = None):
        """Initialize the _UMA (UnMajority and Add) class.

        This gate is used to perform an addition in "A new quantum ripple-carry addition circuit"
        written by Steven A. Cuccaro, Thomas G. Draper, Samuel A. Kutin and David Petrie Moulton
        in 2008.

        Parameters:
            carry (QuantumRegister, int) the carry bit of the previous operation.
            rhs   (QuantumRegister, int) right-hand side.
            lhs   (QuantumRegister, int) left-hand side.
            qcirc (QuantumCircuit)       the associated quantum circuit.
        """
        super().__init__(self.__class__.__name__, # name
                         [],                      # parameters
                         [lhs, rhs, carry],       # qubits
                         qcirc)                   # circuit
        self.x(rhs)
        self.cx(carry, rhs)
        self.ccx(carry, rhs, lhs)
        self.x(rhs)
        self.cx(lhs, carry)
        self.cx(lhs, rhs)

def _uma(self,
         carry: QubitType,
         rhs: QubitType,
         lhs: QubitType,
         qcirc: QuantumCircuit = None) -> _UMAGate:
    self._check_qubit(lhs)
    self._check_qubit(rhs)
    self._check_qubit(carry)
    return self._attach(_UMAGate(carry, rhs, lhs, qcirc))

def _iuma(self,
          carry: QubitType,
          rhs: QubitType,
          lhs: QubitType,
          qcirc: QuantumCircuit = None) -> _UMAGate:
    self._check_qubit(lhs)
    self._check_qubit(rhs)
    self._check_qubit(carry)
    return self._attach(_UMAGate(carry, rhs, lhs, qcirc).inverse())



class AddRCGate(CompositeGate):
    """Ripple-Carry adder.

    Implements the Ripple-Carry Adder presented in "A new quantum ripple-carry addition circuit"
    and written by Steven A. Cuccaro, Thomas G. Draper, Samuel A. Kutin and David Petrie Moulton
    in 2008.
    """
    def __init__(self,
                 lhs: QuantumRegister,
                 rhs: QuantumRegister,
                 output_carry: QubitType,
                 input_carry: QubitType,
                 qcirc: QuantumCircuit = None):
        """Initialise the AddRCGate class.

        Implements the Ripple-Carry Adder presented in "A new quantum ripple-carry addition circuit"
        and written by Steven A. Cuccaro, Thomas G. Draper, Samuel A. Kutin and David Petrie Moulton
        in 2008.

        Parameters:
            lhs          (QuantumRegister)      left-hand side.
            rhs          (QuantumRegister)      right-hand side AND result.
            output_carry (QuantumRegister, int) set to 1 if the addition overflowed.
            input_carry  (QubitType)            input_carry qubit.
            qcirc        (QuantumCircuit)       the circuit on which to add the gates.
        """

        used_qubits = [qubit[i]
                       for qubit in [lhs, rhs]
                       for i in range(len(qubit))] + [output_carry, input_carry]

        super().__init__(self.__class__.__name__, # name
                         [],                      # parameters
                         used_qubits,             # qubits
                         qcirc)                   # circuit

        qubit_number = min(len(lhs), len(rhs))
        # qubit_number is the number of qubits we will use, so it cost nothing to check.
        lhs.check_range(qubit_number-1)
        rhs.check_range(qubit_number-1)

        _maj(self, input_carry, rhs[0], lhs[0], qcirc)
        for i in range(1, qubit_number):
            _maj(self, lhs[i-1], rhs[i], lhs[i], qcirc)

        self.cx(lhs[qubit_number-1], output_carry)
        for i in range(qubit_number-1, 0, -1):
            _uma(self, lhs[i-1], rhs[i], lhs[i], qcirc)
        _uma(self, input_carry, rhs[0], lhs[0], qcirc)


def add_rc(self,
           lhs: QuantumRegister,
           rhs: QuantumRegister,
           output_carry: QubitType,
           input_carry: QubitType,
           qcirc: QuantumCircuit = None) -> AddRCGate:
    """Add to self the gates to perform |lhs>|rhs> -> |lhs>|rhs+lhs>."""
    self._check_qreg(lhs)
    self._check_qreg(rhs)
    self._check_qubit(output_carry)
    self._check_qubit(input_carry)
    self._check_dups([lhs, rhs, output_carry, input_carry])
    return self._attach(AddRCGate(lhs, rhs, output_carry, input_carry, qcirc))

def iadd_rc(self,
            lhs: QuantumRegister,
            rhs: QuantumRegister,
            output_carry: QubitType,
            input_carry: QubitType,
            qcirc: QuantumCircuit = None) -> AddRCGate:
    """Add to self the gates to performs the operation |lhs>|rhs> -> |lhs>|rhs-lhs>"""
    self._check_qreg(lhs)
    self._check_qreg(rhs)
    self._check_qubit(output_carry)
    self._check_qubit(input_carry)
    self._check_dups([lhs, rhs, output_carry, input_carry])
    return self._attach(AddRCGate(lhs, rhs, output_carry, input_carry, qcirc).inverse())




class ApproximateAddFourierStateGate(CompositeGate):
    """Approximate Quantum Fourier state adder.

    Implements the fourier adder presented in "Addition on a Quantum Computer",
    written by Thomas G. Draper in 1998 and revised in 2000. Let F(a) be the
    quantum fourier transform of a, this class implement the gate that compute
    the transformation |a>|F(b)> -> |a>|F(b+a)>.
    """
    def __init__(self,
                 lhs: Union[int, QuantumRegister],
                 rhs: QRegisterPhaseLE,
                 qcirc: QuantumCircuit,
                 approximation: int = None):
        """Initialise the ApproximateAddFourierStateGate class.

        Implements the fourier adder presented in "Addition on a Quantum Computer",
        written by Thomas G. Draper in 1998 and revised in 2000. Let F(a) be the
        quantum fourier transform of a, this class implement the gate that compute
        the transformation |a>|F(b)> -> |a>|F(b+a)>.

        Requires:
            1) rhs' most significant bit is 0 or the addition lhs+rhs does not overflow.
            2) rhs is in a quantum Fourier state.

        Parameters:
            lhs (Union[QuantumRegister,int]): left-hand side.
            rhs (QRegisterPhaseLE): right-hand side AND result.
            qcirc (QuantumCircuit): the circuit on which to add the gates.
            approximation (int) : The order of approximation. All the
                              controlled phase gates with an angle inferior to
                              pi/2**approximation will not be added to the circuit.
                              If not present, take the best approximation possible.
                              See https://arxiv.org/abs/quant-ph/9601018.
        """

        used_qubits = [rhs[i] for i in range(len(rhs))]
        qubit_number = len(rhs)
        if not isinstance(lhs, int):
            used_qubits += [lhs[i] for i in range(len(lhs))]
            qubit_number = min(len(lhs), qubit_number)

        super().__init__(self.__class__.__name__, # name
                         [],                      # parameters
                         used_qubits,             # qubits
                         qcirc)                   # circuit

        # qubit_number is the number of qubits we will use, so it cost nothing to check.
        rhs.check_range(qubit_number-1)

        if isinstance(lhs, int):
            # If the value to add is a classical integer (not stored in a quantum
            # register), then we can optimise greatly the circuit.
            for i in range(qubit_number):
                self.u1(sym.pi * (lhs % 2**(i+1)) / 2**i, rhs[i])
        else:
            lhs.check_range(qubit_number-1)
            if not approximation:
                approximation = math.ceil(math.log2(qubit_number)) + 2

            for i in range(qubit_number-1, -1, -1):
                for j in range(i, -1, -1):
                    if i-j < approximation:
                        self.cu1(sym.pi / 2**(i-j), lhs[qubit_number-1-j], rhs[i])

def approximate_add_fourier_state(self,
                                  lhs: Union[int, QuantumRegister],
                                  rhs: QRegisterPhaseLE,
                                  qcirc: QuantumCircuit,
                                  approximation: int = None) -> ApproximateAddFourierStateGate:
    """Add two registers with rhs in quantum fourier state."""
    if isinstance(lhs, QuantumRegister):
        self._check_qreg(lhs)
        self._check_dups([lhs, rhs])
    self._check_qreg(rhs)
    return self._attach(ApproximateAddFourierStateGate(lhs, rhs, qcirc, approximation))

def iapproximate_add_fourier_state(self,
                                   lhs: Union[int, QuantumRegister],
                                   rhs: QRegisterPhaseLE,
                                   qcirc: QuantumCircuit,
                                   approximation: int = None) -> ApproximateAddFourierStateGate:
    """Substract two registers with rhs in quantum fourier state."""
    if isinstance(lhs, QuantumRegister):
        self._check_qreg(lhs)
        self._check_dups([lhs, rhs])
    self._check_qreg(rhs)
    return self._attach(ApproximateAddFourierStateGate(lhs, rhs, qcirc, approximation).inverse())


class AddFourierStateGate(ApproximateAddFourierStateGate):
    """Quantum Fourier state adder.

    Implements the fourier adder presented in "Addition on a Quantum Computer",
    written by Thomas G. Draper in 1998 and revised in 2000. Let F(a) be the
    quantum fourier transform of a, this class implement the gate that compute
    the transformation |a>|F(b)> -> |a>|F(b+a)>.
    """

    def __init__(self,
                 lhs: Union[int, QuantumRegister],
                 rhs: QRegisterPhaseLE,
                 qcirc: QuantumCircuit):
        """Initialise the AddFourierStateGate class.

        Implements the fourier adder presented in "Addition on a Quantum Computer",
        written by Thomas G. Draper in 1998 and revised in 2000. Let F(a) be the
        quantum fourier transform of a, this class implement the gate that compute
        the transformation |a>|F(b)> -> |a>|F(b+a)>

        Requires:
            1) lhs' and rhs' most significant bit is 0 or the addition lhs+rhs does
               not overflow.
            2) rhs is in a quantum Fourier state.

        Parameters:
            lhs (QuantumRegister): left-hand side.
            rhs (QRegisterPhaseLE): right-hand side AND result.
            qcirc (QuantumCircuit): the circuit on which to add the gates.
        """
        qubit_number = len(rhs)
        if not isinstance(lhs, int):
            qubit_number = min(qubit_number, len(lhs))
        super().__init__(lhs, rhs, qcirc, approximation=qubit_number)

def add_fourier_state(self,
                      lhs: Union[int, QuantumRegister],
                      rhs: QRegisterPhaseLE,
                      qcirc: QuantumCircuit) -> AddFourierStateGate:
    """Add two registers with rhs in quantum fourier state."""
    if not isinstance(lhs, int):
        self._check_qreg(lhs)
        self._check_dups([lhs, rhs])
    self._check_qreg(rhs)
    return self._attach(AddFourierStateGate(lhs, rhs, qcirc))

def iadd_fourier_state(self,
                       lhs: Union[int, QuantumRegister],
                       rhs: QRegisterPhaseLE,
                       qcirc: QuantumCircuit) -> AddFourierStateGate:
    """Substract two registers with rhs in quantum fourier state."""
    if not isinstance(lhs, int):
        self._check_qreg(lhs)
        self._check_dups([lhs, rhs])
    self._check_qreg(rhs)
    return self._attach(AddFourierStateGate(lhs, rhs, qcirc).inverse())
