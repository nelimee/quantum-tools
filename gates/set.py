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

"""This module provide a convenient function for quantum regiter initialisation."""

from qiskit import QuantumCircuit, QuantumRegister
from utils.endianness import apply_LE_operation, QRegisterBase

def qset(qcirc: QuantumCircuit,
         qreg: QRegisterBase,
         bits_to_set: int,
         start: int = 0) -> None:
    """Set qubits in qreg according to N.

    This method sets the qubits in qreg starting from qreg[start] according to
    the value of the bits in bits_to_set. This method will apply a X gate
    to all the qubits qreg[start + i] iff the i-th bit of bits_to_set is set to 1.
    """

    def little_endian_operation(local_gate_container, local_quantum_register):
        little_endian_binary = bin(bits_to_set)[2:][::-1]
        for i, binary_digit in enumerate(little_endian_binary):
            if binary_digit == '1':
                local_gate_container.x(local_quantum_register[start+i])

    apply_LE_operation(qcirc, little_endian_operation, qreg)
