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

"""This module contains an helper gate for measurement."""

from utils.endianness import QRegisterBase, QRegisterLE, QRegisterBE, \
    CRegisterBase, CRegister

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

def measure(qcirc: QuantumCircuit,
            qreg: QuantumRegister,
            creg: ClassicalRegister) -> None:

    qubits_number = len(qreg)
    clbits_number = len(creg)
    number_of_bits_to_measure = min(qubits_number, clbits_number)

    # The classical register is always in little endian because QISKit
    # read values in little endian.
    # So we need to check if the quantum register is in the same endianness
    # or not.
    if isinstance(qreg, QRegisterLE):
        def index(i:int):
            return i
    else:
        def index(i:int):
            return number_of_bits_to_measure-1-i

    for i in range(number_of_bits_to_measure):
        qcirc.measure(qreg[i], creg[index(i)])
