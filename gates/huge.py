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

from qiskit import CompositeGate, QuantumRegister, QuantumCircuit

class HugeGate(CompositeGate):
    """"""
    def __init__(self,
                 qreg: QuantumRegister,
                 order: int,
                 qcirc: QuantumCircuit = None):
        """
        Parameters:
            qreg         (QuantumRegister)      quantum register.
            qcirc        (QuantumCircuit)       the associated circuit.
        """

        used_qubits = [qubit[i]
                       for qubit in [qreg]
                       for i in range(len(qubit))]

        super().__init__(self.__class__.__name__, # name
                         [],                      # parameters
                         used_qubits,             # qubits
                         qcirc)                   # circuit

        if order:
            for i in range(2):
                self._attach(HugeGate(qreg, order-1, qcirc))
        else:
            for qubit in used_qubits:
                self.x(qreg)

def huge_gate(self,
              qreg: QuantumRegister,
              order: int,
              qcirc: QuantumCircuit) -> HugeGate:
    self._check_qreg(qreg)
    return self._attach(HugeGate(qreg, order, qcirc))
