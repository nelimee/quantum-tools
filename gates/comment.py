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

"""
Comment instruction.
"""

from qiskit import QuantumCircuit
from qiskit import Instruction
from qiskit import CompositeGate
from qiskit.extensions.standard.barrier import Barrier

class Comment(Barrier):
    """Code comment."""

    def __init__(self, text: str, qubits, circ):
        """Create new comment."""
        super().__init__(qubits, circ)
        self._text = text

    def inverse(self):
        """Do nothing. Return self."""
        return self

    def qasm(self):
        """Return OPENQASM string."""
        return "// {}".format(self._text)

    def reapply(self, circ):
        """Reapply this comment."""
        self._modifiers(circ.comment(self._text))

    def q_if(self, *qregs):
        self._text = ("c-" * len(qregs)) + self._text
        return self

def comment(self, text:str):
    """Write a comment to circuit."""
    circuit = self
    while not hasattr(circuit, 'get_qregs'):
        circuit = circuit.circuit
    qubits = [(qregister, j)
              for qregister in circuit.get_qregs().values()
              for j in range(len(qregister))
    ]
    return self._attach(Comment(text, qubits, self))


QuantumCircuit.comment = comment
CompositeGate.comment = comment
