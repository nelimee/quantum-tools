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

"""[OUTDATED] This module provide a class to bind QuantumCircuit and QuantumProgram.

This class has not been updated for a while, and QISKit 0.5 may broke it.
With the progressive deprecation of QuantumProgram, this class may become
useless in a near future.
"""

from qiskit import QuantumCircuit, QuantumProgram
from utils.register import QRegisterBase, BondableQuantumRegister
from utils.endianness import QRegisterBE, QRegisterLE, CRegister

class QuantumUnit(QuantumCircuit, QuantumProgram):

    def __init__(self, circuit_name:str):
        # 1. Creating the quantum program
        QuantumProgram.__init__(self)
        # 2. Creating a quantum circuit and update self
        qcirc = self.create_circuit(circuit_name, [], [])
        for circ_entry in qcirc.__dict__:
            self.__dict__[circ_entry] = qcirc.__dict__[circ_entry]
        # 3. Additional member variables
        self._circuit_name = circuit_name
        self._ancilla_qubits = None
        self._ancillas_size = []
        self._api_set = False
        self._backend = None
        self._qobj = None
        self._qresult = None

    def set_api(self) -> bool:
        try:
            import Qconfig
        except ImportError:
            pass
        else:
            super().set_api(Qconfig.APItoken, Qconfig.config['url']) # set the APIToken and API url
            self._api_set = True
        return self._api_set

    def _add_quantum_register(self, name: str, size: int) -> QRegisterBase:
        qreg = QRegisterBase(self, name, size)
        self.add(qreg)
        return qreg

    def add_BE_quantum_register(self, name: str, size: int):
        return QRegisterBE(self._add_quantum_register(name, size))

    def add_LE_quantum_register(self, name: str, size: int):
        return QRegisterLE(self._add_quantum_register(name, size))


    def add_classical_register(self, name: str, size: int):
        creg = CRegister(self, name, size)
        self.add(creg)
        return creg

    def _get_ancilla(self, size: int) -> QRegisterBase:
        number_of_taken_ancillas = sum(self._ancillas_size)
        if self._ancilla_qubits is not None:
            number_of_disponible_ancillas = len(self._ancilla_qubits) - number_of_taken_ancillas
        else:
            number_of_disponible_ancillas = 0
        if size > number_of_disponible_ancillas:
            # Then allocate enough ancilla qubits
            ancilla = self._add_quantum_register("quantumUnitAncilla"+str(len(self._ancilla_qubits or [])),
                                                 size - number_of_disponible_ancillas)
            if self._ancilla_qubits is None:
                self._ancilla_qubits = BondableQuantumRegister(ancilla)
            else:
                self._ancilla_qubits += ancilla

        # Now we know for sure that we have enough qubits.
        start = number_of_taken_ancillas
        end = start + size
        self._ancillas_size.append(size)
        return QRegisterBase(self._ancilla_qubits[start:end])

    def get_ancilla_BE(self, size: int) -> QRegisterBE:
        return QRegisterBE(self._get_ancilla(size))
    def get_ancilla_LE(self, size: int) -> QRegisterLE:
        return QRegisterLE(self._get_ancilla(size))

    def free_last_ancilla(self):
        del self._ancillas_size[-1]

    def execute(self, **kwargs):
        if 'backend' in kwargs:
            return QuantumProgram.execute(self, [self._circuit_name], **kwargs)
        elif self._backend is not None:
            return QuantumProgram.execute(self, [self._circuit_name],
                                          backend=self._backend, **kwargs)
        else:
            backend = self.get_best_backend(local=True)
            return QuantumProgram.execute(self, [self._circuit_name],
                                          backend=backend, **kwargs)

    def compile(self, **kwargs):
        if 'backend' in kwargs:
            self._qobj = QuantumProgram.compile(self, [self._circuit_name],
                                                **kwargs)
        elif self._backend is not None:
            self._qobj = QuantumProgram.compile(self, [self._circuit_name],
                                                backend=self._backend, **kwargs)
        else:
            backend = self.get_best_backend(local=True)
            self._qobj = QuantumProgram.compile(self, [self._circuit_name],
                                                backend=backend, **kwargs)
        return self._qobj

    def get_compiled_qasm(self):
        if self._qobj is None:
            self.compile()
        return QuantumProgram.get_compiled_qasm(self, self._qobj, self._circuit_name)

    def qasm(self):
        return QuantumCircuit.qasm(self)

    def run(self, **kwargs):
        if self._qobj is None:
            self.compile()
        self._qresult = QuantumProgram.run(self, self._qobj, **kwargs)
        print(self._qresult)
        return self._qresult

    def get_job_ID(self):
        if self._qobj is None:
            self.compile()
        return self._qobj['id']

    def get_job_results(self, limit: int = 50):
        if not self._api_set:
            self.set_api()

        return QuantumProgram.get_api(self).get_jobs(limit=limit)

    def get_done_job_results(self, limit: int = 50):
        return [j for j in self.get_job_results(limit) if j['status']=='COMPLETED']

    def retrieve_results_from_ID(self, job_ID: str):
        return QuantumProgram.get_api(self).get_job(job_ID)

    def get_available_backends(self):
        if not self._api_set:
            self.set_api()
        return QuantumProgram.available_backends(self)

    def get_backend_status(self, backend: str = None):
        if not self._api_set:
            self.set_api()
        backends = [backend] if backend is not None else self.get_available_backends()
        return [QuantumProgram.get_backend_status(self, back) for back in backends]

    def set_backend(self, backend: str):
        self._backend = backend

    def get_best_backend(self, local: bool = False):
        if local:
            return 'local_qasm_simulator', -1
        backend = min([back for back in self.get_backend_status()
                       if 'pending_jobs' in back and not back.get('busy', True)],
                      key=lambda backend_status: backend_status['pending_jobs'])
        return backend['backend'], backend['pending_jobs']

    def get_available_qubit_number(self):
        if self._backend is None:
            raise RuntimeError("You should select a backend before asking for its qubit number.")
        if "local" in self._backend:
            return 16
        return QuantumProgram.get_backend_configuration(self, self._backend)['n_qubits']
