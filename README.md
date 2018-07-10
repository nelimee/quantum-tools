This repository host the work I done with QISKit during my Master's thesis. I shared the code according to the CeCILL-B license (see [the license file](LICENSE.txt)).

# Installation

The installation procedure is not really standard and will be changed when I will have the time. At the moment, you **need** to clone the repository in a folder called `utils` because all the imports in the code assume that the folder is called `utils`. Then you need to update your PYTHONPATH. The procedure is summarised below:

 1. Clone the repository as `utils`:

 ```bash
 git clone https://github.com/nelimee/quantum-tools.git utils
 ```

 2. Update your PYTHONPATH to point to the parent folder of `utils`:
 ```bash
 export PYTHONPATH="/parent/folder/of/utils/:${PYTHONPATH}"
 ```
 Instead of updating the PYTHONPATH at each login, you can add the previous line to your `.bashrc`:
 ```bash
 echo 'export PYTHONPATH="/parent/folder/of/utils/:${PYTHONPATH}"' >> ~/.bashrc
 ```

# Directory structure

## `HHL` folder

As its name indicates, this folder is dedicated to scripts related to the HHL algorithm. In this folder you have:

 1. [4x4_system.py](HHL/4x4_system.py): An implementation of the HHL algorithm for a 4x4 matrix. The implementation is based on [Quantum Circuit Design for Solving Linear Systems of Equations (v2)](https://arxiv.org/abs/1110.2232v2) but corrects some errors in the original paper.
 2. [hamiltonian_error.py](HHL/hamiltonian_error.py): A script that will compute the errors in the Hamiltonian simulation procedure and that will output useful information on them. The error is the norm of the difference between the simulated matrix and the "exact" matrix (up to floating-point errors in the numerical methods used).
 3. [hamiltonian_unitary_test.py](HHL/hamiltonian_unitary_test.py): A convenient script that I used to verify the validity of the different steps of HHL by computing the error between the expected unitary matrix and the simulated one. The script has not been formatted to be user-friendly.
 4. [optim_hamil.py](HHL/optim_hamil.py): A script used to compute the best coefficients for the powers of the Hamiltonian simulation procedure. As written in the comments, there is no proof of "why it works?", but the Hamiltonian powers are simulated with a really low error.
 5. [test.py](HHL/test.py): A scratch file that contains an example of the usage of the `qiskit.extensions.quantum_initializer._initializer.initialize` function.

## `error_analysis` folder

This folder links to an other git repository that contains helpers to estimate the errors due to the physical imprecisions of the quantum chips. The code is still draft and does not work as expected for the moment.

## `gates` folder

Each file in this folder declare some personalised quantum gates.

## `visualisation` folder

This folder links to an other git repository: [qasm2image](https://github.com/nelimee/qasm2image).

# Usage

 The module is installed under the name `utils`. You can import the `qpe` gate using
 ```python
 import utils.gates.qpe
 ```

 You can also launch the scripts in `HHL/`
