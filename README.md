# QOptCraft
[![documentation](https://img.shields.io/badge/docs-mkdocs%20material-blue.svg?style=flat)](https://pablovegan.github.io/QOptCraft/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8167528.svg)](https://doi.org/10.5281/zenodo.8167528)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v1.json)](https://github.com/charliermarsh/ruff)
[![black](https://img.shields.io/badge/code%20style-black-black)](https://github.com/psf/black)


A Python package for the design and study of linear optical quantum systems.

## Documentation
Documentation and examples can be found [here](https://pablovegan.github.io/QOptCraft/).


## Installation
Create and activate a new conda environment
```console
conda create --name test python==3.11
conda activate test
```

Install with `pip`:
```console
pip install qoptcraft
```
or, for the latest version,
```console
pip install git+https://github.tel.uva.es/juagar/qoptcraft.git
```

## Quick usage

### Clemens and Reck decompositions

We can decompose any unitary into beamsplitters and phase shifters:
```python
from qoptcraft.optical_elements import clemens_decomposition, reck_decomposition
from qoptcraft.operators import haar_random_unitary

modes = 4
unitary = haar_random_unitary(modes)
left, diag, right = clemens_decomposition(unitary)
diag, right = reck_decomposition(unitary)
```

### Basis

We can get easily get the basis of the unitary algebra
```python
from qoptcraft.basis import unitary_algebra_basis, image_algebra_basis

modes = 2
photons = 3
basis_algebra = unitary_algebra_basis(modes)
basis_image_algebra = image_algebra_basis(modes, photons)
```

or the Fock state basis of the Hilbert space
```python
from qoptcraft.basis import photon_basis, hilbert_dim

photon_basis = photon_basis(modes, photons)
dimension = hilbert_dim(modes, photons)  # should equal len(photon_basis)
```

### States

We can create pure quantum states by summing Fock states:
```python
from math import sqrt

from qoptcraft.state import Fock

in_fock = Fock(1, 1, 0, 0)
bell_state = 1 / sqrt(2) * Fock(1, 0, 1, 0) + 1 / sqrt(2) * Fock(0, 1, 0, 1)
```

### Invariants

To check if transitions between quantum states are forbidden by a linear optical transformation, we simply run
```python
from qoptcraft.invariant import forbidden_transition, photon_invariant

forbidden_transition(in_fock, bell_state, method="reduced")
```
```console
>>> True
```

The invariant can be calculated from density matrices (calculations use the basis of the algebra)
To check if transitions between quantum states are forbidden by a linear optical transformation, we simply run
```python
from qoptcraft.state import MixedState
from qoptcraft.invariant import forbidden_transition, photon_invariant

mixed_state = MixedState.from_mixture(pure_states=[in_fock, bell_state], probs=[0.5, 0.5])
forbidden_transition(mixed_state, bell_state, method="basis")
```
```console
>>> True
```

### Quantizing linear interferomenters

We can easily compute the unitary matrix associated with a linear interferometer S and a certain number of photons. There are four different methods to compute the unitary: `'heisenberg'`, `'hamiltonian'`, `'permanent glynn'` and `'permanent ryser'`.

```python
from qoptcraft.operators import haar_random_unitary
from qoptcraft.evolution import photon_unitary

modes = 2
photons = 3
interferometer = haar_random_unitary(modes)
unitary_heisenberg = photon_unitary(interferometer, photons, method="heisenberg")
unitary_hamiltonian = photon_unitary(interferometer, photons, method="hamiltonian")
unitary_glynn = photon_unitary(interferometer, photons, method="permanent glynn")
unitary_ryser = photon_unitary(interferometer, photons, method="permanent ryser")
```

We can apply this function to a 50:50 beamsplitter to recover the Hong-Ou-Mandel matrix

```python
from numpy import pi as PI
from qoptcraft.optical_elements import beam_splitter

bs_matrix = beam_splitter(angle=PI/4, shift=0, dim=2, mode_1=0, mode_2=1, convention="clemens")
hong_ou_mandel = photon_unitary(bs_matrix, photons=3, method="heisenberg")
```

### Retrieve the linear optical scattering matrix from the quantized unitary
If a given unitary matrix comes from a linear optical scattering matrix, we can retrieve it
```python
from qoptcraft import haar_random_unitary, photon_unitary, scattering_from_unitary

modes = 3
photons = 2
S = haar_random_unitary(modes)
U = photon_unitary(S, photons)
S_rebuilt = scattering_from_unitary(U, modes, photons)
```
If this scattering matrix doesn't exist, it will raise an `InconsistentEquations` error.
### Approximating a unitary with linear optics (Toponogov)
```python
from qoptcraft.operators import qft
from qoptcraft.toponogov import toponogov

modes = 3
photons = 2
unitary = qft(6)
approx_unitary, error = toponogov(unitary, modes, photons)
```

### Version 1.1

Functions from the version 1.1 of QOptCraft can be accessed from the submodule `_legacy`
```python
from qoptcraft._legacy import *
```


## Authors

Versions 1.0 and 1.1 were developed by Daniel Gómez Aguado (gomezaguado99@gmail.com), 2021. Update to version 2.0 has been developed by Pablo V. Parellada (pablo.veganzones@uva.es), 2023.


## Citing

If you are doing research using qoptcraft, please cite our paper:

    Daniel Gómez Aguado et al. qoptcraft: A Python package for the design and study of linear optical quantum systems. 2023. https://doi.org/10.1016/j.cpc.2022.108511


## References

[1] W. R. Clements, P. C. Humphreys, B. J. Metcalf, W. S. Kolthammer, and I. A. Walsmley, ”Optimal Design for Universal Multiport Interferometers”, Optica 3, 1460 (2016).

[2] J. Skaar, J. C. García Escartín, and H. Landro, ”Quantum mechanical description of linear optic”, American Journal of Physics 72, 1385 (2004).

[3] S. Scheel, ”Permanents in linear optics network”, Acta Physica Slovaca 58, 675 (2008).

[4] ”Permanents and Ryser’s algorithm”, numbersandshapes.net.

[5] J. C. García Escartín, V. Gimeno, and J. J. Moyano-Fernández, ”Multiple photon effective Hamiltonians in linear quantum optical networks”, Optics Communications 430 (2019) 434–439.

[6] J. C. García Escartín, V. Gimeno, and J. J. Moyano Fernández, ”A method to determine which quantum operations can be realized with linear optics with a constructive implementation recipe”, Physical Review A 100, 022301 (2019).

[7] J. C. García Escartín and J. J. Moyano Fernández, ”Optimal approximation to unitary quantum operators with linear optics”, arXiv:2011.15048v1 [quant-ph].

[8] N. Tischler, C. Rockstuhl, and K. Slowik, ”Quantum Optical Realization of Arbitrary Linear Transformations Allowing for Loss and Gain”, Physical Review X 8, 021017 (2018).

[9] T. A. Loring, ”Computing a logarithm of a unitary matrix with general spectrum”, Numerical Linear Algebra wth Applications, 21 (6) 744–760 (2014).


## Contributing

We appreciate and welcome contributions. For major changes, please open an issue first
to discuss what you would like to change. Also, make sure to update tests as appropriate.

If you are new to contributing to open source, [this guide](https://opensource.guide/how-to-contribute/) helps explain why, what, and how to get involved.

## Acknowledgment

The development of the code for the version 2.0 has been supported by the European Union.-Next Generation UE/MICIU/Plan de Recuperación, Transformación y Resiliencia/Junta de Castilla y León.


## License

This software is under the [Apache License 2.0](https://choosealicense.com/licenses/apache-2.0/).
