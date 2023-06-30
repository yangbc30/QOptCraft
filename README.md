# qoptcraft

[![License: Apache 2.0](https://img.shields.io/github/license/saltstack/salt)](https://www.apache.org/licenses/LICENSE-2.0)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v1.json)](https://github.com/charliermarsh/ruff)
[![black](https://img.shields.io/badge/code%20style-black-black)](https://github.com/psf/black)


A Python package for the design and study of linear optical quantum systems.

## Documentation and examples
Documentation and examples can be found in [...] 

## Installation
Create and activate a new conda environment
```console
conda create --name test python==3.11
conda activate test
```
Then clone the optimized branch from the GitLab repository
```console
git clone -b optimized https://github.tel.uva.es/juagar/qoptcraft.git
```
Finally, step inside the qoptcraft folder and do a local install
With `pip`:
```console
cd qoptcraft
pip install .
```

## Quick usage

### Clemens and Reck decompositions

We can decompose any unitary into beamsplitters and phase shifters:
```python
from qoptcraft.optical_elements import clemens_decomposition, reck_decomposition

modes = 4
unitary = haar_random_unitary(modes)
left, diag, right = clemens_decomposition(unitary)
diag, right = reck_decomposition(unitary)
```

### Basis

We can get easily get the basis of the unitary algebra
```python
from qoptcraft.basis import get_algebra_basis

modes = 2
photons = 3
basis_algebra, basis_image_algebra = get_algebra_basis(modes, photons)
```

or the Fock state basis of the Hilbert space
```python
from qoptcraft.basis import get_photon_basis, hilbert_dim

photon_basis = get_photon_basis(modes, photons)
dimension = hilbert_dim(modes, photons)  # should equal len(photon_basis)
```

### States

We can create pure quantum states by summing Fock states:
```python
from math import sqrt

from qoptcraft.state import Fock

in_state = Fock(1, 1, 0, 0)
bell_state = 1 / sqrt(2) * Fock(1, 0, 1, 0) + 1 / sqrt(2) * Fock(0, 1, 0, 1)
```

### Invariants

To check if transitions between quantum states are forbidden by a linear optical transformation, we simply run
```python
from qoptcraft.invariant import can_transition, photon_invariant

can_transition(in_state, bell_state, method="reduced")

>>> False
```

The invariant can be calculated from density matrices (calculations use the basis of the algebra)
To check if transitions between quantum states are forbidden by a linear optical transformation, we simply run
```python
from qoptcraft.state import MixedState
from qoptcraft.invariant import can_transition, photon_invariant

mixed_state = MixedState.from_mixture(pure_states=[in_fock, bell_state], probs=[0.5, 0.5])
can_transition(mixed_state, bell_state, method="basis")

>>> False
```

### Quantizing linear interferomenters

We can easily compute the unitary matrix associated with a linear interferometer S and a certain number of photons. There are four different methods to compute the unitary: `'heisenberg'`, `'hamiltonian'`, `'permanent glynn'` and `'permanent ryser'`.

```python
from qoptcraft.operators import haar_random_unitary
from qoptcraft.evolution import photon_unitary

interferometer = haar_random_unitary(modes)
unitary_heisenberg = photon_unitary(interferometer, photons, method="heisenberg")
unitary_hamiltonian = photon_unitary(interferometer, photons, method="hamiltonian")
unitary_glynn = photon_unitary(S, photons, method="permanent glynn")
unitary_ryser = photon_unitary(S, photons, method="permanent ryser")
```

We can apply this function to a 50:50 beamsplitter to recover the Hong-Ou-Mandel matrix

```python
from numpy import pi as PI
from qoptcraft.optical_elements import beam_splitter

bs_matrix = beam_splitter(angle=PI/4, shift=0, dim=2, mode_1=0, mode_2=1, convention="clemens")
hong_ou_mandel = photon_unitary(bs_matrix, photons=3, method="heisenberg")
```


### Approximating a unitary with linear optics (Topogonov)
```python
from qoptcraft.operators import qft
from qoptcraft.topogonov import topogonov

unitary = qft(6)
approx_unitary, error = toponogov(unitary, modes, photons)
```


## Citing

qoptcraft is the work of Daniel Gómez Aguado and Pablo V. Parellada. 

If you are doing research using qoptcraft, please cite our paper:

    Daniel Gómez Aguado et al. qoptcraft: A Python package for the design and study of linear optical quantum systems. 2023. https://doi.org/10.1016/j.cpc.2022.108511


## References

[1] W. R. Clements, P. C. Humphreys, B. J. Metcalf, W. S. Kolthammer, and I. A. Walsmley, ”Optimal Design for Universal Multiport Interferometers”, Optica 3, 1460 (2016).

[2] J. Skaar, J. C. García Escartín, and H. Landro, ”Quantum mechanical description of linear optic”, American Journal of Physics 72, 1385 (2004).

[3] S. Scheel, ”Permanents in linear optics network”, Acta Physica Slovaca 58, 675 (2008).

[4] ”Permanents and Ryser’s algorithm”, numbersandshapes.net.

[5] J. C. García Escartín, V. Gimeno, and J. J. Moyano-Fern ´andez, ”Multiple photon effective Hamiltonians in linear quantum optical networks”, Optics Communications 430 (2019) 434–439.

[6] J. C. García Escartín, V. Gimeno, and J. J. Moyano Fern ´andez, ”A method to determine which quantum operations can be realized with linear optics with a constructive implementation recipe”, Physical Review A 100, 022301 (2019).

[7] J. C. García Escartín and J. J. Moyano Fern ´andez, ”Optimal approximation to unitary quantum operators with linear optics”, arXiv:2011.15048v1 [quant-ph].

[8] N. Tischler, C. Rockstuhl, and K. Slowik, ”Quantum Optical Realization of Arbitrary Linear Transformations Allowing for Loss and Gain”, Physical Review X 8, 021017 (2018).

[9] T. A. Loring, ”Computing a logarithm of a unitary matrix with general spectrum”, Numerical Linear Algebra wth Applications, 21 (6) 744–760 (2014).


## Contributing

We appreciate and welcome contributions. For major changes, please open an issue first
to discuss what you would like to change. Also, make sure to update tests as appropriate.

If you are new to contributing to open source, [this guide](https://opensource.guide/how-to-contribute/) helps explain why, what, and how to get involved.

## License

This software is under the [Apache License 2.0](https://choosealicense.com/licenses/apache-2.0/).


# OLD README


Author: Daniel Gómez Aguado
e-mail: gomezaguado99@gmail.com

QOprCraft is a software package or library developed in Python 3, designed for quantum experiments building via linear optics instruments. 
Divided in ten functions (ver 1.1). It navigates through the subalgebras of mainly unitary matrices.
1) Via decomposing files into beam splitters and shifters, theoretically desired matrices can be implemented (Selements()), 
2) as well as their quantum evolution (read: influence) given a number of photons (StoU()).
3) The reverse operation: giving a desired evolution for finding the original matrix, is also true (SfromU()).
4) For unavailable implementations, the option to obtain its closest, possible evolution matrix is also given via Toponogov().
5) Systems with loss are treated by the fifth function QuasiU().
6) Measurement via computing the Schmidt rank of Fock states vectors after passing a quantum through a given linear optics circuit (StateSchmidt()).
A matrix generator, a test of some algorithms and every logarithm developed are also included as individual functions.

"qoptcraft()" is the main function, making full use of all the algorithms available. 
Its standalone subfunctions or blocks (read user guide) can be deployed on their own as well.
Use the module parameter (1-10) for picking which subfunction to use: 
> Selements (module=1)
> StoU (module=2)
> SfromU (module=3)
> Toponogov (module=4)
> QuasiU (module=5)
> StateSchmidt (module=6)
> iHStoiHU (module=7).
> Use the choice parameter for subsubfunctions in QOCGen (module=8, choice=1-7), QOCTest (module=9, choice=1-3) or QOCLog (module=10, choice=1-5).
More info on the remaining parameters by reading qoptcraft's user guide.
