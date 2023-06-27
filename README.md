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




## References




## Citing

qoptcraft is the work of Daniel Gómez Aguado and Pablo V. Parellada. 

If you are doing research using qoptcraft, please cite our paper:

    Daniel Gómez Aguado et al. qoptcraft: A Python package for the design and study of linear optical quantum systems. 2023. https://doi.org/10.1016/j.cpc.2022.108511


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
