<<<<<<< HEAD:QOptCraft/evolution/hamiltonian_evolution.py
<<<<<<<< HEAD:qoptcraft/evolution/photon_hamiltonian.py
import numpy as np
from numpy.typing import NDArray

from qoptcraft.basis import get_photon_basis
from qoptcraft.operators import annihilation, creation


def photon_hamiltonian(scattering_matrix: NDArray, photons: int) -> NDArray:
    """Lift the scattering linear optical matrix to a photonic hamiltonian.

    Args:
        scattering_matrix (NDArray): matrix of the linear optical system.
        photons (int): number of photons.

    Returns:
        spmatrix: hamiltonian lifted to the Hilbert space of m modes and n photons.
    """
    modes = scattering_matrix.shape[0]
    basis_photon = get_photon_basis(modes, photons)
    dim = len(basis_photon)

    lifted_matrix = np.zeros((dim, dim), dtype=complex)

    for col_img, fock_ in enumerate(basis_photon):
        for row in range(modes):
            for col in range(modes):
                fock, coef = annihilation(col, fock_)
                if coef == 0:
                    continue
                fock, coef_ = creation(row, fock)
                coef = coef * coef_ * scattering_matrix[row, col]
                row_img = basis_photon.index(fock)
                lifted_matrix[row_img, col_img] += coef

    return lifted_matrix
========
=======
>>>>>>> d41662bbef5ea1a918e17800a5bf945002d43f0c:qoptcraft/evolution/hamiltonian_evolution.py
import numpy as np
from numpy.typing import NDArray
import scipy as sp

from qoptcraft.basis import get_photon_basis
from qoptcraft.operators import annihilation, creation
from qoptcraft.math import logm_3


def photon_hamiltonian(scattering_matrix: NDArray, photons: int) -> NDArray:
    """Lift the scattering linear optical matrix to a photonic hamiltonian.

    Args:
        scattering_matrix (NDArray): matrix of the linear optical system.
        photons (int): number of photons.

    Returns:
        spmatrix: hamiltonian lifted to the Hilbert space of m modes and n photons.
    """
    modes = scattering_matrix.shape[0]
    basis_photon = get_photon_basis(modes, photons)
    dim = len(basis_photon)

    lifted_matrix = np.zeros((dim, dim), dtype=complex)

    for col_img, fock_ in enumerate(basis_photon):
        for row in range(modes):
            for col in range(modes):
                fock, coef = annihilation(col, fock_)
                if coef == 0:
                    continue
                fock, coef_ = creation(row, fock)
                coef = coef * coef_ * scattering_matrix[row, col]
                row_img = basis_photon.index(fock)
                lifted_matrix[row_img, col_img] += coef

    return lifted_matrix


def photon_unitary_hamiltonian(scattering_matrix: NDArray, photons: int) -> NDArray:
    """Calulate the multiphoton unitary from the scattering matrix of the interferometer
    through the hamiltonian evolution. First, the hamiltonian is calculated with the logarithm;
    next, it is evolved with the algebra homomorphism, and finally we exponentiate the evolved
    hamiltonian to get the lifted unitary.

    Note:
        Previously known as evolution_3.

    Args:
        scattering_matrix (NDArray): _description_
        photons (int): number of photons.

    Returns:
        NDArray: _description_
    """
    S_hamiltonian = logm_3(scattering_matrix)
    U_hamiltonian = photon_hamiltonian(S_hamiltonian, photons)
    return sp.linalg.expm(U_hamiltonian)
<<<<<<< HEAD:QOptCraft/evolution/hamiltonian_evolution.py
>>>>>>>> d41662bbef5ea1a918e17800a5bf945002d43f0c:qoptcraft/evolution/hamiltonian_evolution.py
=======
>>>>>>> d41662bbef5ea1a918e17800a5bf945002d43f0c:qoptcraft/evolution/hamiltonian_evolution.py
