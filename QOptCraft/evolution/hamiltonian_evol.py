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
