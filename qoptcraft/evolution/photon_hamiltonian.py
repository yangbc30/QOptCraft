import numpy as np
from numpy.typing import NDArray

from qoptcraft.basis import photon_basis
from qoptcraft.operators import annihilation_fock, creation_fock


def photon_hamiltonian(scattering_hamiltonian: NDArray, photons: int) -> NDArray:
    """Lift the scattering linear optical matrix to a photonic hamiltonian.

    Args:
        scattering_hamiltonian (NDArray): hamiltonian of the classical linear optical system.
        photons (int): number of photons.

    Returns:
        spmatrix: hamiltonian lifted to the Hilbert space of m modes and n photons.
    """
    modes = scattering_hamiltonian.shape[0]
    basis_photon = photon_basis(modes, photons)
    dim = len(basis_photon)

    lifted_matrix = np.zeros((dim, dim), dtype=complex)

    for col_img, fock_ in enumerate(basis_photon):
        for row in range(modes):
            for col in range(modes):
                fock, coef = annihilation_fock(col, fock_)
                if coef == 0:
                    continue
                fock, coef_ = creation_fock(row, fock)
                coef = coef * coef_ * scattering_hamiltonian[row, col]
                row_img = basis_photon.index(fock)
                lifted_matrix[row_img, col_img] += coef

    return lifted_matrix
