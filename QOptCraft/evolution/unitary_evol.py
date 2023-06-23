import numpy as np
from numpy.typing import NDArray
import numba

from qoptcraft.basis import get_photon_basis, hilbert_dim
from qoptcraft.math import logm_3
from qoptcraft.evolution import photon_hamiltonian


def photon_unitary_hamiltonian(scattering_matrix: NDArray, photons: int) -> NDArray:
    """Previously known as evolution_3.

    Args:
        scattering_matrix (NDArray): _description_
        photons (int): _description_

    Returns:
        NDArray: _description_
    """
    S_hamiltonian = logm_3(scattering_matrix)
    U_hamiltonian = photon_hamiltonian(S_hamiltonian, photons)
    return np.expm(U_hamiltonian)


@numba.jit(nopython=True)
def in_out_photon_matrix(matrix: NDArray, in_fock: list[int], out_fock: list[int]) -> NDArray:
    """
    Return a matrix with row index 'i' repeated 'row_rep' times
    and column index 'j' repeated 'col_rep' times.

    Parameters
    ----------
    matrix : (m, m) array
        Linear optical scattering matrix with m modes.

    Returns
    -------
    (n, n) array
        Optical scattering matrix with m modes and n photons.

    """
    # assert sum(in_fock) == sum(out_fock), "Error: number of photons must be conserved."
    photons = sum(in_fock)
    final_matrix = np.empty((photons, photons))  # * Correct size? I think so...
    in_photon_count = 0
    in_photon_idx = 0
    cumulative_in_fock = np.cumsum(in_fock)
    cumulative_out_fock = np.cumsum(out_fock)

    for i in range(photons):
        while cumulative_in_fock[in_photon_idx] <= in_photon_count:
            in_photon_idx += 1

        out_photon_idx = 0
        out_photon_count = 0
        for j in range(photons):
            while cumulative_out_fock[out_photon_idx] <= out_photon_count:
                out_photon_idx += 1

            final_matrix[i, j] = matrix[in_photon_idx, out_photon_idx]

            out_photon_count += 1
        in_photon_count += 1

    return final_matrix


def photon_unitary_permanent(scattering_matrix: NDArray, photons: int) -> NDArray:
    """<S|phi(U)|T> = Per(U_ST) / sqrt(s1! ...sm! t1! ... tm!)"""
    ...
