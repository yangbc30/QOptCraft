"""Use Topogonov's theorem to get a scattering matrix that approximates
a given unitary.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm

from QOptCraft.operators import haar_random_unitary
from QOptCraft.basis import hilbert_dim, get_photon_basis, get_algebra_basis
from QOptCraft.math import gram_schmidt, mat_inner_product, mat_norm, Logm3M

from QOptCraft.evolution._2_3rd_evolution_method import evolution_3


def toponogov(matrix: NDArray, modes: int, photons: int) -> NDArray:
    """Use Topogonov's theorem to approximate a given unitary using linear optics.

    Args:
        matrix (NDArray): unitary matrix to approximate.
        modes (int): number of modes.
        photons (int): number of photons.

    Raises:
        ValueError: Matrix dimension doesn't match the number of modes and photons.

    Returns:
        NDArray: approximated unitary.
    """
    dim = len(matrix)
    if dim != hilbert_dim(modes, photons):
        raise ValueError(f"Matrix {dim = } doesn't match with {photons = } and {modes = }.")

    S_rand = haar_random_unitary(modes)
    photon_basis = get_photon_basis(modes, photons)

    basis, basis_image = get_algebra_basis(modes, photons)
    basis_image = gram_schmidt(basis_image)

    S_rand = haar_random_unitary(modes)
    unitary = evolution_3(S_rand, photons, photon_basis)[0]  # initialize approximation U_0

    error: float = mat_norm(matrix - unitary)
    error_prev: float = 0

    while np.abs(error - error_prev) > 1e-8:
        unitary_inv = np.linalg.inv(unitary)
        log_unitary = logm_3(unitary_inv.dot(matrix))

        log_projected = np.zeros_like(unitary)  # Initialize to 0
        for basis_matrix in basis_image:
            coef = mat_inner_product(log_unitary, basis_matrix)
            log_projected += coef * basis_matrix

        unitary = unitary.dot(expm(log_projected))

        error_prev = error
        error = mat_norm(matrix - unitary)

    return unitary
