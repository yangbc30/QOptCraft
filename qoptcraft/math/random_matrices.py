"""Random haar uniform unitary."""

import numpy as np
import scipy as sp
from numpy.typing import NDArray


def haar_random_unitary(dim: int, seed: int | None = None) -> NDArray:
    """Create a random unitary matrix distributed with Haar measure.

    Args:
        dim (int): the dimension of the unitary matrix.

    Returns:
        NDArray: the haar uniform random unitary.

    References:
        The algorithm can be found in
        Francesco Mezzadri, "How to generate random matrices from the classical
        compact groups" arXiv, 2007. https://arxiv.org/abs/math-ph/0609050
    """
    rng = np.random.default_rng() if seed is None else np.random.default_rng(seed)
    Z = rng.normal(0, 1, dim * dim).reshape(dim, dim)
    Z = (Z + 1j * rng.normal(0, 1, dim * dim).reshape(dim, dim)) / np.sqrt(2.0)
    Q, R = sp.linalg.qr(Z)  # QR decomposition
    D = np.diag(R)  # diag() outputs a 1-D array
    Λ = np.diag(D / np.absolute(D))  # diag() outputs a 2-D array again
    return Q @ Λ @ Q


def random_hermitian(dim: int, seed: int | None = None) -> NDArray:
    """Create a random hermitian matrix from a random Haar unitary."""
    rng = np.random.default_rng() if seed is None else np.random.default_rng(seed)
    H = np.diag(rng.normal(0, 1, dim))
    U = haar_random_unitary(dim, seed)
    H = U @ H @ U.conj().T
    return H.round(16)
