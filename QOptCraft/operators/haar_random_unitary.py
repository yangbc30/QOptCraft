"""Random haar uniform unitary.
"""

import numpy as np
from numpy.typing import NDArray
import scipy as sp


def haar_random_unitary(dim: int) -> NDArray:
    """
    Create a random unitary matrix distributed with Haar measure

    Parameters
    ----------
    dim: int
        the dimension of the unitary matrix

    Returns
    -------
    array_like
        the haar uniform random unitary

    References
    ----------
    The algorithm can be found in [1]_.

    .. [1] Francesco Mezzadri, "How to generate random matrices from the classical
        compact groups" arXiv, 2007. https://arxiv.org/abs/math-ph/0609050
    """
    rng = np.random.default_rng()
    Z = rng.normal(0, 1, dim * dim).reshape(dim, dim)
    Z = (Z + 1j * rng.normal(0, 1, dim * dim).reshape(dim, dim)) / np.sqrt(2.0)
    Q, R = sp.linalg.qr(Z)  # QR decomposition
    D = np.diag(R)  # diag() outputs a 1-D array
    Λ = np.diag(D / np.absolute(D))  # diag() outputs a 2-D array again
    return Q @ Λ @ Q
