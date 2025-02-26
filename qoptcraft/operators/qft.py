import numpy as np
from numpy.typing import NDArray


def qft(dim: int) -> NDArray:
    """Matrix of the Quantum Fourier Transform.

    Args:
        dim (int): dimension of the matrix.

    Returns:
        NDArray: matrix of the transform.
    """
    matrix = np.zeros((dim, dim), dtype=np.complex128)
    frequency = 2j * np.pi / dim
    for i in range(dim):
        for j in range(dim):
            matrix[i, j] = np.exp(frequency * i * j)
    return matrix / np.sqrt(dim)


def qft_inv(dim: int) -> NDArray:
    """Matrix of the inverse of the Quantum Fourier Transform.

    Args:
        dim (int): dimension of the matrix.

    Returns:
        NDArray: matrix of the transform.
    """
    matrix = np.zeros((dim, dim), dtype=np.complex128)
    frequency = -2j * np.pi / dim
    for i in range(dim):
        for j in range(dim):
            matrix[i, j] = np.exp(frequency * i * j)
    return matrix / np.sqrt(dim)
