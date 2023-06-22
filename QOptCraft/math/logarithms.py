"""Logarithms of matrices.

Todo:
    Reference papers of each logarithm.

"""
import numpy as np
from numpy.linalg import inv
from numpy.typing import NDArray
from scipy.linalg import logm, schur, sqrtm


def logm_1(matrix: NDArray) -> NDArray:
    """Logarithm of matrix.

    Args:
        matrix (NDArray): square matrix.

    Returns:
        NDArray: logarithm.
    """
    dim = len(matrix)
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    diagonal = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        diagonal[i, i] = eigenvalues[i] / np.abs(eigenvalues[i])
    H = eigenvectors.dot(logm(diagonal).dot(np.linalg.inv(eigenvectors)))
    return 0.5 * (H + np.transpose(np.conj(H)))


def logm_2(matrix: NDArray) -> NDArray:
    """Logarithm of matrix.

    Args:
        matrix (NDArray): square matrix.

    Returns:
        NDArray: logarithm.
    """
    return 0.5 * (logm(matrix) + np.transpose(np.conj(logm(matrix))))


def logm_3(matrix: NDArray) -> NDArray:
    """Logarithm of matrix using Schur's decomposition.

    Args:
        matrix (NDArray): square matrix.

    Returns:
        NDArray: logarithm.
    """
    U, Q = schur(matrix)
    dim = len(matrix)
    D = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        D[i, i] = U[i, i] / abs(U[i, i])
    log = Q.dot(logm(D).dot(np.transpose(np.conj(Q))))
    return log


def logm_4(matrix: NDArray) -> NDArray:
    """Logarithm of matrix using Schur's decomposition.

    Args:
        matrix (NDArray): square matrix.

    Returns:
        NDArray: logarithm.
    """
    V = matrix.dot(inv(sqrtm(np.transpose(np.conj(matrix)).dot(matrix))))
    U, Q = schur(V)
    dim = len(matrix)
    D = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        D[i, i] = U[i, i] / abs(U[i, i])
    return Q.dot(logm(D).dot(np.transpose(np.conj(Q))))


def logm_5(matrix: NDArray) -> NDArray:
    """Logarithm of matrix using Schur's decomposition.

    Args:
        matrix (NDArray): square matrix.

    Returns:
        NDArray: logarithm.
    """
    V1 = (matrix + np.transpose(np.conj(inv(matrix)))) / 2.0
    V = (V1 + np.transpose(np.conj(inv(V1)))) / 2.0
    U, Q = schur(V)
    dim = len(matrix)
    D = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        D[i, i] = U[i, i] / abs(U[i, i])
    return Q.dot(logm(D).dot(np.transpose(np.conj(Q))))
