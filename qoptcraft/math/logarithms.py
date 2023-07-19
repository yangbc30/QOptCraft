"""Logarithms of matrices.

References:
    Algorithms can be found in
        T.A. Loring, Numer. Linear Algebra Appl. 21 (6) (2014) 744-760.
        https://arxiv.org/abs/1203.6151
"""
from typing import Literal

import numpy as np
from numpy.linalg import inv
from numpy.typing import NDArray
from scipy.linalg import logm, schur, sqrtm


def log_matrix(
    matrix: NDArray,
    method: Literal[
        "diagonalization", "symmetrized", "schur", "polar", "newton"
    ] = "diagonalization",
) -> NDArray:
    """Logarithm of matrix via symmetrized diagonalization.

    Args:
        matrix (NDArray): square matrix.

    Returns:
        NDArray: matrix logarithm.
    """
    if method == "diagonalization":
        return log_matrix_diag(matrix)
    if method == "symmetrized":
        return log_matrix_sym(matrix)
    if method == "schur":
        return log_matrix_schur(matrix)
    if method == "polar":
        return log_matrix_polar_schur(matrix)
    if method == "newton":
        return log_matrix_newton_schur(matrix)
    raise ValueError(
        "Values for method are 'diagonalization', 'symmetrized', 'schur', 'polar' or 'newton'."
    )


def log_matrix_diag(matrix: NDArray) -> NDArray:
    """Logarithm of matrix via symmetrized diagonalization.

    Args:
        matrix (NDArray): square matrix.

    Returns:
        NDArray: matrix logarithm.
    """
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    diag = np.diag(eigenvalues / np.abs(eigenvalues))
    log_matrix = eigenvectors @ logm(diag) @ inv(eigenvectors)
    return (log_matrix + log_matrix.conj().T) / 2


def log_matrix_sym(matrix: NDArray) -> NDArray:
    """Symmetrized logarithm of matrix.

    Args:
        matrix (NDArray): square matrix.

    Returns:
        NDArray: matrix logarithm.
    """
    log_matrix = logm(matrix)
    return (log_matrix + log_matrix.conj().T) / 2


def log_matrix_schur(matrix: NDArray) -> NDArray:
    """Logarithm of matrix using Schur's decomposition. Used to diagonalize
    nearly unitary matrices.

    Args:
        matrix (NDArray): square matrix.

    Returns:
        NDArray: matrix logarithm.
    """
    U, Q = schur(matrix)
    diag = np.diag(np.diag(U) / np.abs(np.diag(U)))
    return Q @ logm(diag) @ Q.conj().T


def log_matrix_polar_schur(matrix: NDArray) -> NDArray:
    """Logarithm of matrix using Polar and Schur's decomposition. Used to
    diagonalize nearly unitary matrices.

    Args:
        matrix (NDArray): square matrix.

    Returns:
        NDArray: matrix logarithm.
    """
    V = matrix @ inv(sqrtm(matrix.conj().T @ matrix))  # TODO: use SVD
    U, Q = schur(V)
    diag = np.diag(np.diag(U) / np.abs(np.diag(U)))
    return Q @ logm(diag) @ Q.conj().T


def log_matrix_newton_schur(matrix: NDArray) -> NDArray:
    """Logarithm of matrix using Schur's decomposition. Used to diagonalize
    nearly unitary matrices.

    Args:
        matrix (NDArray): square matrix.

    Returns:
        NDArray: matrix logarithm.
    """
    V1 = (matrix + inv(matrix).conj().T) / 2
    V = (V1 + inv(V1).conj().T) / 2
    U, Q = schur(V)
    diag = np.diag(np.diag(U) / np.abs(np.diag(U)))
    return Q @ logm(diag) @ Q.conj().T
