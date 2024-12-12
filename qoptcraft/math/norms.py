"""Matrix inner product and norm.
"""

from numbers import Number
from itertools import permutations

import numpy as np

from .matrix_type import Matrix


def trace_norm(matrix: Matrix) -> Number:
    """Trace norm of a matrix.

    Args:
        matrix (Matrix): first matrix

    Raises:
        ValueError: Matrix type is not array or scipy sparse.

    Returns:
        Number: norm.
    """
    result = np.linalg.norm(matrix, ord="nuc")
    # assert not np.isnan(result), "Matrix inner product is not a number."
    return result


def hs_inner_product(matrix_1: Matrix, matrix_2: Matrix) -> Number:
    """Hilbert-Schmidt product of two matrices.

    Args:
        matrix_1 (Matrix): first matrix
        matrix_2 (Matrix): second matrix

    Raises:
        ValueError: Matrix type is not array or scipy sparse.

    Returns:
        Number: scalar value of the product.
    """
    result = (matrix_1.conj().T @ matrix_2).trace()
    # assert not np.isnan(result), "Matrix inner product is not a number."
    return result


def hs_scalar_product(matrix_1: Matrix, matrix_2: Matrix) -> Number:
    """Hilbert-Schmidt product of two matrices.

    Args:
        matrix_1 (Matrix): first matrix
        matrix_2 (Matrix): second matrix

    Raises:
        ValueError: Matrix type is not array or scipy sparse.

    Returns:
        Number: scalar value of the product.
    """
    result = 0.5 * (matrix_1.conj().T @ matrix_2 + matrix_2.conj().T @ matrix_1).trace()
    # assert not np.isnan(result), "Matrix inner product is not a number."
    if isinstance(result, np.matrix):
        return result.item()
    return result


def hs_norm(matrix: Matrix) -> float:
    """Hilbert-Schmidt norm of a matrix

    Args:
        matrix (Matrix): a matrix.

    Returns:
        Number: the norm.
    """
    return np.sqrt(np.real(hs_inner_product(matrix, matrix)))


def eigenvalue_distance(matrix_1: Matrix, matrix_2: Matrix) -> float:
    """Hilbert-Schmidt norm of a matrix

    Args:
        matrix_1 (Matrix): a hermitian matrix.
        matrix_2 (Matrix): another hermitian matrix.

    Returns:
        Number: the minimum (among all permutations) euclidean distance between the eigenvalues.
    """
    eig_1 = np.linalg.eigvalsh(matrix_1)
    eig_2 = np.linalg.eigvalsh(matrix_2)
    min_norm = np.inf
    for eig_2_perm in permutations(eig_2):
        norm = np.linalg.norm(eig_1 - eig_2_perm)
        if norm < min_norm:
            min_norm = norm
    return min_norm

