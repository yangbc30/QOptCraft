"""Matrix inner product and norm.
"""

from numbers import Number

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import spmatrix


Matrix = NDArray | spmatrix


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
    return result


def hs_norm(matrix: Matrix) -> float:
    """Hilbert-Schmidt norm of a matrix

    Args:
        matrix (Matrix): a matrix.

    Returns:
        Number: the norm.
    """
    return np.sqrt(np.real(hs_inner_product(matrix, matrix)))
