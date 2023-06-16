"""Matrix inner product and norm.
"""

from numbers import Number

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import spmatrix


Matrix = NDArray | spmatrix


def mat_inner_product(matrix_1: Matrix, matrix_2: Matrix) -> Number:
    """Hilbert-Schmidt product of two matrices.

    Args:
        U (Matrix): first matrix
        V (Matrix): second matrix

    Raises:
        ValueError: Matrix type is not array or scipy sparse.

    Returns:
        Number: scalar value of the product.
    """
    if isinstance(matrix_1, spmatrix):
        matrix_1 = matrix_1.toarray()
    if isinstance(matrix_2, spmatrix):
        matrix_2 = matrix_2.toarray()
    res = matrix_1.conj().T @ matrix_2 + matrix_2.conj().T @ matrix_1
    assert not np.isnan(res.trace()), f"Nan trace. {matrix_1 = }\n\n {matrix_2 = }"
    return 0.5 * res.trace()


def mat_norm(matrix: Matrix) -> Number:
    """Hilbert-Schmidt norm of a matrix

    Args:
        matrix (Matrix): _description_

    Returns:
        Number: _description_
    """
    return np.sqrt(np.real(mat_inner_product(matrix, matrix)))
