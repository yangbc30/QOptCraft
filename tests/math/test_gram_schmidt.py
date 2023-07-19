import pytest
import numpy as np
from numpy.testing import assert_almost_equal

from qoptcraft.math import gram_schmidt, hs_scalar_product


BASIS_1 = [
    np.array([[1, 1], [1, 1]]),
    np.array([[1, 0], [1, 0]]),
    np.array([[0, 0], [0, 1]]),
    np.array([[1, 2], [0, 0]]),
]


BASIS_2 = [
    np.array([[1, 0], [0, 0]]),
    np.array([[0, 1], [3, 0]]),
    np.array([[0, 0], [1, 0]]),
    np.array([[1, 0], [0, 1]]),
]


BASIS_3 = [
    np.array([[1, 0], [0, 0]]),
    np.array([[0, 1], [0, 0]]),
    np.array([[0, 0], [1, 0]]),
    np.array([[0, 0], [0, 1]]),
]


BASIS_4 = [
    np.array([[1, 0], [0, 1]]),
    np.array([[0, 1], [1, 0]]),
    np.array([[0, -1j], [1j, 0]]),
    np.array([[1, 0], [0, -1]]),
]


@pytest.mark.parametrize(("basis"), (BASIS_1, BASIS_2, BASIS_3, BASIS_4))
def test_gram_schmidt(basis: list[np.ndarray]) -> None:
    orth_basis = gram_schmidt(basis)

    for matrix_1 in orth_basis:
        for matrix_2 in orth_basis:
            if (matrix_1 == matrix_2).all():
                assert_almost_equal(hs_scalar_product(matrix_1, matrix_2), 1)
            else:
                assert_almost_equal(hs_scalar_product(matrix_1, matrix_2), 0)
