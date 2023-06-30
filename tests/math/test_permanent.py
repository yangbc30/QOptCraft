from math import sqrt, isclose
from numbers import Number

import numpy as np
from numpy.typing import NDArray
import pytest

from qoptcraft.math import permanent


rng = np.random.default_rng()
unitary_3d = rng.standard_normal(9).reshape(3, 3)


def permanent_3d(matrix: NDArray) -> Number:
    """Hard-coded 3d permanent."""
    return (
        matrix[0, 2] * matrix[1, 1] * matrix[2, 0]
        + matrix[0, 1] * matrix[1, 2] * matrix[2, 0]
        + matrix[0, 2] * matrix[1, 0] * matrix[2, 1]
        + matrix[0, 0] * matrix[1, 2] * matrix[2, 1]
        + matrix[0, 1] * matrix[1, 0] * matrix[2, 2]
        + matrix[0, 0] * matrix[1, 1] * matrix[2, 2]
    )


@pytest.mark.parametrize(
    ("matrix"),
    (
        (rng.standard_normal(9).reshape(3, 3)),
        (rng.standard_normal(9).reshape(3, 3)),
    ),
)
def test_permanent(matrix: NDArray):
    perm_glynn = permanent(matrix, method="glynn")
    perm_ryser = permanent(matrix, method="ryser")
    assert isclose(perm_glynn, perm_ryser)


@pytest.mark.parametrize(
    ("matrix"),
    (
        (rng.standard_normal(4 * 4).reshape(4, 4)),
        (rng.standard_normal(6 * 6).reshape(6, 6)),
        (rng.standard_normal(9 * 9).reshape(9, 9)),
    ),
)
def test_equal_permanents(matrix: NDArray) -> None:
    perm_glynn = permanent(matrix, method="glynn")
    perm_ryser = permanent(matrix, method="ryser")
    assert isclose(perm_glynn, perm_ryser)
