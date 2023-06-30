import numpy as np
from numpy.testing import assert_allclose
import pytest

from qoptcraft.operators import haar_random_unitary


@pytest.mark.parametrize(("dim"), (3, 4, 6))
def test_unitarity(dim: int):
    unitary = haar_random_unitary(dim)
    assert_allclose(unitary @ unitary.conj().T, np.eye(dim), rtol=1e-13, atol=1e-10)
