import numpy as np
from numpy.typing import NDArray
from numpy.testing import assert_allclose
import pytest

from qoptcraft.operators import qft
from qoptcraft.topogonov import toponogov
from qoptcraft.basis import hilbert_dim


@pytest.mark.parametrize(("unitary", "modes", "photons"), ((qft(6), 3, 2),))
def test_unitarity(unitary: NDArray, modes: int, photons: int):
    dim = hilbert_dim(modes, photons)
    approx_unitary, _ = toponogov(unitary, modes, photons)
    assert_allclose(approx_unitary @ approx_unitary.conj().T, np.eye(dim), atol=1e-10)


@pytest.mark.parametrize(("unitary", "modes", "photons"), ((qft(6), 3, 2),))
def test_qft(unitary: NDArray, modes: int, photons: int):
    min_error = 10
    for seed in range(30):
        expected, error = toponogov(unitary, modes, photons, seed)
        if error < min_error:
            min_error = error
    assert min_error < 2.5
