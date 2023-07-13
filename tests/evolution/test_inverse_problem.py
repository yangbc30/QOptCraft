import numpy as np
from numpy.testing import assert_allclose
import pytest
from hypothesis import given, settings
import hypothesis.strategies as st

from qoptcraft import photon_unitary, haar_random_unitary
from qoptcraft.evolution import scattering_from_unitary


@given(st.integers(2, 4), st.integers(1, 3), st.integers(1))
@settings(deadline=None)
def test_scattering_from_unitary(modes: int, photons: int, seed: int) -> None:
    S = haar_random_unitary(modes, seed)
    U = photon_unitary(S, photons)
    S_rebuilt = scattering_from_unitary(U, modes, photons)
    quasi_id = S_rebuilt.conj().T @ S  # should be a product of the identity
    assert_allclose(quasi_id[0, 0] * np.eye(modes), quasi_id, atol=1e-7, rtol=1e-5)
