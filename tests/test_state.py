import pytest
import numpy as np
from numpy.typing import NDArray
from scipy.special import factorial

from qoptcraft.state import Fock, PureState


@pytest.mark.parametrize(("state_in"), ((2, 0, 3), (2, 1), (3, 0, 1, 5)))
def test_creation(state_in: tuple[int, ...]) -> PureState:
    """Recover a state |n1...nk> from number of photons"""
    modes = len(state_in)

    state_out: PureState = Fock(*([0] * modes))
    for mode, photons in enumerate(state_in):
        for _ in range(photons):
            state_out = state_out.creation(mode)

    coef = np.prod(factorial(state_in))
    state_out = state_out / np.sqrt(coef)

    assert Fock(*state_in) == state_out
