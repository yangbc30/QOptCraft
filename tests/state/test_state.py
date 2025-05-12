import pytest
import numpy as np
from numpy.typing import NDArray, ArrayLike
from numpy.testing import assert_allclose
from scipy.special import factorial

from qoptcraft.basis import BasisPhoton
from qoptcraft.state import Fock, PureState, Vacuum


@pytest.mark.parametrize(
    ("state_1", "state_2", "result"),
    (
        (Fock(2, 1), Fock(2, 1), True),
        (Fock(0, 1, 1), Fock(1, 1, 0), False),
        (Fock(0, 1), PureState([(0, 1)], [1]), True),
        (Fock(0, 1), PureState([(1, 0)], [1]), False),
        (PureState([(4, 0), (3, 1)], [1, 1]), PureState([(4, 0), (3, 1)], [1, 1]), True),
        (PureState([(4, 0), (3, 1)], [1, 1]), PureState([(2, 2), (3, 1)], [1, 1]), False),
    ),
)
def test_equal(state_1: PureState, state_2: PureState, result: bool) -> None:
    assert (state_1 == state_2) is result


@pytest.mark.parametrize(
    ("fock_list", "coefs"),
    (
        ([(2, 0, 1), (0, 3, 0), (1, 1, 1)], [1, 2, 1]),
        ([(1, 0)], [1]),
        ([(1, 3), (4, 0), (2, 2)], [1, -1, 1]),
    ),
)
def test_fock_sum(fock_list: BasisPhoton, coefs: ArrayLike) -> None:
    state_fock = Vacuum()
    for fock, coef in zip(fock_list, coefs):
        state_fock += coef * Fock(*fock)
    state_pure = PureState(fock_list, coefs)
    assert state_fock == state_pure


@pytest.mark.parametrize(("state_in"), ((2, 0, 3), (2, 1), (3, 0, 1, 5)))
def test_creation(state_in: tuple[int, ...]) -> None:
    """Recover a state |n1...nk> from number of photons"""
    modes = len(state_in)

    state_out: PureState = Fock(*([0] * modes))
    for mode, photons in enumerate(state_in):
        for _ in range(photons):
            state_out = state_out.creation(mode)

    coef = np.prod(factorial(state_in))
    state_out = state_out / np.sqrt(coef)

    assert Fock(*state_in) == state_out


@pytest.mark.parametrize(("state_in"), ((2, 0, 3), (2, 1), (3, 0, 1, 5)))
def test_coefs(state_in: tuple[int, ...]) -> None:
    """Recover a state |n1...nk> from number of photons"""
    modes = len(state_in)

    state_out: PureState = Fock(*([0] * modes))
    for mode, photons in enumerate(state_in):
        for _ in range(photons):
            state_out = state_out.creation(mode)

    coef = np.prod(factorial(state_in))
    state_out = state_out / np.sqrt(coef)
    assert_allclose(state_out.coefs, np.array([1]))
