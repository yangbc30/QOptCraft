"""Creation and annihilation operators.
"""

from numbers import Number
from math import sqrt


def creation(mode: int, state: list[int]) -> Number:
    """Creation operator acting on a specific mode. Modifies state in-place.

    Args:
        mode (int): a quantum mode.
        state (list[int]): fock basis state.
        coef (Number): coefficient of the state.

    Returns:
        tuple[list[int], Number]: created state and its coefficient.
    """
    photons = state[mode]
    coef = sqrt(photons + 1)
    state[mode] = photons + 1  # * modified in-place
    return coef


def annihilation(mode: int, state: list[int]) -> Number:
    """Annihilation operator acting on a specific mode.

    Args:
        mode (int): a quantum mode.
        state (list[int]): fock basis state.
        coef (Number): coefficient of the state.

    Returns:
        tuple[list[int], Number]: annihilated state and its coefficient.
    """
    photons = state[mode]
    coef = sqrt(photons)
    state[mode] = photons - 1  # * modified in-place
    return coef
