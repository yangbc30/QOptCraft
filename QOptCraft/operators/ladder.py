"""Creation and annihilation operators.
"""

from math import sqrt


def creation(mode: int, state: list[int]) -> float:
    """Creation operator acting on a specific mode. Modifies state in-place.

    Args:
        mode (int): a quantum mode.
        state (list[int]): fock basis state.

    Returns:
        float: coefficient resulting from the creation operation.
    """
    photons = state[mode]
    coef = sqrt(photons + 1)
    state[mode] = photons + 1  # * modified in-place
    return coef


def annihilation(mode: int, state: list[int]) -> float:
    """Annihilation operator acting on a specific mode.

    Args:
        mode (int): a quantum mode.
        state (list[int]): fock basis state.

    Returns:
        float: coefficient resulting from the annihilation operation.
    """
    photons = state[mode]
    coef = sqrt(photons)
    state[mode] = photons - 1  # * modified in-place
    return coef
