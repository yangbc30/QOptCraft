"""Creation and annihilation operators.
"""

from math import sqrt

from numba import jit


def creation_fock(mode: int, fock: tuple[int]) -> tuple[tuple[int, ...], float]:
    """Creation operator acting on a specific mode. Modifies state in-place.

    Args:
        mode (int): a quantum mode.
        fock (tuple[int, ...]): fock basis state.

    Returns:
        tuple[int, ...], float: created Fock state and its coefficient.
    """
    photons = fock[mode]
    coef = sqrt(photons + 1)
    fock = list(fock)
    fock[mode] = photons + 1
    return tuple(fock), coef


def annihilation_fock(mode: int, fock: tuple[int, ...]) -> tuple[tuple[int, ...], float]:
    """Annihilation operator acting on a specific mode.

    Args:
        mode (int): a quantum mode.
        fock (tuple[int, ...]): fock basis state.

    Returns:
        tuple[int, ...], float: annihilated Fock state and its coefficient.
    """
    photons = fock[mode]
    coef = sqrt(photons)
    fock = list(fock)
    fock[mode] = photons - 1
    return tuple(fock), coef
