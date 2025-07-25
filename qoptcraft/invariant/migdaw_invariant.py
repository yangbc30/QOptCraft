"""Migdał et al. invariants

References:
    Migdał et al. Multiphoton states related via linear optics.
    https://arxiv.org/abs/1403.3069
"""

from itertools import combinations_with_replacement

import numpy as np
from numpy.typing import NDArray
from scipy.special import comb, factorial

from qoptcraft.state import PureState, Vacuum


def migdaw_invariant(state: PureState, order: int) -> NDArray:
    """Calculate the k-th order correlator of a pure state.

    Args:
        state (State): a photonic quantum state.
        order (int): order k of the correlator.

    Returns:
        NDArray: matrix of the correlator of order k.
    """
    modes = state.modes
    matrix_dim = int(comb(modes, order, repetition=True))
    invariant = np.zeros((matrix_dim, matrix_dim), dtype=np.complex64)

    for i, modes_annih in enumerate(combinations_with_replacement(range(modes), order)):
        for j, modes_creat in enumerate(combinations_with_replacement(range(modes), order)):
            invariant[i, j] = migdaw_element(state, modes_annih, modes_creat)
    print(f"{invariant = }")
    return np.linalg.eigvals(invariant).round(23)


def migdaw_element(
    state: PureState | Vacuum, modes_annih: tuple[int, ...], modes_creat: tuple[int, ...]
) -> float:
    r"""Compute the expecation value of $a^\dagger_i a_j$.

    Args:
        mode_creat (tuple[int,...]): mode where we apply the creation operator.
        mode_annih (tuple[int,...]): mode where we apply the annihilation operator.

    Returns:
        float: expectation value.
    """
    state.coefs = state.amplitudes
    state_copy = state

    annih_exponents = np.array([modes_annih.count(mode) for mode in range(state.modes)])
    creat_exponents = np.array([modes_creat.count(mode) for mode in range(state.modes)])
    coef = np.sqrt(np.prod(factorial(annih_exponents)) * np.prod(factorial(creat_exponents)))

    for mode in modes_creat:
        state = state.creation(mode)
    for mode in modes_annih:
        state = state.annihilation(mode)

    return state_copy._dot_coefs(state) / coef
