"""Migdał et al. invariants

References:
    Migdał et al. Multiphoton states related via linear optics.
    https://arxiv.org/abs/1403.3069
"""
from typing import Literal, Generator
import itertools
from copy import deepcopy

import numpy as np
from numpy.typing import NDArray
from scipy.special import comb

from qoptcraft.state import State, PureState
from qoptcraft.math import hs_scalar_product, gram_schmidt_generator
from qoptcraft.basis import get_algebra_basis


def moments_invariant(state: PureState, order: int) -> NDArray:
    """Calculate the k-th order correlator of a pure state.

    Note:
        Calculations without using the basis are much faster and memory efficient.

    Args:
        state (State): a photonic quantum state.
        order (int): order k of the correlator.

    Returns:
        NDArray: matrix of the correlator of order k.
    """
    modes = state.modes
    matrix_dim = int(comb(modes, order, repetition=True))
    invariant = np.zeros((matrix_dim, matrix_dim))

    for i, modes_annih in enumerate(itertools.combinations_with_replacement(range(modes), order)):
        for j, modes_creat in enumerate(itertools.combinations_with_replacement(range(modes), order)):
            invariant[i,j] = moment_element(state, modes_annih, modes_creat)
    return invariant


def moment_element(state: PureState, modes_annih: int, modes_creat: int) -> float:
    r"""Compute the expecation value of $a^\dagger_i a_j$.

    Args:
        mode_creat (int): mode where we apply the creation operator.
        mode_annih (int): mode where we apply the annihilation operator.

    Returns:
        float: expectation value.
    """
    state_copy = state

    for mode in modes_annih:
        state = state.creation(mode)
    for mode in modes_creat:
        state = state.annihilation(mode)

    print(f"{state_copy = }")
    print(f"{state = }")

    return state_copy.dot(state)
