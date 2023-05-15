"""Implement a class to describe a pure state photonic state.
"""
import math

import numpy as np
from numpy.typing import NDArray

from QOptCraft.basis import state_in_basis


def photon_basis(photons: int, modes: int) -> list[list[int]]:
    """Given a number of photons and modes, generate the basis of the Hilbert space.

    Args:
        photons (int): number of photons.
        modes (int): number of modes.

    Returns:
        list[list[int]]: basis of the Hilbert space.
    """

    if photons < 0:
        photons = 0
    if modes == 1:
        return [[photons]]

    new_vector_list = []
    for n in range(photons + 1):
        vector_list = photon_basis(photons - n, modes - 1)
        for vector in vector_list:
            new_vector_list.append([n, *vector])

    return new_vector_list


class State:
    """Base class for our quantum states."""

    ...


class PureState(State):
    """A pure quantum state.

    Args:
        fock_list (list[list[int]]): Fock states that, in superposition,
        constitute our pure state.
        coef_list (list[float]): coefficient of each Fock state in the superposition.
        basis (list[list[int]], optional): basis of the Hilbert state. Defaults to None.

    Attributes:

        num_photons (int): number of photons.
        num_modes (int): number of modes in the optical network.
        basis (list[list[int]]): basis of the Hilbert state. Defaults to None.
    """

    def __init__(
        self,
        fock_list: list[list[int]],
        coef_list: list[float],
        basis: list[list[int]] | None = None,
    ) -> None:
        self.num_photons: int = sum(fock_list[0])
        self.num_modes: int = len(fock_list[0])

        if basis is None:
            basis = photon_basis(self.num_photons, self.num_modes)
        self.basis = basis
        self.state_in_basis = state_in_basis(fock_list, coef_list, self.basis)

    @property
    def density_matrix(self):
        return np.outer(self.state_in_basis, self.state_in_basis.conj().T)


class MixedState(State):
    def __init__(
        self,
        pure_states: list[PureState],
        probs: list[float],
        density_matrix: NDArray | None = None,
    ) -> None:
        if density_matrix is not None:
            self.matrix = density_matrix
        else:
            assert len(pure_states) == len(
                probs
            ), "Error: unequal length of states and probabilities list."
            assert math.isclose(
                1, sum(probs)
            ), "Probabilities don't add up to 1."
            self.matrix = probs[0] * pure_states[0].density_matrix
            for i in range(1, len(probs)):
                self.matrix += probs[i] * pure_states[i].density_matrix
