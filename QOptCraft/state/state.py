"""Implement a class to describe a pure state photonic state.
"""
import math

import numpy as np
from numpy.typing import NDArray

from QOptCraft.basis import state_in_basis, photon_basis


class State:
    """Base type for our quantum states."""

    ...


class MixedState(State):
    """A mixed quantum state.

    Args:
        density_matrix (NDArray): provide directly the density matrix of the mixed state.

    Attributes:
        basis (list[list[int]]): basis of the Hilbert state. Defaults to None.
    """

    def __init__(
        self,
        density_matrix: NDArray,
        basis: list[list[int]],
    ) -> None:
        self.density_matrix = density_matrix
        self.basis = basis


class PureState(MixedState):
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
        # TODO: add variable number of photons
        self.num_photons: int = sum(fock_list[0])
        self.num_modes: int = len(fock_list[0])

        if basis is None:
            basis = photon_basis(self.num_photons, self.num_modes)
        self.basis = basis
        self.state_in_basis = state_in_basis(fock_list, coef_list, self.basis)
        density_matrix = np.outer(self.state_in_basis, self.state_in_basis.conj().T)
        super.__init__(density_matrix, basis)


class PureMixture(MixedState):
    """A mixed quantum state.

    Args:
        density_matrix (NDArray): provide directly the density matrix of the mixed state.

    Attributes:
        basis (list[list[int]]): basis of the Hilbert state. Defaults to None.
    """

    def __init__(
        self, pure_states: list[PureState], probs: list[float], basis: list[list[int]]
    ) -> None:
        # TODO: implement different number of photons (modify photon_basis)
        # ? What if we have a statistical mixture of different number of photons?
        # * We should define a Hilbert space that is the sum of both spaces

        assert len(pure_states) == len(probs, "States and probabilities differ in length.")
        assert math.isclose(1, sum(probs)), "Probabilities don't add up to 1."

        density_matrix = probs[0] * pure_states[0].density_matrix
        for i in range(1, len(probs)):
            density_matrix += probs[i] * pure_states[i].density_matrix

        super().__init__(density_matrix, basis)
