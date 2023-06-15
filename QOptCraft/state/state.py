"""Implement a class to describe a pure state photonic state.
"""
from __future__ import annotations

from numbers import Number

import numpy as np
from numpy.typing import NDArray

from QOptCraft.basis import get_photon_basis
from QOptCraft.utils import creation, annihilation
from ._exceptions import (
    ProbabilityError,
    PureStateLengthError,
    NumberPhotonsError,
    NumberModesError,
    NotHermitianError,
)


class State:
    """Base type for our quantum states."""

    ...


class MixedState(State):
    """A mixed quantum state.

    Args:
        density_matrix (NDArray): provide directly the density matrix of the mixed state.
        modes (int): number of modes in the optical network.
        photons (int): number of photons.

    Attributes:
        density_matrix (NDArray): density matrix of the state.
        modes (int): number of modes in the optical network.
        photons (int): number of photons.
    """

    def __init__(
        self,
        density_matrix: NDArray,
        modes: int,
        photons: int,
    ) -> None:
        if not np.allclose(density_matrix, density_matrix.conj().T):
            raise NotHermitianError()
        self.density_matrix = density_matrix
        self.photons = photons
        self.modes = modes

    @classmethod
    def from_mixture(cls, pure_states: list[PureState], probs: list[float]):
        """Initialize a mixed state from a superposition of pure states instead of
        initializing directly the density matrix.

        Args:
            pure_states (list[PureState]): pure states in the statistical mixture.
            probs (list[float]): probability of measuring each pure state.

        Raises:
            PureStateLengthError: States and probabilities differ in length.
            ProbabilityError: Probabilities don't add up to 1.
            NumberPhotonsError: Not all states have the same number of photons.
            NumberModesError: Not all states have the same number of modes.

        Returns:
            MixedState
        """
        if not len(pure_states) == len(probs):
            raise PureStateLengthError()
        if not np.isclose(1, sum(probs)):
            raise ProbabilityError(sum(probs))

        photons_list = [pure_state.photons for pure_state in pure_states]
        if not all(photons == photons_list[0] for photons in photons_list):
            raise NumberPhotonsError()

        modes_list = [len(pure_state) for pure_state in pure_states]
        if not all(modes == modes_list[0] for modes in modes_list):
            raise NumberModesError()

        density_matrix = probs[0] * pure_states[0].density_matrix
        for i in range(1, len(probs)):
            density_matrix += probs[i] * pure_states[i].density_matrix

        return cls(density_matrix, modes_list[0], photons_list[0])


class PureState(State):
    """A pure quantum state.

    Args:
        fock_list (list[list[int]]): Fock states that, in superposition,
            constitute our pure state.
        amplitudes (list[float]): amplitude of each Fock state in the superposition.

    Attributes:
        modes (int): number of modes in the optical network.
        photons (int): number of photons.
        basis (list[list[int]]): basis of the Hilbert state. Defaults to None.
        density_matrix (NDArray): density matrix of the state.
        state_in_basis (NDArray): pure state in the given basis.
    """

    def __init__(
        self,
        fock_list: list[list[int]],
        amplitudes: list[Number],
    ) -> None:
        self._assert_inputs(fock_list, amplitudes)

        self.photons: int = sum(fock_list[0])
        self.modes: int = len(fock_list[0])
        self.basis: list[list[int]] | None = None
        self.fock_list = fock_list
        self.amplitudes = amplitudes
        self.probabilites: list[Number] = [amp * amp.conjugate() for amp in amplitudes]

    def __str__(self) -> str:
        """Tensor product or scalar multiplication of a state or number times
        another state.
        """
        str_ = ""
        for fock, amp in zip(self.fock_list, self.amplitudes, strict=True):
            str_ = str_ + f"{amp:.2f} * {fock} + \n"
        str_ = str_[:-4]
        return str_

    def __mul__(self, other: PureState) -> PureState:
        """Tensor product of self with other state."""
        fock_list = []
        amplitudes = []
        for fock, amp in zip(self.fock_list, self.amplitudes, strict=True):
            for fock_other, amp_other in zip(other.fock_list, other.amplitudes, strict=True):
                fock_list.append(fock + fock_other)
                amplitudes.append(amp * amp_other)
        return PureState(fock_list, amplitudes)

    def __pow__(self, exponent: int, modulo=None) -> PureState:
        """Tensor product of self with itself a given number of times."""
        state = self
        for _ in range(exponent - 1):
            state = state * self
        return state

    @staticmethod
    def _assert_inputs(fock_list: list[list[int]], amplitudes: list[float]):
        """Assert the instance inputs are not contradictory.

        Args:
            fock_list (list[list[int]]): Fock states that, in superposition,
                constitute our pure state.
            amplitudes (list[float]): amplitude of each Fock state in the superposition.

        Raises:
            NumberPhotonsError: Not all states have the same number of photons.
            NumberModesError: Not all states have the same number of modes.
            ProbabilityError: Probabilities don't add up to 1.
        """
        photons_list = [sum(fock_state) for fock_state in fock_list]
        if not all(photons == photons_list[0] for photons in photons_list):
            raise NumberPhotonsError()

        modes_list = [len(fock_state) for fock_state in fock_list]
        if not all(modes == modes_list[0] for modes in modes_list):
            raise NumberModesError()

        sum_amps = np.array(amplitudes).dot(np.array(amplitudes).conj())
        if not np.isclose(1, sum_amps):
            raise ProbabilityError(sum_amps)

    @property
    def density_matrix(self):
        """Density matrix of the pure state in a certain basis."""
        state_in_basis = self.state_in_basis
        return np.outer(state_in_basis, state_in_basis.conj().T)

    @property
    def state_in_basis(self) -> NDArray:
        """Given a vector in terms of elements of a basis and amplitudes,
        output the state vector.
        """
        if self.basis is None:
            self.basis = get_photon_basis(self.modes, self.photons)

        state = np.zeros(len(self.basis), dtype=complex)

        for i, fock in enumerate(self.fock_list):
            for j, basis_fock in enumerate(self.basis):
                if fock == basis_fock:
                    state[j] = self.amplitudes[i]
        return state

    def exp_photons(self, mode_creat: int, mode_annih: int) -> float:
        r"""Compute the expecation value of $a^\dagger_i a_j$.

        Args:
            mode_creat (int): mode where we apply the creation operator.
            mode_annih (int): mode where we apply the annihilation operator.

        Returns:
            float: expectation value.
        """
        exp = 0
        if mode_creat == mode_annih:
            for i, fock in enumerate(self.fock_list):
                exp += self.probabilites[i] * fock[mode_creat]
        else:
            for i, fock_ in enumerate(self.fock_list):
                fock = fock_.copy()
                coef = self.amplitudes[i]
                coef *= annihilation(mode_annih, fock)
                coef *= creation(mode_creat, fock)
                try:
                    j = self.fock_list.index(fock)
                    exp += coef * self.amplitudes[j]
                except ValueError:
                    continue
        return exp
