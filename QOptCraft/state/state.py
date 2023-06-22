"""Implement a class to describe a pure state photonic state.
"""
from __future__ import annotations

import logging
from numbers import Number

import numpy as np
from numpy.typing import NDArray, ArrayLike

from QOptCraft.basis import get_photon_basis
from QOptCraft.operators import creation, annihilation
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
    def from_mixture(cls, pure_states: list[PureState], probs: ArrayLike):
        """Initialize a mixed state from a superposition of pure states instead of
        initializing directly the density matrix.

        Args:
            pure_states (list[PureState]): pure states in the statistical mixture.
            probs (ArrayLike): probability of measuring each pure state.

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
        fock_states (tuple[tuple[int, ...]]): Fock states that, in superposition,
            constitute our pure state.
        amplitudes (ArrayLike): amplitude of each Fock state in the superposition.

    Attributes:
        modes (int): number of modes in the optical network.
        photons (int): number of photons.
        basis (tuple[tuple[int, ...]]): basis of the Hilbert state. Defaults to None.
        density_matrix (NDArray): density matrix of the state.
    """

    def __init__(self, fock_states: tuple[tuple[int, ...]], coefs: ArrayLike) -> None:
        self._assert_inputs(fock_states, coefs)

        self.photons: int = sum(fock_states[0])
        self.modes: int = len(fock_states[0])
        self.fock_states = fock_states
        self.coefs = np.array(coefs)

        sum_coefs = self.coefs.dot(self.coefs.conj())
        self.amplitudes = np.array(coefs) / np.sqrt(sum_coefs)
        self.probabilites: NDArray = self.amplitudes * self.amplitudes.conj()

        self.basis: tuple[tuple[int, ...]] | None = None

    @staticmethod
    def _assert_inputs(fock_states: tuple[tuple[int, ...]], amplitudes: ArrayLike) -> None:
        """Assert the instance inputs are not contradictory.

        Args:
            fock_states (tuple[tuple[int, ...]]): Fock states that, in superposition,
                constitute our pure state.
            amplitudes (list[float]): amplitude of each Fock state in the superposition.

        Raises:
            NumberPhotonsError: Not all states have the same number of photons.
            NumberModesError: Not all states have the same number of modes.
            ProbabilityError: Probabilities don't add up to 1.
        """
        photons_list = [sum(fock_state) for fock_state in fock_states]
        if not all(photons == photons_list[0] for photons in photons_list):
            raise NumberPhotonsError()

        modes_list = [len(fock_state) for fock_state in fock_states]
        if not all(modes == modes_list[0] for modes in modes_list):
            raise NumberModesError()

        if len(fock_states) != len(amplitudes):
            raise ValueError("Error: fock_states and amplitudes must have the same length.")

    def __str__(self) -> str:
        """Tensor product or scalar multiplication of a state or number times
        another state.
        """
        str_ = ""
        for fock, amp in zip(self.fock_states, self.amplitudes, strict=True):
            str_ = str_ + f"{amp:.2f} * {fock} + \n"
        str_ = str_[:-4]
        return str_

    def __add__(self, other: PureState) -> PureState:
        """Tensor product of self with other state."""
        if isinstance(other, PureState):
            if self.photons != other.photons:
                raise NumberPhotonsError()
            if self.modes != other.modes:
                raise NumberModesError()

            coefs = []
            fock_states = list(set(self.fock_states) | set(other.fock_states))
            for fock in fock_states:
                try:
                    idx_self = self.fock_states.index(fock)
                    coefs.append(self.coefs[idx_self])
                    try:
                        idx_other = other.fock_states.index(fock)
                        coefs[-1] += other.coefs[idx_other]
                    except ValueError:
                        continue
                except ValueError:
                    idx_other = other.fock_states.index(fock)
                    coefs.append(other.coefs[idx_other])
            return PureState(fock_states, coefs)

        logging.error(f"Operation not implemented for opperand type {type(other)}")
        raise NotImplementedError

    def __mul__(self, other: PureState | Number) -> PureState:
        """Tensor product of self with other state."""
        if isinstance(other, Number):
            return PureState(self.fock_states, self.coefs * other)
        if isinstance(other, PureState):
            fock_states = []
            amplitudes = []
            for fock, amp in zip(self.fock_states, self.amplitudes, strict=True):
                for fock_other, amp_other in zip(other.fock_states, other.amplitudes, strict=True):
                    fock_states.append(fock + fock_other)
                    amplitudes.append(amp * amp_other)
            return PureState(fock_states, amplitudes)
        else:
            logging.error(f"Operation not implemented for opperand type {type(other)}")
            raise NotImplementedError

    def __rmul__(self, other: PureState | Number) -> PureState:
        """Tensor product of self with other state."""
        return self * other

    def __truediv__(self, other: Number) -> PureState:
        """Return the division of self and the other number.

        Args:
            other: Other scalar value (rhs).

        Raises:
            ValueError: Division by zero.
            TypeError: Not int/float passed in.

        Returns:
            The multiplication of self and the other vector/number.
        """
        if not isinstance(other, Number):
            raise TypeError("You must pass in a scalar value.")
        return self * (1 / other)

    def __pow__(self, exponent: int, modulo=None) -> PureState:
        """Tensor product of self with itself a given number of times."""
        state = self
        for _ in range(exponent - 1):
            state = state * self
        return state

    @property
    def density_matrix(self):
        """Density matrix of the pure state in a certain basis."""
        state_in_basis = self.state_in_basis()
        return np.outer(state_in_basis, state_in_basis.conj().T)

    def state_in_basis(self) -> NDArray:
        """Given a vector in terms of elements of a basis and amplitudes,
        output the state vector.
        """
        if self.basis is None:
            self.basis = get_photon_basis(self.modes, self.photons)

        state = np.zeros(len(self.basis), dtype=complex)

        for i, fock in enumerate(self.fock_states):
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
            for i, fock in enumerate(self.fock_states):
                exp += self.probabilites[i] * fock[mode_creat]
        else:
            for i, fock in enumerate(self.fock_states):
                fock_, coef_ = annihilation(mode_annih, fock)
                coef = self.amplitudes[i] * coef_
                fock_, coef_ = creation(mode_creat, fock_)
                coef *= coef_
                try:
                    j = self.fock_states.index(fock_)
                    exp += coef * (self.amplitudes[j].conjugate()).real
                except ValueError:
                    continue
        return exp


class Fock(PureState):
    def __init__(self, *photons: int, coef: Number = 1) -> None:
        super().__init__(fock_states=(photons,), coefs=(coef,))

    def __repr__(self) -> str:
        return f"{self.fock_states[0]}"
