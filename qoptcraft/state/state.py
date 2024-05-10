"""Implement a class to describe a pure state photonic state.
"""

from __future__ import annotations

import logging
from typing import Self
from numbers import Number

import numpy as np
from numpy.typing import NDArray, ArrayLike
import scipy

from qoptcraft.basis import get_photon_basis
from qoptcraft.operators import creation, annihilation
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
        density_matrix: NDArray | scipy.sparse.spmatrix,
        modes: int,
        photons: int,
    ) -> None:
        def is_hermitian(density_matrix, rtol: float = 1e-5, atol: float = 1e-8):
            """Reimplement numpy isclose for spare matrices"""
            mat_1 = density_matrix
            mat_2 = density_matrix.conj().T
            return abs(mat_1 - mat_2).max() <= atol + rtol * abs(mat_2).max()

        if not is_hermitian(density_matrix):
            raise NotHermitianError()
        if not np.isclose(density_matrix.trace(), 1):
            raise ValueError("Density matrix trace is not 1.")
        self.density_matrix = density_matrix
        self.photons = photons
        self.modes = modes

    @classmethod
    def from_mixture(cls, pure_states: list[PureState], probs: ArrayLike) -> Self:
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

        photons_list = [state.photons for state in pure_states]
        if not all(photons == photons_list[0] for photons in photons_list):
            raise NumberPhotonsError()

        modes_list = [state.modes for state in pure_states]
        if not all(modes == modes_list[0] for modes in modes_list):
            raise NumberModesError()

        density_matrix = probs[0] * pure_states[0].density_matrix
        for i in range(1, len(probs)):
            density_matrix += probs[i] * pure_states[i].density_matrix

        return cls(density_matrix, modes_list[0], photons_list[0])

    def evolution(self, unitary: NDArray) -> Self:
        new_density = unitary @ self.density_matrix @ unitary.T.conj()
        return MixedState(new_density, self.modes, self.photons)


class PureState(State):
    """A pure quantum state.

    Args:
        fock_states (tuple[tuple[int, ...]]): Fock states that, in superposition,
            constitute our pure state.
        coefs (ArrayLike): amplitude of each Fock state in the superposition.

    Attributes:
        modes (int): number of modes in the optical network.
        photons (int): number of photons.
        basis (tuple[tuple[int, ...]]): basis of the Hilbert state. Defaults to None.
        density_matrix (NDArray): density matrix of the state.
    """

    def __init__(self, fock_states: list[tuple[int, ...]], coefs: ArrayLike) -> None:
        self._assert_inputs(fock_states, coefs)

        self.photons: int = sum(fock_states[0])
        self.modes: int = len(fock_states[0])

        sorted_inputs = sorted(
            zip(fock_states, coefs, strict=True), key=lambda pair: pair[0], reverse=True
        )
        # sorted_inputs = [pair for pair in sorted_inputs if not np.isclose(pair[1], 0)]
        self.fock_states, coefs = zip(*sorted_inputs, strict=True)
        self.coefs = np.array(coefs)

        sum_coefs = np.sum(np.abs(self.coefs) ** 2)
        if sum_coefs == 1:
            self.amplitudes = self.coefs
        else:
            self.amplitudes = np.array(coefs) / np.sqrt(sum_coefs)
        self.probabilites: NDArray = np.abs(self.amplitudes) ** 2

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
        for fock in fock_states:
            for i in fock:
                if i < 0:
                    raise ValueError(f"Invalid fock state {fock}")

        photons_list = [sum(fock_state) for fock_state in fock_states]
        if not all(photons == photons_list[0] for photons in photons_list):
            raise NumberPhotonsError()

        modes_list = [len(fock_state) for fock_state in fock_states]
        if not all(modes == modes_list[0] for modes in modes_list):
            raise NumberModesError()

        if len(fock_states) != len(amplitudes):
            raise ValueError("Error: fock_states and amplitudes must have the same length.")

    def __str__(self) -> str:
        """String representation of the state."""
        str_ = ""
        for fock, amp in zip(self.fock_states, self.amplitudes, strict=True):
            str_ = str_ + f"{amp:.2f} * {fock} + \n"
        str_ = str_[:-4]
        return str_

    def __repr__(self) -> str:
        """Representation of the state."""
        return str(self)

    def __eq__(self, other: PureState) -> bool:
        """Compare if two pure states are equal."""
        if isinstance(other, PureState):
            if self.photons != other.photons or self.modes != other.modes:
                return False
            if self.fock_states != other.fock_states:
                fock_diff = set(self.fock_states).symmetric_difference(other.fock_states)
                print(f"Unequal fock states. Difference in fock states = {fock_diff}.")
                return False
            if not np.allclose(self.amplitudes, other.amplitudes):
                print("Amplitudes differ")
                return False
            return True
        return False

    def isclose(self, other: PureState, atol: float = 1e-3, rtol=1e-05) -> bool:
        """Compare if two pure states are equal."""
        if isinstance(other, PureState):
            if self.photons != other.photons or self.modes != other.modes:
                return False
            fock_states = list(set(self.fock_states) | set(other.fock_states))
            for fock in fock_states:
                try:
                    coef_self = np.abs(self._coef_fock(fock))
                    try:
                        coef_other = np.abs(other._coef_fock(fock))
                        if not np.isclose(coef_self, coef_other, atol=atol, rtol=rtol):
                            print(f"{fock = }, {coef_self = }, {coef_other = }")
                            return False
                    except ValueError:
                        if not np.isclose(coef_self, 0, atol=atol, rtol=rtol):
                            print(f"{fock = }, {coef_self = }, coef_other = 0")
                            return False
                except ValueError:
                    coef_other = np.abs(other._coef_fock(fock))
                    if not np.isclose(coef_other, 0, atol=atol, rtol=rtol):
                        print(f"{fock = }, coef_self = 0, {coef_other = }")
                        return False
            return True
        raise ValueError("other is not PureState")

    def __add__(self, other: PureState) -> Self:
        """Tensor product of self with other state."""
        if isinstance(other, Vacuum):
            return self
        if isinstance(other, PureState):
            if self.photons != other.photons:
                raise NumberPhotonsError()
            if self.modes != other.modes:
                raise NumberModesError()

            coefs = []
            fock_states = list(set(self.fock_states) | set(other.fock_states))
            for fock in fock_states:
                try:
                    coefs.append(self._coef_fock(fock))
                    try:
                        coefs[-1] += other._coef_fock(fock)
                    except ValueError:
                        continue
                except ValueError:
                    coefs.append(other._coef_fock(fock))
            return PureState(fock_states, coefs)

        logging.error(f"Addition not implemented for opperand type {type(other)}")
        raise NotImplementedError

    def __radd__(self, other: PureState) -> Self:
        """Tensor product of self with other state."""
        return self + other

    def __iadd__(self, other: PureState) -> Self:
        """Inplace addition of states."""
        return self + other

    def __sub__(self, other: PureState) -> Self:
        """Substraction of states."""
        if isinstance(other, Vacuum):
            return self
        if isinstance(other, PureState):
            if self.photons != other.photons:
                raise NumberPhotonsError()
            if self.modes != other.modes:
                raise NumberModesError()
            return self + (-1) * other
        logging.error(f"Substraction not implemented for opperand type {type(other)}")
        raise NotImplementedError

    def __neg__(self) -> Self:
        return PureState(self.fock_states, -self.coefs)

    def __isub__(self, other: PureState) -> Self:
        """Inplace substraction of states."""
        return self - other

    def __mul__(self, other: PureState | Number) -> Self:
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

    def __rmul__(self, other: PureState | Number) -> Self:
        """Tensor product of self with other state."""
        return self * other

    def __imul__(self, other: PureState | Number) -> Self:
        """Tensor product of self with other state."""
        return self * other

    def __truediv__(self, other: Number) -> Self:
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

    def __pow__(self, exponent: int, modulo=None) -> Self:
        """Tensor product of self with itself a given number of times."""
        state = self
        for _ in range(exponent - 1):
            state = state * self
        return state

    def conj(self) -> Self:
        """State with conjugated coefficients."""
        return PureState(self.fock_states, self.coefs.conj())

    @property
    def density_matrix(self):
        """Density matrix of the pure state in a certain basis."""
        state_in_basis = self.state_in_basis()
        return np.outer(state_in_basis, state_in_basis.conj().T)

    def state_in_basis(self) -> NDArray:
        """Output the state vector in the Fock basis."""
        if self.basis is None:
            self.basis = get_photon_basis(self.modes, self.photons)

        state = np.zeros(len(self.basis), dtype=complex)

        for i, fock in enumerate(self.fock_states):
            for j, basis_fock in enumerate(self.basis):
                if fock == basis_fock:
                    state[j] = self.amplitudes[i]
        return state

    def amp_fock(self, fock: tuple[int, ...]) -> Number:
        """Get the amplitude of a certain fock state in the basis."""
        index = self.fock_states.index(fock)
        return self.amplitudes[index]

    def _coef_fock(self, fock: tuple[int, ...]) -> Number:
        """Get the coefficient of a certain fock state in the basis."""
        index = self.fock_states.index(fock)
        return self.coefs[index]

    def coefs_in_basis(self) -> NDArray:
        """Provide the coefs of the state in the fock basis."""
        if self.basis is None:
            self.basis = get_photon_basis(self.modes, self.photons)

        state = np.zeros(len(self.basis), dtype=complex)

        for i, fock in enumerate(self.fock_states):
            for j, basis_fock in enumerate(self.basis):
                if fock == basis_fock:
                    state[j] = self.coefs[i]
        return state

    def dot(self, other: PureState) -> complex:
        if isinstance(other, Vacuum):
            return 0
        prod = 0
        for i, fock_self in enumerate(self.fock_states):
            try:
                prod += self.amplitudes[i] * other.amp_fock(fock_self).conjugate()
            except ValueError:
                continue
        return round(prod, 7)

    def dot_coefs(self, other: PureState) -> complex:
        if isinstance(other, Vacuum):
            return 0
        prod = 0
        for i, fock_self in enumerate(self.fock_states):
            try:
                prod += self.coefs[i] * other._coef_fock(fock_self).conjugate()
            except ValueError:
                continue
        return round(prod, 7)

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
                # if coef_ == 0:  # not really necessary to check this
                ##### continue  # since it will raise a ValueError below
                coef = self.amplitudes[i] * coef_
                fock_, coef_ = creation(mode_creat, fock_)
                coef *= coef_
                try:
                    exp += coef * self.amp_fock(fock_).conjugate()
                except ValueError:
                    continue
        return exp

    def creation(self, mode: int) -> PureState:
        """Creation of a photon in a certain mode.

        Args:
            mode (int): mode in which to create.

        Returns:
            PureState: created state.
        """
        state = Vacuum()
        for fock, coef in zip(self.fock_states, self.coefs, strict=True):
            fock_ = list(fock)
            modes = len(fock)
            if mode + 1 > modes:
                fock_ = fock_ + (mode - modes) * [0] + [1]
                coef_creat = 1
            else:
                photons = fock[mode]
                coef_creat = np.sqrt(photons + 1)
                fock_[mode] = photons + 1
            state += Fock(*fock_, coef=coef_creat * coef)
        return state

    def annihilation(self, mode: int) -> Self:
        """Annihilation of a photon in a certain mode.

        Args:
            mode (int): mode in which to annihilate.

        Returns:
            PureState: created state.
        """
        if mode + 1 > self.modes:
            return Vacuum()

        state = Vacuum()
        for fock, coef in zip(self.fock_states, self.coefs, strict=True):
            photons = fock[mode]
            coef_annih = np.sqrt(photons)
            fock_ = list(fock)
            fock_[mode] = photons - 1
            if coef_annih == 0 or sum(fock_) == 0:
                continue
            state += Fock(*fock_, coef=coef_annih * coef)
        return state

    def evolution(self, unitary: NDArray) -> Self:
        "Evolve the state with a unitary."
        if self.basis is None:
            self.basis = get_photon_basis(self.modes, self.photons)
        count = 0
        amp_list = list(self.amplitudes)
        for i, state in enumerate(self.basis):
            if state not in self.fock_states:
                amp_list.insert(i + count, 0)
                count += 1
        new_amps = unitary @ np.array(amp_list)
        new_amps_wo_zeros = []
        new_fock = []
        for fock, amp in zip(self.basis, new_amps, strict=True):
            if amp != 0:
                new_amps_wo_zeros.append(amp)
                new_fock.append(fock)
        return PureState(new_fock, new_amps_wo_zeros)


class Fock(PureState):
    def __init__(self, *photons: int, coef: Number = 1) -> None:
        super().__init__(fock_states=(photons,), coefs=(coef,))

    def __repr__(self) -> str:
        return f"{self.fock_states[0]}"

    def __iter__(self):
        """Iterate the photons of the Fock state."""
        yield from self.fock_states[0]

    def __len__(self):
        """Number of modes in the fock state."""
        return self.modes


class Vacuum(State):

    def __repr__(self) -> str:
        return "Vacuum"

    def __str__(self) -> str:
        return "Vacuum"

    def creation(self, mode: int) -> PureState:
        """Creation of a photon in a certain mode.

        Args:
            mode (int): mode in which to create.

        Returns:
            PureState: created state.
        """
        if mode == 0:
            return Fock(1)
        return Fock(*(mode * [0]), 1)

    def annihilation(self, mode: int) -> Self:
        """Annihilation of a photon in a certain mode.

        Args:
            mode (int): mode in which to annihilate.

        Returns:
            PureState: created state.
        """
        return self
