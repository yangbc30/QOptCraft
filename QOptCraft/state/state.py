"""Implement a class to describe a pure state photonic state.
"""

import numpy as np

from QOptCraft import state_in_basis


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

    def density_matrix(self):
        return np.matrix(self.state_in_basis).T @ np.matrix(self.state_in_basis).conj()


class MixedState(State):
    ...
