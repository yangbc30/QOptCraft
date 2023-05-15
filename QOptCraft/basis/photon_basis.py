from numbers import Number

import numpy as np
from numpy.typing import NDArray


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

    new_basis: list = []
    for n in range(photons + 1):
        basis = photon_basis(photons - n, modes - 1)
        for vector in basis:
            new_basis.append([n, *vector])
    return new_basis


def state_in_basis(
    vectors: list[list[int]], amplitudes: list[Number], basis: list[list[int]]
) -> NDArray:
    """Given a vector in terms of elements of a basis and amplitudes,
    output the state vector.
    """
    state = np.zeros(len(basis), dtype=complex)

    for i, vector in enumerate(vectors):
        for j, basis_vector in enumerate(basis):
            if vector == basis_vector:
                state[j] = amplitudes[i]
    return state
