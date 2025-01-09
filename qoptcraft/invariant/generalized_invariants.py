from itertools import product

import numpy as np
from numpy.typing import NDArray

from qoptcraft.state import State

from qoptcraft.basis import unitary_algebra_basis, image_algebra_basis, hilbert_dim
from qoptcraft.math import Matrix


def invariant_coef(indices, scattering_basis):
    matrix = np.eye(scattering_basis[0].shape[0], dtype=np.complex64)
    for i in indices:
        matrix @= scattering_basis[i]
    return np.trace(matrix)


def invariant_operator(modes: int, photons: int, order: int) -> Matrix:
    """These operators are multiples of the identity."""

    scattering_basis = unitary_algebra_basis(modes)
    image_basis = image_algebra_basis(modes, photons)

    dim = hilbert_dim(modes, photons)
    invariant = np.zeros((dim, dim), dtype=np.complex64)

    for indices in product(*[list(range(modes * modes))] * order):
        matrix = np.eye(dim, dtype=np.complex128)
        coef = invariant_coef(indices, scattering_basis)
        if coef != 0:
            for idx in indices:
                matrix @= image_basis[idx]
            invariant += coef * matrix

    return invariant


def scalar_invariant(state: State, order: int) -> Matrix:

    scattering_basis = unitary_algebra_basis(state.modes)
    image_basis = image_algebra_basis(state.modes, state.photons)

    invariant = 0

    for indices in product(*[list(range(state.modes**2))] * order):
        coef = invariant_coef(indices, scattering_basis)
        if coef != 0:
            for idx in indices:
                coef *= np.trace(image_basis[idx] @ state.density_matrix)
            invariant += coef

    return invariant


def scalar_invariant_from_matrix(matrix: NDArray, modes: int, photons: int, order: int) -> Matrix:

    scattering_basis = unitary_algebra_basis(modes)
    image_basis = image_algebra_basis(modes, photons)

    invariant = 0

    for indices in product(*[list(range(modes * modes))] * order):
        coef = invariant_coef(indices, scattering_basis)
        if coef != 0:
            for idx in indices:
                coef *= np.trace(image_basis[idx] @ matrix)
            invariant += coef

    return invariant
