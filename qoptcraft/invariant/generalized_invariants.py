from itertools import product

import numpy as np
from numpy.typing import NDArray

from qoptcraft.state import State

from qoptcraft.basis import unitary_algebra_basis, image_algebra_basis, hilbert_dim, basis_image_orthonormal
from qoptcraft.math import Matrix


def invariant_coef(indices, scattering_basis):
    matrix = np.eye(scattering_basis[0].shape[0], dtype=np.complex64)
    for i in indices:
        matrix @= scattering_basis[i]
    return np.trace(matrix)


def casimir_operator(modes: int, photons: int, order: int, orthonormal=False) -> Matrix:
    """These operators are multiples of the identity."""

    scattering_basis = unitary_algebra_basis(modes)
    if orthonormal:
        image_basis = basis_image_orthonormal(modes, photons)
    else:
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


def invariant_operator(state: State, order: int, orthonormal=False) -> Matrix:
    """These operators are multiples of the identity."""

    scattering_basis = unitary_algebra_basis(state.modes)
    if orthonormal:
        image_basis = basis_image_orthonormal(state.modes, state.photons)
    else:
        image_basis = image_algebra_basis(state.modes, state.photons)

    dim = hilbert_dim(state.modes, state.photons)
    invariant = np.zeros((dim, dim), dtype=np.complex64)

    for indices in product(*[list(range(state.modes * state.modes))] * order):
        matrix = np.eye(dim, dtype=np.complex128)
        # TODO: use the cyclic property of the trace to reduce the number of calls to invariant_coef
        coef = invariant_coef(indices, scattering_basis)
        if coef != 0:
            for idx in indices:
                matrix @= image_basis[idx] @ state.density_matrix
            invariant += coef * matrix

    return np.linalg.eigvals(invariant)


def commutator(A, B):
    return A @ B - B @ A


def invariant_operator_commutator(state: State, order: int, orthonormal=False) -> Matrix:
    """These operators are multiples of the identity."""

    scattering_basis = unitary_algebra_basis(state.modes)
    if orthonormal:
        image_basis = basis_image_orthonormal(state.modes, state.photons)
    else:
        image_basis = image_algebra_basis(state.modes, state.photons)

    dim = hilbert_dim(state.modes, state.photons)
    invariant = np.zeros((dim, dim), dtype=np.complex64)

    for indices in product(*[list(range(state.modes * state.modes))] * order):
        matrix = np.eye(dim, dtype=np.complex128)
        coef = invariant_coef(indices, scattering_basis)
        if coef != 0:
            for idx in indices:
                matrix @= commutator(image_basis[idx], state.density_matrix)
            invariant += coef * matrix

    return np.linalg.eigvals(invariant)


def invariant_operator_nested_commutator(state: State, order: int, orthonormal=False) -> Matrix:
    """These operators are multiples of the identity."""

    scattering_basis = unitary_algebra_basis(state.modes)
    if orthonormal:
        image_basis = basis_image_orthonormal(state.modes, state.photons)
    else:
        image_basis = image_algebra_basis(state.modes, state.photons)

    dim = hilbert_dim(state.modes, state.photons)
    invariant = np.zeros((dim, dim), dtype=np.complex64)

    for indices in product(*[list(range(state.modes * state.modes))] * order):
        matrix = np.eye(dim, dtype=np.complex128)
        coef = invariant_coef(indices, scattering_basis)
        if coef != 0:
            matrix = commutator(image_basis[indices[0]], state.density_matrix)
            for idx in indices[1:]:
                matrix = commutator(image_basis[idx], matrix)
            invariant += coef * matrix

    return np.linalg.eigvals(invariant)


def invariant_operator_traces(state: State, order: int, orthonormal=False) -> Matrix:
    """These operators are multiples of the identity."""

    scattering_basis = unitary_algebra_basis(state.modes)
    if orthonormal:
        image_basis = basis_image_orthonormal(state.modes, state.photons)
    else:
        image_basis = image_algebra_basis(state.modes, state.photons)

    dim = hilbert_dim(state.modes, state.photons)
    invariant = np.zeros((dim, dim), dtype=np.complex64)

    for indices in product(*[list(range(state.modes * state.modes))] * order):
        matrix = np.eye(dim, dtype=np.complex128)
        coef = invariant_coef(indices, scattering_basis)
        if coef != 0:
            for idx in indices:
                matrix @= image_basis[idx] @ state.density_matrix
            invariant += coef * matrix

    traces = []
    for i in range(1, dim + 1):
        traces.append(np.trace(np.linalg.matrix_power(invariant, i)))

    return traces


def scalar_invariant(state: State, order: int, orthonormal=False) -> Matrix:

    scattering_basis = unitary_algebra_basis(state.modes)
    if orthonormal:
        image_basis = basis_image_orthonormal(state.modes, state.photons)
    else:
        image_basis = image_algebra_basis(state.modes, state.photons)

    invariant = 0

    for indices in product(*[list(range(state.modes**2))] * order):
        coef = invariant_coef(indices, scattering_basis)
        if coef != 0:
            for idx in indices:
                coef *= np.trace(image_basis[idx] @ state.density_matrix)
            invariant += coef

    return invariant


def scalar_invariant_from_matrix(matrix: NDArray, modes: int, photons: int, order: int, orthonormal=False) -> Matrix:

    scattering_basis = unitary_algebra_basis(modes)
    if orthonormal:
        image_basis = basis_image_orthonormal(modes, photons)
    else:
        image_basis = image_algebra_basis(modes, photons)

    invariant = 0

    for indices in product(*[list(range(modes * modes))] * order):
        coef = invariant_coef(indices, scattering_basis)
        if coef != 0:
            for idx in indices:
                coef *= np.trace(image_basis[idx] @ matrix)
            invariant += coef

    return invariant
