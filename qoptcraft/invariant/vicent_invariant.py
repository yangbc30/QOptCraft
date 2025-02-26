"""Migdał et al. invariants

References:
    Migdał et al. Multiphoton states related via linear optics.
    https://arxiv.org/abs/1403.3069
"""

import numpy as np
from numpy.typing import NDArray
from numpy.linalg import matrix_power

from qoptcraft.state import PureState
from qoptcraft.basis import image_algebra_basis, BasisAlgebra
from .projection import projection_density


def two_basis_invariant(state: PureState, orthonormal=False) -> NDArray:
    """Calculate M_ij = Tr(O_iO_j rho).

    Args:
        state (State): a photonic quantum state.

    Returns:
        NDArray: spectrum of the total matrix.
    """
    image_basis = image_algebra_basis(state.modes, state.photons, orthonormal)

    dim = len(image_basis)
    invariant = np.zeros((dim, dim), dtype=np.complex64)
    for i, basis_i in enumerate(image_basis):
        for j, basis_j in enumerate(image_basis):
            invariant[i, j] = np.trace(basis_i @ basis_j @ state.density_matrix)

    return np.linalg.eigvals(invariant).round(23)


def m1_invariant(state: PureState, orthonormal=False) -> NDArray:
    """Calculate M1 = Tr(O_i rho)Tr(O_j rho).

    Args:
        state (State): a photonic quantum state.

    Returns:
        NDArray: spectrum of the total matrix.
    """
    image_basis = image_algebra_basis(state.modes, state.photons, orthonormal)

    dim = len(image_basis)
    invariant = np.zeros((dim, dim), dtype=np.complex64)
    for i, basis_i in enumerate(image_basis):
        for j, basis_j in enumerate(image_basis):
            invariant[i, j] = np.trace(basis_i @ state.density_matrix) * np.trace(
                basis_j @ state.density_matrix
            )

    return np.linalg.eigvals(invariant).round(23)


def m2_invariant(state: PureState, orthonormal=False) -> NDArray:
    """Calculate M2 = Tr(O_iO_j + O_jO_i rho).

    Args:
        state (State): a photonic quantum state.

    Returns:
        NDArray: spectrum of the total matrix.
    """
    image_basis = image_algebra_basis(state.modes, state.photons, orthonormal)

    dim = len(image_basis)
    invariant = np.zeros((dim, dim), dtype=np.complex64)
    for i, basis_i in enumerate(image_basis):
        for j, basis_j in enumerate(image_basis):
            invariant[i, j] = 0.5 * np.trace(
                (basis_i @ basis_j + basis_j @ basis_i) @ state.density_matrix
            )

    return np.linalg.eigvals(invariant).round(23)

def covariance_invariant(state: PureState, orthonormal=False) -> NDArray:
    """Calculate M_ij = Tr(O_i rho)Tr(O_j rho) - 0.5 Tr(O_iO_j + O_jO_i rho).

    Args:
        state (State): a photonic quantum state.

    Returns:
        NDArray: spectrum of the total matrix.
    """
    image_basis = image_algebra_basis(state.modes, state.photons, orthonormal)
    dim = len(image_basis)
    invariant = np.zeros((dim, dim), dtype=np.complex64)
    for i, basis_i in enumerate(image_basis):
        for j, basis_j in enumerate(image_basis):
            invariant[i, j] = np.trace(basis_i @ state.density_matrix) * np.trace(
                basis_j @ state.density_matrix
            )
            invariant[i, j] -= 0.5 * np.trace(
                (basis_i @ basis_j + basis_j @ basis_i) @ state.density_matrix
            )

    return np.linalg.eigvals(invariant).round(23)


def vicent_invariant(state: PureState, order: tuple[int, int, int] = (1, 0, 0), orthonormal=False) -> NDArray:
    """Calculate Vicent's invariant, which uses the commutators of (projected) density matrices
    with the basis of passive linear optical Hamiltonians.

    Args:
        state (State): a photonic quantum state.
        order (tuple[int,int,int]): The first integer: exponent of M(state_t); second: exponent
            of M(state_o); third: exponent of M(state).

    Returns:
        NDArray: spectrum of the total matrix.
    """

    image_basis = image_algebra_basis(state.modes, state.photons, orthonormal)

    invariant = np.identity(len(image_basis), dtype=np.complex64)

    if order[0] != 0:
        density_tangent = projection_density(state, subspace="image", orthonormal=orthonormal)
        invariant_tangent = _vicent_invariant_matrix(density_tangent, image_basis)
        invariant @= matrix_power(invariant_tangent, order[0])
    if order[1] != 0:
        density_orthogonal = projection_density(state, subspace="complement", orthonormal=orthonormal)
        invariant_orthogonal = _vicent_invariant_matrix(density_orthogonal, image_basis)
        invariant @= matrix_power(invariant_orthogonal, order[1])
    if order[2] != 0:
        invariant_density = _vicent_invariant_matrix(state.density_matrix, image_basis)
        invariant @= matrix_power(invariant_density, order[2])

    return np.linalg.eigvals(invariant).round(23)


def _vicent_invariant_matrix(state_matrix: NDArray, image_basis: BasisAlgebra) -> NDArray:
    """Calculate the matrix M_ij = Tr([basis_i, state][basis_j, state])).

    Args:
        state (State): a photonic quantum state.
        order (int): order k of the correlator.

    Returns:
        NDArray: matrix whose espectrum is invariant.
    """
    dim = len(image_basis)
    invariant = np.zeros((dim, dim), dtype=np.complex64)

    for i, basis_i in enumerate(image_basis):
        commutator_i = basis_i @ state_matrix - state_matrix @ basis_i
        for j, basis_j in enumerate(image_basis):
            commutator_j = basis_j @ state_matrix - state_matrix @ basis_j
            invariant[i, j] = np.trace(commutator_i @ commutator_j)
    print(f"{invariant.round(23) = }")
    return -invariant  # minus accounts for the i factors in the basis_i, basis_j


def vicent_matricial_invariant(
    state: PureState, order: tuple[int, int, int] = (1, 0, 0), orthonormal=False
) -> NDArray:
    """Calculate Vicent's invariant, which uses the commutators of (projected) density matrices
        with the basis of passive linear optical Hamiltonians.

        Args:
            state (State): a photonic quantum state.
            order (tuple[int,int,int]): The first integer: exponent of M(state_t); second: exponent
                of M(state_o); third: exponent of M(state).
    llo
        Returns:
            NDArray: spectrum of the total matrix.
    """

    image_basis = image_algebra_basis(state.modes, state.photons, orthonormal)

    if order[0] == 1:
        density_tangent = projection_density(state, subspace="image", orthonormal=orthonormal)
        invariant = _vicent_matricial_invariant_matrix(density_tangent, image_basis)
    if order[1] == 1:
        density_orthogonal = projection_density(state, subspace="complement", orthonormal=orthonormal)
        invariant = _vicent_matricial_invariant_matrix(density_orthogonal, image_basis)
    if order[2] == 1:
        invariant = _vicent_matricial_invariant_matrix(state.density_matrix, image_basis)
    return np.linalg.eigvals(invariant).round(15)


def _vicent_matricial_invariant_matrix(
    state_matrix: NDArray, image_basis: BasisAlgebra
) -> NDArray:
    invariant = np.zeros_like(state_matrix)

    for basis_i in image_basis:
        invariant += basis_i @ basis_i @ state_matrix
    return invariant
