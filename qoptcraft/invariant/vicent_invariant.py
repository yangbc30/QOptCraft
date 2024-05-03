"""Migdał et al. invariants

References:
    Migdał et al. Multiphoton states related via linear optics.
    https://arxiv.org/abs/1403.3069
"""

import numpy as np
from numpy.typing import NDArray
from numpy.linalg import matrix_power

from qoptcraft.state import PureState
from qoptcraft.basis import get_image_algebra_basis, projection_density, BasisAlgebra


def vicent_invariant(state: PureState, order: tuple[int,int,int] = (1,0,0)) -> NDArray:
    """Calculate Vicent's invariant, which uses the commutators of (projected) density matrices
    with the basis of passive linear optical Hamiltonians.

    Args:
        state (State): a photonic quantum state.
        order (tuple[int,int,int]): The first integer: exponent of M(state_t); second: exponent
            of M(state_o); third: exponent of M(state).

    Returns:
        NDArray: spectrum of the total matrix.
    """

    algebra_basis = get_image_algebra_basis(state.modes, state.photons)
    invariant = np.identity(len(algebra_basis), dtype=np.complex64)

    if order[0] != 0:
        density_tangent = projection_density(state, subspace="image")
        # print(f"{density_tangent = }")
        invariant_tangent = _vicent_invariant_matrix(density_tangent, algebra_basis)
        invariant @= matrix_power(invariant_tangent, order[0])
    if order[1] != 0:
        density_orthogonal = projection_density(state, subspace="complement")
        # print(f"{density_orthogonal = }")
        invariant_orthogonal = _vicent_invariant_matrix(density_orthogonal, algebra_basis)
        invariant @= matrix_power(invariant_orthogonal, order[1])
    if order[2] != 0:
        # print(f"{state.density_matrix = }")
        invariant_density = _vicent_invariant_matrix(state.density_matrix, algebra_basis)
        invariant @= matrix_power(invariant_density, order[2])

    return np.linalg.eigvals(invariant).round(23)


def _vicent_invariant_matrix(state_matrix: NDArray, algebra_basis: BasisAlgebra) -> NDArray:
    """Calculate the matrix M_ij = Tr([basis_i, state][basis_j, state])).

    Args:
        state (State): a photonic quantum state.
        order (int): order k of the correlator.

    Returns:
        NDArray: matrix whose espectrum is invariant.
    """
    dim = len(algebra_basis)
    invariant = np.zeros((dim, dim), dtype=np.complex64)

    for i, basis_i in enumerate(algebra_basis):
        for j, basis_j in enumerate(algebra_basis):
            commutator_i = basis_i @ state_matrix - state_matrix @ basis_i
            commutator_j = basis_j @ state_matrix - state_matrix @ basis_j
            invariant[i, j] = np.trace(commutator_i @ commutator_j)
    print(f"{invariant.round(23) = }")
    return invariant
