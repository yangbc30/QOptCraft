from typing import Literal

import numpy as np
from numpy.typing import NDArray

from qoptcraft.state import State
from qoptcraft.basis import image_algebra_basis, unitary_algebra_basis
from qoptcraft.math import Matrix, hs_scalar_product, hs_inner_product


def projection_density(
    state: State,
    subspace: list[NDArray] | Literal["preimage", "image", "complement", "full"] = "preimage",
    orthonormal: bool = False,
) -> Matrix:
    """Project a state onto the linear optical subalgebra or its complement.

    Args:
        state (State): a photonic quantum state.
        subspace (str): "image" or "complement".
        orthonormal (bool, optional): if true, it orthonormalizes the basis of linear optical
            hamiltonians before projecting onto it. Defaults to False.

    Returns:
        Matrix: the projected density matrix.
    """
    matrix = 1j * state.density_matrix

    if subspace == "preimage":
        unitary_basis = unitary_algebra_basis(state.modes)
        image_basis = image_algebra_basis(state.modes, state.photons, orthonormal)

        projection = np.zeros_like(unitary_basis[0])
        for unitary_matrix, image_matrix in zip(unitary_basis, image_basis, strict=True):
            projection += hs_scalar_product(image_matrix, matrix) * unitary_matrix
        return projection

    if isinstance(subspace, list):
        projection = np.zeros_like(matrix)
        for basis_matrix in subspace:
            projection += hs_scalar_product(basis_matrix, matrix) * basis_matrix
        return projection

    image_basis = image_algebra_basis(state.modes, state.photons, orthonormal)

    projection_image = np.zeros_like(matrix)
    for basis_matrix in image_basis:
        projection_image += hs_scalar_product(basis_matrix, matrix) * basis_matrix
    if subspace == "image":
        return projection_image
    if subspace == "complement" and orthonormal is True:
        return matrix - projection_image
    raise ValueError("The complement invariant only makes sense if the basis is orthonormal.")


def projection_coefs(state, orthonormal=False):
    image_basis = image_algebra_basis(state.modes, state.photons, orthonormal)

    coefs = []
    matrix = 1j * state.density_matrix
    for basis_matrix in image_basis:
        coefs.append(hs_inner_product(basis_matrix, matrix))
    return np.array(coefs)
