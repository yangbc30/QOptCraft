from typing import Literal

import numpy as np

from qoptcraft.state import State
from qoptcraft.basis import image_algebra_basis
from qoptcraft.math import Matrix, hs_scalar_product, hs_inner_product


def projection_density(
    state: State, subspace: Literal["image", "complement"] = "image", orthonormal: bool = False
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
    if subspace != "image" and subspace != "complement":
        raise ValueError("Supported options for the subspace are 'image' and 'complement'.")

    image_basis = image_algebra_basis(state.modes, state.photons, orthonormal)

    matrix = 1j * state.density_matrix
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
