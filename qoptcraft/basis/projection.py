from qoptcraft.state import State
from typing import Literal

import numpy as np

from qoptcraft.basis import basis_image_orthonormal, get_image_algebra_basis
from qoptcraft.math import Matrix, hs_scalar_product


def projection_density(state: State, subspace: Literal["image", "complement"], orthonormal: bool = False) -> Matrix:
    """Project a state onto the linear optical subalgebra or its complement.

    Args:
        state (State): a photonic quantum state.
        subspace (str): "image" or "complement".
        orthonormal (bool, optional): if true, it orthonormalizes the basis of linear optical hamiltonians
            before projecting onto it. Defaults to False.

    Returns:
        Matrix: the projected density matrix.
    """
    if orthonormal:
        basis_image = basis_image_orthonormal(state.modes, state.photons)
    else:
        basis_image = get_image_algebra_basis(state.modes, state.photons)
    matrix = 1j * state.density_matrix
    projection_image = np.zeros_like(matrix)
    for basis_matrix in basis_image:
        projection_image += hs_scalar_product(basis_matrix, matrix) * basis_matrix
    if subspace == "image":
        return projection_image
    if subspace == "complement":
        return matrix - projection_image
    raise ValueError("Supported options for the subspace are 'image' and 'complement'.")
