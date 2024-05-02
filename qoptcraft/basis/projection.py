from qoptcraft.state import State
from typing import Literal

import numpy as np

from qoptcraft.basis import basis_image_orthonormal
from qoptcraft.math import Matrix, hs_scalar_product


def projection_density(state: State, subspace: Literal["image", "complement"]) -> Matrix:
    basis_image = basis_image_orthonormal(state.modes, state.photons)
    matrix = 1j * state.density_matrix
    projection_image = np.zeros_like(matrix)
    for basis_matrix in basis_image:
        projection_image += hs_scalar_product(basis_matrix, matrix) * basis_matrix
    if subspace == "image":
        return projection_image
    if subspace == "complement":
        return matrix - projection_image
    raise ValueError("Supported options for the subspace are 'image' and 'complement'.")
