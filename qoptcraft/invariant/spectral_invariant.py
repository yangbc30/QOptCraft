from qoptcraft.state import State
from typing import Literal

import numpy as np
from numpy.typing import NDArray

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
        return state.density_matrix - projection_image
    raise ValueError("Supported options for the subspace are 'image' and 'complement'.")


def spectral_invariant(
    state: State, subspace: Literal["image", "complement", "full"]
) -> NDArray:
    """Calculate the photonic invariant for a given state.

    Args:
        state (State): a photonic quantum state.

    Returns:
        tuple[float, float]: tangent invariant.
    """
    if subspace == "image":
        projection = projection_density(state, subspace="image")
        return np.linalg.eigvals(projection)
    elif subspace == "complement":
        projection = projection_density(state, subspace="complement")
        return np.linalg.eigvals(projection)
    elif subspace == "full":
        projection_image = projection_density(state, subspace="image")
        projection_complement = projection_density(state, subspace="complement")
        spectrum_image = np.linalg.eigvals(projection_image)
        spectrum_complement = np.linalg.eigvals(projection_complement)
        return np.concatenate((spectrum_image, spectrum_complement))
    raise ValueError("Supported options for the subspace are 'image', 'complement' and 'full'.")
