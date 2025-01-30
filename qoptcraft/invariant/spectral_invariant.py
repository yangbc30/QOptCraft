from typing import Literal

import numpy as np
from numpy.typing import NDArray

from qoptcraft.state import State
from .projection import projection_density


def spectral_invariant(
    state: State, subspace: Literal["image", "complement", "full"] = "image", orthonormal=False
) -> NDArray:
    """Calculate the photonic invariant for a given state.

    Args:
        state (State): a photonic quantum state.

    Returns:
        tuple[float, float]: tangent invariant.
    """
    if subspace == "image":
        projection = projection_density(state, subspace="image", orthonormal=orthonormal)
        return np.linalg.eigvals(projection).imag
    elif subspace == "complement":
        projection = projection_density(state, subspace="complement", orthonormal=orthonormal)
        return np.linalg.eigvals(projection).imag
    elif subspace == "full":
        projection_image = projection_density(state, subspace="image", orthonormal=orthonormal)
        projection_complement = projection_density(
            state, subspace="complement", orthonormal=orthonormal
        )
        spectrum_image = np.linalg.eigvals(projection_image)
        spectrum_complement = np.linalg.eigvals(projection_complement)
        return np.concatenate((spectrum_image, spectrum_complement))
    raise ValueError("Supported options for the subspace are 'image', 'complement' and 'full'.")
