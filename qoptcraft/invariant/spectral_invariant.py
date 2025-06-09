from typing import Literal

import numpy as np
from numpy.typing import NDArray

from qoptcraft.state import State

from .projection import higher_order_projection_density, projection_density


def spectral_invariant(
    state: State,
    subspace: Literal["preimage", "image", "complement", "full"] | list[NDArray] = "preimage",
    orthonormal=False,
) -> NDArray:
    """Calculate the photonic invariant for a given state.

    Args:
        state (State): a photonic quantum state.
        subspace (Literal or List(NDArray)): basis of the subspace

    Returns:
        tuple[float, float]: tangent invariant.
    """
    if subspace == "full":
        projection_image = projection_density(state, subspace="image", orthonormal=orthonormal)
        projection_complement = projection_density(
            state, subspace="complement", orthonormal=orthonormal
        )
        spectrum_image = np.linalg.eigvalsh(projection_image)
        spectrum_complement = np.linalg.eigvalsh(projection_complement)
        return np.concatenate((spectrum_image, spectrum_complement))

    projection = projection_density(state, subspace=subspace, orthonormal=orthonormal)
    return np.linalg.eigvalsh(projection)


def higher_order_spectral_invariant(
    state: State, order: int, subspace: Literal["preimage", "image"] = "image", orthonormal=False
) -> NDArray:
    projection = higher_order_projection_density(state, order, subspace, orthonormal)
    return np.linalg.eigvalsh(projection)
