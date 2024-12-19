"""Module docstrings.
"""

from typing import Literal

import numpy as np
from scipy.special import factorial as fact

from qoptcraft.state import State, PureState
from qoptcraft.math import hs_scalar_product, gram_schmidt_generator
from qoptcraft.basis import image_algebra_basis


def photon_invariant(
    state: State, method: Literal["basis", "no basis", "reduced"] = "basis"
) -> float:
    """Photonic invariant for a given state.

    Args:
        state (State): a photonic quantum state.
        method (str): method to calculate the invariant. Options are 'reduced',
            'no basis', 'basis'. Default is 'basis'.

    Returns:
        float: invariant.
    """
    if method == "basis":
        return photon_invariant_basis(state)
    if not isinstance(state, PureState):
        raise ValueError("Non pure states only accept method basis.")
    if method == "reduced":
        return photon_invariant_reduced(state)
    if method == "no basis":
        return photon_invariant_no_basis(state)
    raise ValueError("Options for 'method' are 'reduced', 'no basis' or 'basis'.")


def photon_invariant_reduced(state: PureState) -> float:
    """Calculate the reduced photonic invariant for a given state without using the
    Hilbert space basis.

    Note:
        Calculations without using the basis are much faster and memory efficient.

    Args:
        state (State): a photonic quantum state.

    Returns:
        float: invariant.
    """
    modes = state.modes
    invariant = 0
    for mode_1 in range(modes):
        for mode_2 in range(mode_1 + 1, modes):
            invariant += state.exp_photons(mode_1, mode_2) * state.exp_photons(mode_2, mode_1)
            invariant -= state.exp_photons(mode_1, mode_1) * state.exp_photons(mode_2, mode_2)
    return invariant.real


def photon_invariant_no_basis(state: PureState) -> float:
    """Calculate the photonic invariant for a given state without using the
    Hilbert space basis.

    Note:
        Calculations without using the basis are much faster and memory efficient.

    Args:
        state (State): a photonic quantum state.

    Returns:
        float: invariant.
    """
    invariant = photon_invariant_reduced(state)
    modes = state.modes
    photons = state.photons
    C1 = (modes * photons + 1) * fact(modes) * fact(photons) / fact(modes + photons)
    C2 = 2 * fact(modes + 1) * fact(photons - 1) / fact(modes + photons)
    if np.isnan(C1):
        C1 = 0
    if np.isnan(C2):
        C2 = 0
    return C1 + C2 * invariant


def photon_invariant_basis(state: State) -> tuple[float, float]:
    """Calculate the photonic invariant for a given state.

    Args:
        state (State): a photonic quantum state.

    Returns:
        tuple[float, float]: tangent invariant.
    """
    basis_img_algebra = image_algebra_basis(state.modes, state.photons)
    coefs = []
    for basis_matrix in gram_schmidt_generator(basis_img_algebra):
        coefs.append(hs_scalar_product(1j * state.density_matrix, basis_matrix))
    tangent_invariant = sum(np.abs(coefs) ** 2)
    return tangent_invariant
