"""Module docstrings.
"""
import numpy as np
from scipy.special import factorial as fact

from qoptcraft.state import State, PureState
from qoptcraft.math import mat_inner_product, mat_norm, gram_schmidt
from qoptcraft.basis import get_algebra_basis


def photon_invariant(state: PureState) -> float:
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


def photon_invariant_full(state: PureState) -> float:
    """Calculate the photonic invariant for a given state without using the
    Hilbert space basis.

    Note:
        Calculations without using the basis are much faster and memory efficient.

    Args:
        state (State): a photonic quantum state.

    Returns:
        float: invariant.
    """
    invariant = photon_invariant(state)
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
        tuple[float, float]: tangent and orthogonal invariants.
    """
    basis_img_algebra = get_algebra_basis(state.modes, state.photons)[1]
    orthonormal_basis = gram_schmidt(basis_img_algebra)
    coefs = []
    for basis_matrix in orthonormal_basis:
        coefs.append(mat_inner_product(1j * state.density_matrix, basis_matrix))
    tangent = sum(np.abs(coefs) ** 2)
    orthogonal = mat_norm(state.density_matrix) ** 2 - tangent
    return tangent, orthogonal
