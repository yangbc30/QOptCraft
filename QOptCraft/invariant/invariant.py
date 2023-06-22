"""Module docstrings.
"""
import numpy as np
from scipy.special import factorial as fact

from QOptCraft.state import State, PureState
from QOptCraft.math import mat_inner_product, mat_norm, gram_schmidt
from QOptCraft.basis import get_algebra_basis


def photon_invariant(state: PureState) -> float:
    """Calculate the photonic invariant for a given state without using the Hilbert space basis.
    This makes the calculations much faster and memory efficient.

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


def can_transition(state_in: PureState, state_out: PureState) -> bool:
    """Check if we cannot transition from an input state to an output state
    through an optical network. Calculations are done without a basis of the Hilbert space,
    so they are faster and more efficient.

    Note:
        This criterion is necessary, but not sufficient. Even if the transition
        doesn't violate the criterion (`True` output), a unitary may not exist.

    Args:
        state_in (State): input state of the optical circuit.
        state_out (State): desired output state of the optical circuit.

    Returns:
        bool: False if the transition is impossible. True if it is possible.
    """
    assert state_in.photons == state_out.photons, "Number of photons don't coincide."
    assert state_in.modes == state_out.modes, "Number of modes don't coincide."

    in_invariant = photon_invariant(state_in)
    out_invariant = photon_invariant(state_out)

    modes = state_in.modes
    photons = state_in.photons

    photons = state_in.photons
    C1 = (modes * photons + 1) * fact(modes) * fact(photons) / fact(modes + photons)
    C2 = 2 * fact(modes + 1) * fact(photons - 1) / fact(modes + photons)
    if np.isnan(C1):
        C1 = 0
    if np.isnan(C2):
        C2 = 0
    print(f"\nIn invariant = {C1 + C2 * in_invariant} \t Out invariant = {C1 + C2 * out_invariant}")
    return np.isclose(in_invariant, out_invariant)


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


def can_transition_basis(state_in: State, state_out: State) -> bool:
    """Check if we cannot transition from an input state to an output state
    through an optical network.

    Note:
        This criterion is necessary, but not sufficient. Even if the transition
        doesn't violate the criterion (`True` output), a unitary may not exist.

    Args:
        state_in (State): input state of the optical circuit.
        state_out (State): desired output state of the optical circuit.

    Returns:
        bool: False if the transition is impossible. True if it is possible.
    """
    assert state_in.photons == state_out.photons, "Number of photons don't coincide."
    assert state_in.modes == state_out.modes, "Number of modes don't coincide."

    tangent_in, orthogonal_in = photon_invariant_basis(state_in)
    tangent_out, orthogonal_out = photon_invariant_basis(state_out)

    print("The values of the invariants are:")
    print(f"{tangent_in = :.7f} \t\t {tangent_out = :.7f}")
    print(f"{orthogonal_in = :.7f} \t\t {orthogonal_out = :.7f}")
    return np.isclose(tangent_in, tangent_out)
