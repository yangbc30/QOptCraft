"""Module docstrings.
"""
import numpy as np
from scipy.special import factorial as fact

from QOptCraft.state import State, PureState
from .mat_inner_product import mat_inner_product
from .gram_schmidt import gram_schmidt
from QOptCraft.basis import get_algebra_basis


def invariant(state: State) -> tuple[float, float]:
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
    orthogonal = mat_inner_product(state.density_matrix, state.density_matrix) - tangent
    return tangent, orthogonal


def can_transition(state_in: State, state_out: State) -> bool:
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

    tangent_in, orthogonal_in = invariant(state_in)
    tangent_out, orthogonal_out = invariant(state_out)

    print("The values of the invariants are:")
    print(f"{tangent_in = :.7f} \t\t {tangent_out = :.7f}")
    print(f"{orthogonal_in = :.7f} \t\t {orthogonal_out = :.7f}")
    return np.isclose(tangent_in, tangent_out)


def invariant_no_basis(state_in: PureState, state_out: PureState) -> tuple[float, float]:
    """Calculate the photonic invariant for a given state without using the Hilbert space basis.
    This makes the calculations much faster and memory efficient.

    Args:
        state (State): a photonic quantum state.

    Returns:
        tuple[float, float]: tangent and orthogonal invariants.
    """
    assert state_in.photons == state_out.photons, "Number of photons don't coincide."
    assert state_in.modes == state_out.modes, "Number of modes don't coincide."

    modes = state_in.modes
    in_ = 0  # Invariant for state_in
    out = 0  # Invariant for state_out

    for mode_1 in range(modes):
        for mode_2 in range(mode_1 + 1, modes):
            in_ += state_in.exp_photons(mode_1, mode_2) * state_in.exp_photons(mode_2, mode_1)
            in_ -= state_in.exp_photons(mode_1, mode_1) * state_in.exp_photons(mode_2, mode_2)

            out += state_out.exp_photons(mode_1, mode_2) * state_out.exp_photons(mode_2, mode_1)
            out -= state_out.exp_photons(mode_1, mode_1) * state_out.exp_photons(mode_2, mode_2)

    in_invariant = round(in_, 7)
    out_invariant = round(out, 7)

    return in_invariant, out_invariant


def can_transition_no_basis(state_in: PureState, state_out: PureState) -> bool:
    """Check if we cannot transition from an input state to an output state
    through an optical network. The function tests if the invariant defined in
    corollary 3 is conserved.
    """
    in_invariant, out_invariant = invariant_no_basis(state_in, state_out)

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

def can_transition_no_basis(state_in: PureState, state_out: PureState) -> bool:
    """Check if we cannot transition from an input state to an output state
    through an optical network. The function tests if the invariant defined in
    corollary 3 is conserved.
    """
    in_invariant, out_invariant = invariant_no_basis(state_in, state_out)

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
