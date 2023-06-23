"""Module docstrings.
"""
import numpy as np

from qoptcraft.state import State, PureState
from .invariant import photon_invariant, photon_invariant_full, photon_invariant_basis


def can_transition(state_in: PureState, state_out: PureState) -> bool:
    """Check if we cannot transition from an input state to an output state
    through an optical network. Calculations are done with the reduced invariant,
    which is faster and more efficient.

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

    print(f"In reduced invariant = {in_invariant} \t Out reduced invariant = {out_invariant}")
    return np.isclose(in_invariant, out_invariant)


def can_transition_full(state_in: PureState, state_out: PureState) -> bool:
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

    in_invariant = photon_invariant_full(state_in)
    out_invariant = photon_invariant_full(state_out)

    print(f"In full invariant = {in_invariant} \t Out full invariant = {out_invariant}")
    return np.isclose(in_invariant, out_invariant)


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
