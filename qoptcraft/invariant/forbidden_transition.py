"""Check if a transition violates the necessary criterion
of the invariant conservation.
"""
from typing import Literal

import numpy as np

from qoptcraft.state import State, PureState
from .invariant import photon_invariant_reduced, photon_invariant_no_basis, photon_invariant_basis


def forbidden_transition(
    state_in: State,
    state_out: State,
    method: Literal["reduced", "no basis", "basis"] = "basis",
    print_invariant: bool = True,
) -> bool:
    """Check if we cannot transition from an input state to an output state
    through an optical network.

    Args:
        state_in (State): input state of the optical circuit.
        state_out (State): desired output state of the optical circuit.
        method (str): method to calculate the invariant. Options are 'reduced',
            'full', 'basis'. Default is 'basis'.

    Returns:
        bool: True if the transition is forbidden. True if it is not.
    """
    if method == "basis":
        return forbidden_transition_basis(state_in, state_out, print_invariant=print_invariant)
    if not isinstance(state_in, PureState) or not isinstance(state_out, PureState):
        raise ValueError("Non pure states only accept method basis.")
    if method == "reduced":
        return forbidden_transition_reduced(state_in, state_out, print_invariant=print_invariant)
    if method == "no basis":
        return forbidden_transition_no_basis(state_in, state_out, print_invariant=print_invariant)
    raise ValueError("Options for 'method' are 'reduced', 'no basis' or 'basis'.")


def forbidden_transition_reduced(
    state_in: PureState, state_out: PureState, print_invariant: bool = True
) -> bool:
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
        bool: True if the transition is forbidden. True if it is not.
    """
    assert state_in.photons == state_out.photons, "Number of photons don't coincide."
    assert state_in.modes == state_out.modes, "Number of modes don't coincide."

    in_invariant = photon_invariant_reduced(state_in)
    out_invariant = photon_invariant_reduced(state_out)

    if print_invariant:
        print(
            f"In reduced invariant = {in_invariant:.7f}"
            f"\t Out reduced invariant = {out_invariant:.7f}"
        )
    return not np.isclose(in_invariant, out_invariant)


def forbidden_transition_no_basis(
    state_in: PureState, state_out: PureState, print_invariant: bool = True
) -> bool:
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
        bool: True if the transition is forbidden. True if it is not.
    """
    assert state_in.photons == state_out.photons, "Number of photons don't coincide."
    assert state_in.modes == state_out.modes, "Number of modes don't coincide."

    in_invariant = photon_invariant_no_basis(state_in)
    out_invariant = photon_invariant_no_basis(state_out)
    if print_invariant:
        print(f"In full invariant = {in_invariant:.7f} \t Out full invariant = {out_invariant:.7f}")
    return not np.isclose(in_invariant, out_invariant)


def forbidden_transition_basis(
    state_in: State, state_out: State, print_invariant: bool = True
) -> bool:
    """Check if we cannot transition from an input state to an output state
    through an optical network.

    Note:
        This criterion is necessary, but not sufficient. Even if the transition
        doesn't violate the criterion (`True` output), a unitary may not exist.

    Args:
        state_in (State): input state of the optical circuit.
        state_out (State): desired output state of the optical circuit.

    Returns:
        bool: True if the transition is forbidden. True if it is not.
    """
    assert state_in.photons == state_out.photons, "Number of photons don't coincide."
    assert state_in.modes == state_out.modes, "Number of modes don't coincide."

    in_invariant = photon_invariant_basis(state_in)
    out_invariant = photon_invariant_basis(state_out)

    if print_invariant:
        print(f"In invariant = {in_invariant:.7f} \t Out invariant = {out_invariant:.7f}")
    return not np.isclose(in_invariant, out_invariant)
