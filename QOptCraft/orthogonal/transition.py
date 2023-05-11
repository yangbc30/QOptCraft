"""Module docstrings.
"""

from QOptCraft.state import State


def can_transition(in_state: State, out_state: State) -> bool:
    """Check if we cannot transition from an input state to an output state
    through an optical network. It is just a necessary condition, so if the
    output is True, we cannot know if there is a transition matrix.
    """
    assert in_state.num_photons == out_state.num_photons
    assert in_state.num_modes == out_state.num_modes
