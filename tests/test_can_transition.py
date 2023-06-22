import pytest
import numpy as np

from qoptcraft.state import PureState, State
from qoptcraft.invariant import can_transition

PHOTONS_1 = 2
PHOTONS_2 = 3
EXTRA = 1

BELL_STATE = PureState(
    [
        [1, 0, 1, 0] + [1] * (PHOTONS_1 - 2) + [0] * EXTRA,
        [0, 1, 0, 1] + [1] * (PHOTONS_1 - 2) + [0] * EXTRA,
    ],
    [1 / np.sqrt(2), 1 / np.sqrt(2)],
)
INPUT_STATE = PureState([[1, 0, 1, 0] + [1] * (PHOTONS_1 - 2) + [0] * EXTRA], [1])


HONG_HU_MANDEL_INPUT = PureState([[1, 1]], [1])
HONG_HU_MANDEL_OUTPUT = PureState([[2, 0], [0, 2]], [1 / np.sqrt(2), 1 / np.sqrt(2)])


@pytest.mark.parametrize(
    ("in_state", "out_state", "result"),
    ((BELL_STATE, INPUT_STATE, False), (HONG_HU_MANDEL_INPUT, HONG_HU_MANDEL_OUTPUT, True)),
)
def test_can_transition(in_state: State, out_state: State, result: bool) -> None:
    test_result = can_transition(in_state, out_state)
    assert result == test_result
