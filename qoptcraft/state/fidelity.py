"""Functions to compute the fidelity between mixed and pure states
"""

import numpy as np
from scipy.linalg import sqrtm

from .state import State, PureState


def fidelity(state_1: State, state_2: State) -> float:
    if isinstance(state_1, PureState) and isinstance(state_2, PureState):
        return np.abs(state_1.dot(state_2)) ** 2
    sqrt_density_1 = sqrtm(state_1.density_matrix)
    return np.real(np.trace(sqrtm(sqrt_density_1 @ state_2.density_matrix @ sqrt_density_1))) ** 2
