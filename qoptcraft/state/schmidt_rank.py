import numpy as np

from qoptcraft.basis import BasisPhoton
from qoptcraft.state import PureState


def schmidt_rank(state: PureState, basis_1: BasisPhoton, basis_2: BasisPhoton):
    """Calculate the Schmidt rank of a state."""
    M = np.zeros((len(basis_1), len(basis_2)), dtype=np.complex128)

    for i, vec_1 in enumerate(basis_1):
        for j, vec_2 in enumerate(basis_2):
            M[i, j] = np.vdot(np.kron(vec_1, vec_2), state)  # ! qué calcula vdot??

    u, s, vh = np.linalg.svd(M, full_matrices=False)

    return len(s) - np.sum(np.isclose(s, np.zeros_like(s)))
