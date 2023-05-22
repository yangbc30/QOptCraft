"""Module docstrings.
"""
import numpy as np

from QOptCraft.state import State, PureState, MixedState
from QOptCraft import algebra_basis_sparse, gram_schmidt_modified, mat_inner_product


def can_transition(in_state: State, out_state: State) -> bool:
    """Check if we cannot transition from an input state to an output state
    through an optical network. It is just a necessary condition, so if the
    output is True, we cannot know if there is a transition matrix.
    """
    assert in_state.num_photons == out_state.num_photons
    assert in_state.num_modes == out_state.num_modes

    if isinstance(in_state, PureState) and isinstance(out_state, PureState):
        modes = in_state.num_modes
        photons = in_state.num_photons
    elif isinstance(in_state, MixedState) and isinstance(out_state, MixedState):
        # TODO: what if the state is a mixture of different number of photons??
        modes = in_state.num_modes[0]
        photons = in_state.num_photons[0]
    else:
        raise ValueError("Input states must be both PureState or both MixedState.")

    basis_img_algebra = algebra_basis_sparse(modes, photons)[1]
    orthonormal_basis = gram_schmidt_modified(basis_img_algebra)

    in_state_coefs = []
    out_state_coefs = []
    for basis_matrix in orthonormal_basis:
        in_state_coefs.append(mat_inner_product(1j * in_state.density_matrix, basis_matrix))
        out_state_coefs.append(mat_inner_product(1j * out_state.density_matrix, basis_matrix))

    in_energy = sum(np.abs(in_state_coefs) ** 2)
    out_energy = sum(np.abs(out_state_coefs) ** 2)

    print("In state energy ", in_energy)
    print("Out state energy ", out_energy)

    return in_energy == out_energy
