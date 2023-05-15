"""Module docstrings.
"""
import numpy as np
from scipy.special import comb

from QOptCraft.state import State
from QOptCraft import algebra_basis_sparse, gram_schmidt_modified, mat_inner_product


def can_transition(in_state: State, out_state: State) -> bool:
    """Check if we cannot transition from an input state to an output state
    through an optical network. It is just a necessary condition, so if the
    output is True, we cannot know if there is a transition matrix.
    """
    assert in_state.num_photons == out_state.num_photons
    assert in_state.num_modes == out_state.num_modes

    modes = in_state.num_modes
    photons = in_state.num_photons
    dim = int(comb(modes + photons - 1, photons))

    basis_img_algebra = algebra_basis_sparse(modes, dim, photons)[1]
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
