import numpy as np
from numpy.typing import NDArray
from scipy.special import factorial

from qoptcraft.state import Fock, PureState
from qoptcraft.basis import get_photon_basis, hilbert_dim


def photon_unitary(scattering_matrix: NDArray, photons: int) -> NDArray:
    """Standard evolution given by quantum mechanics.

    Args:
        scattering_matrix (NDArray): _description_
        photons (int): number of photons.

    Returns:
        NDArray: _description_
    """
    modes = scattering_matrix.shape[0]
    dim = hilbert_dim(modes, photons)
    unitary = np.zeros((dim, dim), dtype=np.complex64)
    photon_basis = get_photon_basis(modes, photons)

    for col, fock_in in enumerate(photon_basis):
        unitary[:, col] = fock_evolution(scattering_matrix, fock_in).state_in_basis()
    return unitary


def fock_evolution(scattering_matrix: NDArray, fock_in: tuple[int, ...]) -> PureState:
    """Evolution of a single Fock state using the definition given by basic
    quantum mechanics.

    Args:
        scattering_matrix (NDArray): _description_
        fock_in (Fock): _description_

    Returns:
        PureState: _description_
    """
    modes = len(fock_in)

    state_out: PureState = Fock(*([0] * modes))
    for mode_2 in range(modes):
        for _ in range(fock_in[mode_2]):
            state_aux = state_out
            state_out = scattering_matrix[0, mode_2] * state_aux.creation(0)
            for mode_1 in range(1, modes):
                state_out += scattering_matrix[mode_1, mode_2] * state_aux.creation(mode_1)
    coef = np.prod(factorial(fock_in))
    return state_out / np.sqrt(coef)
