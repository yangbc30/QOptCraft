from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.special import factorial
import numba

from qoptcraft.state import Fock, PureState
from qoptcraft.basis import get_photon_basis, BasisPhoton
from qoptcraft.math import permanent


def fock_evolution(
    scattering_matrix: NDArray,
    fock_in: tuple[int, ...],
    method: Literal["heisenberg", "permanent glynn", "permanent ryser"] = "permanent glynn",
) -> PureState:
    """Evolution of a single Fock state using the definition given by basic
    quantum mechanics.

    Args:
        scattering_matrix (NDArray): _description_
        fock_in (Fock): _description_
        method (str): method to calculate the evolution of the Fock state. Options are
            'heisenberg', 'permanent glynn' or 'permanent ryser'. Default is 'permanent glynn'.

    Returns:
        PureState: _description_
    """
    if method == "heisenberg":
        return fock_evolution_heisenberg(scattering_matrix, fock_in)
    if method.split()[0] == "permanent":
        return fock_evolution_permanent(scattering_matrix, fock_in, method=method.split()[1])
    raise ValueError("Options for method are 'heisenberg', 'permanent glynn' or 'permanent ryser'.")


def fock_evolution_heisenberg(scattering_matrix: NDArray, fock_in: tuple[int, ...]) -> PureState:
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


def fock_evolution_permanent(
    scattering_matrix: NDArray,
    fock_in: tuple[int, ...],
    method: str = "glynn",
    photon_basis: BasisPhoton = None,
) -> NDArray:
    """Evolution of a single Fock state using the definition given by basic
    quantum mechanics.

    Args:
        scattering_matrix (NDArray): _description_
        fock_in (Fock): _description_

    Returns:
        PureState: _description_
    """
    if len(fock_in) != scattering_matrix.shape[0]:
        raise ValueError("Dimension of scattering_matrix and number of modes don't match.")
    if photon_basis is None:
        photon_basis = get_photon_basis(len(fock_in), sum(fock_in))
    dim = len(photon_basis)
    state_out = np.empty(dim, dtype=np.complex128)

    for row, fock_out in enumerate(photon_basis):
        sub_matrix = in_out_submatrix(scattering_matrix, fock_in, fock_out)
        coef = np.prod(factorial(fock_in)) * np.prod(factorial(fock_out))
        state_out[row] = permanent(sub_matrix, method=method) / np.sqrt(coef)
    return state_out


@numba.jit(nopython=True)
def in_out_submatrix(matrix, fock_in: tuple[int, ...], fock_out: tuple[int, ...]):
    """Return a matrix with row index 'i' repeated 'row_rep' times
    and column index 'j' repeated 'col_rep' times.

    Args:
        matrix ((m, m) array): Linear optical scattering matrix with m modes.

    Returns:
        (n, n) array: Optical scattering matrix with m modes and n photons.
    """
    modes = len(fock_in)
    photons = sum(fock_in)
    interm_matrix = np.empty((matrix.shape[0], photons), dtype="complex128")
    col = 0
    for mode in range(modes):
        for _ in range(fock_in[mode]):
            interm_matrix[:, col] = matrix[:, mode]
            col += 1
    final_matrix = np.empty((photons, photons), dtype="complex128")
    row = 0
    for mode in range(modes):
        for _ in range(fock_out[mode]):
            final_matrix[row, :] = interm_matrix[mode, :]
            row += 1
    return final_matrix


@numba.jit(nopython=True)
def in_out_submatrix_alt(
    matrix: NDArray, fock_in: tuple[int, ...], fock_out: tuple[int, ...]
) -> NDArray:
    """Return a matrix with row index 'i' repeated 'row_rep' times
    and column index 'j' repeated 'col_rep' times.

    Args:
        matrix ((m, m) array): Linear optical scattering matrix with m modes.

    Returns:
        (n, n) array: Optical scattering matrix with m modes and n photons.

    TODO:
        Revise this function, doesn't work properly
    """
    # assert sum(fock_in) == sum(fock_out), "Error: number of photons must be conserved."
    photons = sum(fock_in)
    final_matrix = np.empty((photons, photons), dtype=np.complex128)
    in_photon_count = 0
    in_photon_idx = 0
    cumulative_in_fock = np.cumsum(np.array(fock_in))
    cumulative_out_fock = np.cumsum(np.array(fock_out))

    for i in range(photons):
        while cumulative_in_fock[in_photon_idx] <= in_photon_count:
            in_photon_idx += 1
        out_photon_idx = 0
        out_photon_count = 0
        for j in range(photons):
            while cumulative_out_fock[out_photon_idx] <= out_photon_count:
                out_photon_idx += 1
            final_matrix[i, j] = matrix[in_photon_idx, out_photon_idx]
            out_photon_count += 1
        in_photon_count += 1

    return final_matrix
