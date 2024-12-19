from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.special import factorial
import numba

from qoptcraft.state import Fock, PureState
from qoptcraft.basis import photon_basis, BasisPhoton
from qoptcraft.math import permanent


def fock_evolution(
    scattering_matrix: NDArray,
    fock_in: tuple[int, ...],
    method: Literal["heisenberg", "permanent glynn", "permanent ryser"] = "permanent glynn",
) -> NDArray:
    """Evolution of a single Fock state using the definition given by basic
    quantum mechanics.

    Args:
        scattering_matrix (NDArray): scattering matrix of a linear optical interferometer.
        fock_in (Fock): fock state to evolve.
        method (str): method to calculate the evolution of the Fock state. Options are
            'heisenberg', 'permanent glynn' or 'permanent ryser'. Default is 'permanent glynn'.

    Returns:
        NDArray: fock state given in the photon basis.
    """
    if method == "heisenberg":
        return fock_evolution_heisenberg(scattering_matrix, fock_in)
    if method.split()[0] == "permanent":
        return fock_evolution_permanent(scattering_matrix, fock_in, method=method.split()[1])
    raise ValueError("Options for method are 'heisenberg', 'permanent glynn' or 'permanent ryser'.")


def fock_evolution_heisenberg(scattering_matrix: NDArray, fock_in: tuple[int, ...]) -> NDArray:
    """Evolution of a single Fock state using the definition given by basic
    quantum mechanics.

    Args:
        scattering_matrix (NDArray): scattering matrix of a linear optical interferometer.
        fock_in (Fock): fock state to evolve.

    Returns:
        NDArray: fock state given in the photon basis.
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
    method: Literal["glynn", "ryser"] = "glynn",
    photon_basis: BasisPhoton = None,
) -> NDArray:
    """Evolution of a single Fock state using the definition given by basic
    quantum mechanics.

    Args:
        scattering_matrix (NDArray): scattering matrix of a linear optical interferometer.
        fock_in (Fock): fock state to evolve.
        method (str): method to compute the permanent. Must be 'glynn' or 'ryser'.
            Defaults to 'glynn'.

    Returns:
        NDArray: fock state given in the photon basis.
    """
    if len(fock_in) != scattering_matrix.shape[0]:
        raise ValueError("Dimension of scattering_matrix and number of modes don't match.")
    if photon_basis is None:
        photon_basis = photon_basis(len(fock_in), sum(fock_in))
    dim = len(photon_basis)
    state_out = np.empty(dim, dtype=np.complex128)

    for row, fock_out in enumerate(photon_basis):
        sub_matrix = in_out_submatrix(scattering_matrix, fock_in, fock_out)
        coef = np.prod(factorial(fock_in)) * np.prod(factorial(fock_out))
        state_out[row] = permanent(sub_matrix, method=method) / np.sqrt(coef)
    return state_out


@numba.jit(nopython=True)
def in_out_submatrix(matrix, fock_in: tuple[int, ...], fock_out: tuple[int, ...]) -> NDArray:
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
