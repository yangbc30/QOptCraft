import numpy as np
from numpy.typing import NDArray
import scipy as sp
from scipy.special import factorial
import numba

from qoptcraft.state import Fock, PureState
from qoptcraft.basis import get_photon_basis, hilbert_dim
from qoptcraft.math import logm_3, permanent, permanent_ryser
from qoptcraft.evolution import photon_hamiltonian


def photon_unitary(scattering_matrix: NDArray, photons: int) -> NDArray:
    """Previously known as evolution_3.

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


def photon_unitary_hamiltonian(scattering_matrix: NDArray, photons: int) -> NDArray:
    """Previously known as evolution_3.

    Args:
        scattering_matrix (NDArray): _description_
        photons (int): number of photons.

    Returns:
        NDArray: _description_
    """
    S_hamiltonian = logm_3(scattering_matrix)
    U_hamiltonian = photon_hamiltonian(S_hamiltonian, photons)
    return sp.linalg.expm(U_hamiltonian)


@numba.jit(nopython=True)
def in_out_photon_matrix_numba(
    matrix: NDArray, fock_in: tuple[int, ...], fock_out: tuple[int, ...]
) -> NDArray:
    """
    Return a matrix with row index 'i' repeated 'row_rep' times
    and column index 'j' repeated 'col_rep' times.

    Parameters
    ----------
    matrix : (m, m) array
        Linear optical scattering matrix with m modes.

    Returns
    -------
    (n, n) array
        Optical scattering matrix with m modes and n photons.

    Todo:
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


@numba.jit(nopython=True)
def in_out_submatrix(
    matrix, fock_in: tuple[int, ...], fock_out: tuple[int, ...]
):
    """
    Return a matrix with row index 'i' repeated 'row_rep' times
    and column index 'j' repeated 'col_rep' times.

    Parameters
    ----------
    matrix : (m, m) array
        Linear optical scattering matrix with m modes.

    Returns
    -------
    (n, n) array
        Optical scattering matrix with m modes and n photons.

    """
    modes = len(fock_in)
    photons = sum(fock_in)
    interm_matrix = np.zeros((matrix.shape[0], photons), dtype="complex128")
    col = 0
    for mode in range(modes):
        for _ in range(fock_in[mode]):
            interm_matrix[:, col] = matrix[:, mode]
            col += 1
    final_matrix = np.zeros((photons, photons), dtype="complex128")
    row = 0
    for mode in range(modes):
        for _ in range(fock_out[mode]):
            final_matrix[row, :] = interm_matrix[mode, :]
            row += 1
    return final_matrix


# @numba.jit(nopython=True)
def photon_unitary_permanent(scattering_matrix: NDArray, photons: int) -> NDArray:
    """<S|phi(U)|T> = Per(U_ST) / sqrt(s1! ...sm! t1! ... tm!)"""
    modes = scattering_matrix.shape[0]
    dim = hilbert_dim(modes, photons)
    unitary = np.zeros((dim, dim), dtype=np.complex128)
    photon_basis = get_photon_basis(modes, photons)

    for col, fock_in in enumerate(photon_basis):
        for row, fock_out in enumerate(photon_basis):
            # sub_matrix = in_out_photon_matrix(scattering_matrix, fock_in, fock_out)
            sub_matrix = in_out_submatrix(scattering_matrix, fock_in, fock_out)
            coef = np.prod(factorial(fock_in)) * np.prod(factorial(fock_out))
            unitary[row, col] = permanent(sub_matrix) / np.sqrt(coef)

    return unitary  # minus global phase in concordance with the other methods


def photon_unitary_permanent_ryser(scattering_matrix: NDArray, photons: int) -> NDArray:
    """<S|phi(U)|T> = Per(U_ST) / sqrt(s1! ...sm! t1! ... tm!)"""
    modes = scattering_matrix.shape[0]
    dim = hilbert_dim(modes, photons)
    unitary = np.zeros((dim, dim), dtype=np.complex128)
    photon_basis = get_photon_basis(modes, photons)

    for col, fock_in in enumerate(photon_basis):
        for row, fock_out in enumerate(photon_basis):
            sub_matrix = in_out_submatrix(scattering_matrix, fock_in, fock_out)
            coef = np.prod(factorial(fock_in)) * np.prod(factorial(fock_out))
            unitary[row, col] = permanent_ryser(sub_matrix) / np.sqrt(coef)

    return unitary  # minus global phase in concordance with the other methods
