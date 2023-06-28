import numpy as np
from numpy.typing import NDArray
from scipy.special import factorial
import numba

from qoptcraft.basis import get_photon_basis, hilbert_dim, BasisPhoton
from qoptcraft.math import permanent_glynn, permanent_ryser


def photon_unitary_permanent(
    scattering_matrix: NDArray, photons: int, method: str = "glynn"
) -> NDArray:
    """<S|phi(U)|T> = Per(U_ST) / sqrt(s1! ...sm! t1! ... tm!)"""
    modes = scattering_matrix.shape[0]
    dim = hilbert_dim(modes, photons)
    unitary = np.empty((dim, dim), dtype=np.complex128)
    photon_basis = get_photon_basis(modes, photons)

    for col, fock_in in enumerate(photon_basis):
        unitary[:, col] = fock_evolution_permanent(scattering_matrix, fock_in, method, photon_basis)
    return unitary


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
    if photon_basis is None:
        photon_basis = get_photon_basis(len(fock_in), sum(fock_in))
    dim = len(photon_basis)
    state_out = np.empty(dim, dtype=np.complex128)

    if method == "glynn":
        permanent = permanent_glynn
    elif method == "ryser":
        permanent = permanent_ryser
    else:
        raise ValueError("Method argument only supports options 'glynn' and 'ryser'.")

    for row, fock_out in enumerate(photon_basis):
        sub_matrix = in_out_submatrix(scattering_matrix, fock_in, fock_out)
        coef = np.prod(factorial(fock_in)) * np.prod(factorial(fock_out))
        state_out[row] = permanent(sub_matrix) / np.sqrt(coef)
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
