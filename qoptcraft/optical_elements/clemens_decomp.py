"""Decompose a unitary into beamsplitters following Clemens et al.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import fsolve

from .optical_elements import beam_splitter


def clemens_decomposition(unitary: NDArray) -> tuple[list[NDArray], NDArray, list[NDArray]]:
    """Given a unitary matrix calculates the Clemens et al. decompositon
    into beam splitters and phase shifters:

    D = L_n ... L_1 · U · R_1.inv ... R_n.inv  =>
    => U = L_1.inv ... L_n.inv · D · R_n ... R_1

    Args:
        unitary (NDArray): unitary matrix to decompose.

    Returns:
        list[NDArray]: list of matrices that decompose U in the order of the decomposition

    References:
        The algorithm can be found in [1]_.

    .. [1] Clements et al., "An Optimal Design for Universal Multiport
        Interferometers" arXiv, 2007. https://arxiv.org/pdf/1603.08788.pdf
    """

    assert unitary.shape[0] == unitary.shape[1], "The matrix is not square"
    dim = unitary.shape[0]

    right_list = []
    left_list = []

    for i in range(1, dim):
        if (i % 2) == 1:
            for j in range(i):
                # substract 1 to all indices to match the paper with python arrays
                row = dim - j - 1
                col = i - j - 1
                mode_1 = i - j - 1  # mode_1 must equal col
                mode_2 = i - j

                # We calculate the beamsplitter angles phi y theta
                angle, shift = _solve_angles(unitary, mode_1, mode_2, row, col, is_odd=True)
                R = beam_splitter(angle, shift, dim, mode_1, mode_2, convention="clemens")
                unitary = unitary @ R.conj().T
                right_list.append(R)
        else:
            for j in range(1, i + 1):
                # substract 1 to all indices to match the paper with python arrays
                row = dim + j - i - 1
                col = j - 1
                mode_1 = dim + j - i - 2
                mode_2 = dim + j - i - 1  # mode_2 must equal row

                # We calculate the beamsplitter angles phi y theta
                angle, shift = _solve_angles(unitary, mode_1, mode_2, row, col, is_odd=False)
                left = beam_splitter(angle, shift, dim, mode_1, mode_2, convention="clemens")
                unitary = left @ unitary
                left_list.append(left.conj().T)
    diag = unitary

    right_list.reverse()  # save as [R_n, ..., R_1]
    # left_list = [L_1.inv, ... L_n.inv]

    return left_list, diag, right_list


def _solve_angles(
    unitary: NDArray, mode_1: int, mode_2: int, row: int, col: int, is_odd: bool
) -> NDArray:
    """Solve for the angles that make a certain element of U zero.

    Args:
        unitary (NDArray): the unitary matrix.
        mode_1 (int): the first mode to which the beamsplitter is applied.
        mode_2 (int): the second mode to which the beamsplitter is applied.
        row (int): row to be multiplied.
        col (int): column to be multiplied.
        is_odd (bool): whether the product corresponds to an odd or even step
            in the unitary decomposition.

    Returns:
        NDArray: the real and imaginary parts of the dot product
    """

    def null_U_entry(angles):
        return _U_times_BS_entry(angles, unitary, mode_1, mode_2, row, col, is_odd)

    return fsolve(null_U_entry, np.ones(2))  # type: ignore


def _U_times_BS_entry(
    angles: tuple[float, float],
    U: NDArray,
    mode_1: int,
    mode_2: int,
    row: int,
    col: int,
    is_odd: bool,
) -> NDArray:
    """If is_odd is True, multiply a row of the unitary U times a column of a
    beamsplitter's inverse. If is_odd is False, multiply a row of a beamsplitter
    times a column of the unitary U.

    Args:
        angles (float, float): angles and shift of the beamsplitter

    Returns:
        NDArray: the real and imaginary parts of the dot product
    """

    θ = angles[0]
    φ = angles[1]

    # Since T and T.inv() are sparse there is no need on creating them from scratch
    if is_odd:
        # U[row,:] @ T.inv[:, col]
        dot_prod = U[row, mode_1] * np.exp(-1j * φ) * np.cos(θ) - U[row, mode_2] * np.sin(θ)
    else:
        # T[row,:] @ U[:, col]
        dot_prod = np.exp(1j * φ) * np.sin(θ) * U[mode_1, col] + np.cos(θ) * U[mode_2, col]

    return np.array([dot_prod.real, dot_prod.imag])
