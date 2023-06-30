<<<<<<< HEAD:QOptCraft/optical_elements/reck_decomp.py
"""Decompose a unitary into beamsplitters following Reck et al.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import fsolve

from .optical_elements import beam_splitter


def reck_decomposition(unitary: NDArray) -> tuple[NDArray, list[NDArray]]:
    """Given a unitary matrix calculates the Reck et al. decompositon
    into beam splitters and phase shifters:

    D = U · BS_1... BS_n  =>  U = D · BS_n.inv ... BS_1.inv

    Args:
        unitary (NDArray): unitary matrix to decompose.

    Returns:
        list[NDArray]: list of matrices that decompose U in the order of the decomposition

    References:
        The algorithm can be found in [1]_.

        .. [1] Reck et al., "Experimental realization of any discrete unitary operator"
            Phys. Rev. Lett. 73, 58, 1994. https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.73.58
    """

    assert unitary.shape[0] == unitary.shape[1], "The matrix is not square"
    dim = unitary.shape[0]

    bs_list = []

    for row in range(dim - 1, 0, -1):
        for col in range(row - 1, -1, -1):
            # Eliminate (row, col) element of U with a beamsplitter T_(row, col)

            # We calculate the beamsplitter angles phi y theta
            angle, shift = _solve_angles(unitary, row, col)
            right = beam_splitter(angle, shift, dim, mode_1=row, mode_2=col, convention="reck")
            unitary = unitary @ right
            bs_list.append(right.conj().T)
    diag = unitary
    bs_list.reverse()
    return diag, bs_list


def _solve_angles(U: NDArray, row: int, col: int) -> NDArray:
    """Solve for the angles that make a certain element of U zero.

    Args:
        U (NDArray): the unitary matrix.
        mode_1 (int): the first mode to which the beamsplitter is applied.
        mode_2 (int): the second mode to which the beamsplitter is applied.
        row (int): row to be multiplied.
        col (int): column to be multiplied.

    Returns:
        NDArray: the real and imaginary parts of the dot product
    """

    def null_U_entry(angles):
        return _U_times_BS_entry(angles, U, row, col)

    return fsolve(null_U_entry, np.ones(2))  # type: ignore


def _U_times_BS_entry(
    angles: tuple[float, float],
    U: NDArray,
    row: int,
    col: int,
) -> NDArray:
    """Multiply a row of the unitary U times a column of a beamsplitter's inverse.

    Args:
        angles (float, float): angles and shift of the beamsplitter

    Returns:
        NDArray: the real and imaginary parts of the dot product
    """
    θ = angles[0]
    φ = angles[1]

    # U[row, row] * BS_(row, col)[row,col] + U[row, col] * BS_(row, col)[col,col]
    dot_prod = U[row, row] * np.exp(1j * φ) * np.cos(θ) - U[row, col] * np.sin(θ)
    return np.array([dot_prod.real, dot_prod.imag])
=======
"""Decompose a unitary into beamsplitters following Reck et al.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import fsolve

from .optical_elements import beam_splitter_reck


def reck_decomposition(unitary: NDArray) -> list[NDArray]:
    """Given a unitary matrix calculates the Reck et al. decompositon
    into beam splitters and phase shifters:

    D = U · BS_1... BS_n  =>  U = D · BS_n.inv ... BS_1.inv

    Args:
        unitary (NDArray): unitary matrix to decompose.

    Returns:
        list[NDArray]: list of matrices that decompose U in the order of the decomposition

    References:
        The algorithm can be found in [1]_.

        .. [1] Reck et al., "Experimental realization of any discrete unitary operator"
            Phys. Rev. Lett. 73, 58, 1994. https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.73.58
    """

    assert unitary.shape[0] == unitary.shape[1], "The matrix is not square"
    dim = unitary.shape[0]

    bs_list = []

    for row in range(dim - 1, 0, -1):
        for col in range(row - 1, -1, -1):
            # Eliminate (row, col) element of U with a beamsplitter T_(row, col)

            # We calculate the beamsplitter angles phi y theta
            angle, shift = _solve_angles(unitary, row, col)
            R = beam_splitter_reck(angle, shift, dim, mode_1=row, mode_2=col)
            unitary = unitary @ R
            bs_list.append(R.conj().T)
    D = unitary
    bs_list.reverse()
    return [D] + bs_list


def _solve_angles(U: NDArray, row: int, col: int) -> NDArray:
    """Solve for the angles that make a certain element of U zero.

    Args:
        U (NDArray): the unitary matrix.
        mode_1 (int): the first mode to which the beamsplitter is applied.
        mode_2 (int): the second mode to which the beamsplitter is applied.
        row (int): row to be multiplied.
        col (int): column to be multiplied.

    Returns:
        NDArray: the real and imaginary parts of the dot product
    """

    def null_U_entry(angles):
        return _U_times_BS_entry(angles, U, row, col)

    return fsolve(null_U_entry, np.ones(2))  # type: ignore


def _U_times_BS_entry(
    angles: tuple[float, float],
    U: NDArray,
    row: int,
    col: int,
) -> NDArray:
    """Multiply a row of the unitary U times a column of a beamsplitter's inverse.

    Args:
        angles (float, float): angles and shift of the beamsplitter

    Returns:
        NDArray: the real and imaginary parts of the dot product
    """
    θ = angles[0]
    φ = angles[1]

    # U[row, row] * BS_(row, col)[row,col] + U[row, col] * BS_(row, col)[col,col]
    dot_prod = U[row, row] * np.exp(1j * φ) * np.cos(θ) - U[row, col] * np.sin(θ)
    return np.array([dot_prod.real, dot_prod.imag])
>>>>>>> d41662bbef5ea1a918e17800a5bf945002d43f0c:qoptcraft/optical_elements/reck_decomp.py
