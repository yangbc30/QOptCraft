"""Beamsplitters as defined in Clemens et al. and in Reck et al., respectively.
"""

import numpy as np


def beamsplitter(angle: float, shift: float, dim: int, mode_1: int, mode_2: int) -> np.ndarray:
    """
    Create the beamsplitter matrix with reflectivity cos(θ) and phase shift φ
    acting on mode_1 and mode_2.

    Parameters
    ----------
    dim_matrix: int
        dimension of the circuit matrix
    angle: float
        reflectivity angle
    φ: float
        phase shift angle
    mode_1: int
        the first mode to which the beamsplitter is applied
    mode_2: int
        the second mode to which the beamsplitter is applied

    Returns
    -------
    ndarray
        the matrix of the beamsplitter

    References
    ----------
    The matrix can be found in [1]_.

    .. [1] Clements et al., "An Optimal Design for Universal Multiport
        Interferometers" arXiv, 2007. https://arxiv.org/pdf/1603.08788.pdf
    """

    T = np.eye(dim, dtype=np.complex64)
    T[mode_1, mode_1] = np.exp(1j * shift) * np.cos(angle)
    T[mode_1, mode_2] = -np.sin(angle)
    T[mode_2, mode_1] = np.exp(1j * shift) * np.sin(angle)
    T[mode_2, mode_2] = np.cos(angle)

    return T


def beamsplitter_reck(angle: float, shift: float, dim: int, mode_1: int, mode_2: int) -> np.ndarray:
    """
    Create the beamsplitter matrix with reflectivity cos(θ) and phase shift φ
    acting on mode_1 and mode_2.

    Parameters
    ----------
    angle: float
        reflectivity angle
    φ: float
        phase shift angle
    dim_matrix: int
        dimension of the circuit matrix
    mode_1: int
        the first mode to which the beamsplitter is applied
    mode_2: int
        the second mode to which the beamsplitter is applied

    Returns
    -------
    ndarray
        the matrix of the beamsplitter

    References
    ----------
    The matrix can be found in [1]_.

    .. [1] Clements et al., "An Optimal Design for Universal Multiport
        Interferometers" arXiv, 2007. https://arxiv.org/pdf/1603.08788.pdf
    """

    T = np.eye(dim, dtype=np.complex64)
    T[mode_1, mode_1] = np.exp(1j * shift) * np.sin(angle)
    T[mode_1, mode_2] = np.exp(1j * shift) * np.cos(angle)
    T[mode_2, mode_1] = np.cos(angle)
    T[mode_2, mode_2] = -np.sin(angle)

    return T
