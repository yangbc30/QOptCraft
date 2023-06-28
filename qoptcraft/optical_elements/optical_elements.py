"""Beamsplitters as defined in Clemens et al. and in Reck et al., respectively.
"""

import numpy as np
from numpy.typing import NDArray


def beam_splitter(angle: float, shift: float, dim: int, mode_1: int, mode_2: int) -> NDArray:
    """Create the beam splitter matrix with reflectivity cos(θ) and phase shift φ
    acting on mode_1 and mode_2.

    Parameters:
        angle (float): reflectivity angle.
        shift (float): phase shift angle.
        dim (int): dimension of the circuit matrix.
        mode_1 (int): the first mode to which the beam splitter is applied. Starts at 0.
        mode_2 (int): the second mode to which the beam splitter is applied. Starts at 0.

    Returns:
        NDArray: the matrix of the beam splitter

    References:
        The matrix can be found in [1].

        [1] Clements et al., "An Optimal Design for Universal Multiport
            Interferometers" arXiv, 2007. https://arxiv.org/pdf/1603.08788.pdf
    """
    T = np.eye(dim, dtype=np.complex64)
    T[mode_1, mode_1] = np.exp(1j * shift) * np.cos(angle)
    T[mode_1, mode_2] = -np.sin(angle)
    T[mode_2, mode_1] = np.exp(1j * shift) * np.sin(angle)
    T[mode_2, mode_2] = np.cos(angle)

    return T


def beam_splitter_reck(angle: float, shift: float, dim: int, mode_1: int, mode_2: int) -> NDArray:
    """Create the beam splitter matrix with reflectivity cos(θ) and phase shift φ
    acting on mode_1 and mode_2.

    Parameters:
        angle (float): reflectivity angle.
        shift (float): phase shift angle.
        dim (int): dimension of the circuit matrix.
        mode_1 (int): the first mode to which the beam splitter is applied. Starts at 0.
        mode_2 (int): the second mode to which the beam splitter is applied. Starts at 0.

    Returns:
        NDArray: the matrix of the beam splitter

    References:
        The matrix can be found in [1].

        [1] Clements et al., "An Optimal Design for Universal Multiport
            Interferometers" arXiv, 2007. https://arxiv.org/pdf/1603.08788.pdf
    """
    T = np.eye(dim, dtype=np.complex64)
    T[mode_1, mode_1] = np.exp(1j * shift) * np.sin(angle)
    T[mode_1, mode_2] = np.exp(1j * shift) * np.cos(angle)
    T[mode_2, mode_1] = np.cos(angle)
    T[mode_2, mode_2] = -np.sin(angle)

    return T


def phase_shifter(shift: float, dim: int, mode: int) -> NDArray:
    """Create a phase shifter of a certain angle acting on a mode.

    Parameters:
        shift (float): phase shift angle.
        dim (int): dimension of the circuit matrix.
        mode (int): the mode to which the beam splitter is applied. Starts at 0.

    Returns:
        ndarray: the matrix of the phase shifter.

    References:
        The matrix can be found in [1].

        [1] Clements et al., "An Optimal Design for Universal Multiport
            Interferometers" arXiv, 2007. https://arxiv.org/pdf/1603.08788.pdf
    """
    matrix = np.eye(dim, dtype=np.complex64)
    matrix[mode, mode] = np.exp(1j * shift)
    return matrix
