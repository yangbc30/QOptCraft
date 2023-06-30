<<<<<<< HEAD:QOptCraft/optical_elements/optical_elements.py
<<<<<<< HEAD:QOptCraft/optical_elements/optical_elements.py
"""Beamsplitters as defined in Clemens et al. and in Reck et al., respectively.
"""
import numpy as np
from numpy.typing import NDArray


def phase_shifter(shift: float, dim: int, mode: int) -> NDArray:
    """Create a phase shifter of a certain angle acting on a mode.

    Args:
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


def beam_splitter(
    angle: float, shift: float, dim: int, mode_1: int, mode_2: int, convention: str = "clemens"
) -> NDArray:
    """Create the beam splitter matrix with reflectivity cos(θ) and phase shift φ
    acting on mode_1 and mode_2.

    Args:
        angle (float): reflectivity angle.
        shift (float): phase shift angle.
        dim (int): dimension of the circuit matrix.
        mode_1 (int): the first mode to which the beam splitter is applied. Starts at 0.
        mode_2 (int): the second mode to which the beam splitter is applied. Starts at 0.
        convention (str): Beamsplitter definition according to different articles.
            Defaults to 'clemens'.

    Returns:
        NDArray: the matrix of the beam splitter
    """
    if convention == "clemens":
        return _beam_splitter_clemens(angle, shift, dim, mode_1, mode_2)
    if convention == "reck":
        return _beam_splitter_reck(angle, shift, dim, mode_1, mode_2)
    if convention == "chernikov":
        return _beam_splitter_chernikov(angle, shift, dim, mode_1, mode_2)
    raise ValueError("Options for convention parameter are 'clemens', 'reck' or 'chernikov'.")


def _beam_splitter_clemens(
    angle: float, shift: float, dim: int, mode_1: int, mode_2: int
) -> NDArray:
    """Create the beam splitter matrix with reflectivity cos(θ) and phase shift φ
    acting on mode_1 and mode_2. Follows Clemens et al. [1].

    References:
        [1] Clements et al., "An Optimal Design for Universal Multiport
            Interferometers" arXiv, 2007. https://arxiv.org/pdf/1603.08788.pdf
    """
    T = np.eye(dim, dtype=np.complex64)
    T[mode_1, mode_1] = np.exp(1j * shift) * np.cos(angle)
    T[mode_1, mode_2] = -np.sin(angle)
    T[mode_2, mode_1] = np.exp(1j * shift) * np.sin(angle)
    T[mode_2, mode_2] = np.cos(angle)
    return T


def _beam_splitter_reck(angle: float, shift: float, dim: int, mode_1: int, mode_2: int) -> NDArray:
    """Create the beam splitter matrix with reflectivity cos(θ) and phase shift φ
    acting on mode_1 and mode_2. Follows Reck et al. [1].

    References:
        [1] Reck et al., "Experimental realization of any discrete unitary operator" PRL, 1994.
            https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.73.58
    """
    T = np.eye(dim, dtype=np.complex64)
    T[mode_1, mode_1] = np.exp(1j * shift) * np.sin(angle)
    T[mode_1, mode_2] = np.exp(1j * shift) * np.cos(angle)
    T[mode_2, mode_1] = np.cos(angle)
    T[mode_2, mode_2] = -np.sin(angle)
    return T


def _beam_splitter_chernikov(
    angle: float, shift: float, dim: int, mode_1: int, mode_2: int
) -> NDArray:
    """Create the beam splitter matrix with reflectivity cos(θ) and phase shift φ
    acting on mode_1 and mode_2. Follows Chernikov et al. [1].

    References:
        [1] Chernikov et al., "Heralded gate search with genetic algorithms for
            quantum computation" arXiv, 2023. https://arxiv.org/abs/2303.05855
    """
    T = np.eye(dim, dtype=np.complex64)
    T[mode_1, mode_1] = np.cos(angle)
    T[mode_1, mode_2] = -np.exp(1j * shift) * np.sin(angle)
    T[mode_2, mode_1] = np.exp(-1j * shift) * np.sin(angle)
    T[mode_2, mode_2] = np.cos(angle)
    return T
=======
=======
>>>>>>> d41662bbef5ea1a918e17800a5bf945002d43f0c:qoptcraft/optical_elements/optical_elements.py
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
<<<<<<< HEAD:QOptCraft/optical_elements/optical_elements.py
>>>>>>> d41662bbef5ea1a918e17800a5bf945002d43f0c:qoptcraft/optical_elements/optical_elements.py
=======
>>>>>>> d41662bbef5ea1a918e17800a5bf945002d43f0c:qoptcraft/optical_elements/optical_elements.py
