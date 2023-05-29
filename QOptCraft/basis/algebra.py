from pathlib import Path
import pickle
from numbers import Number
import warnings

import numpy as np
from numpy.typing import NDArray
import scipy as sp
from scipy.sparse import spmatrix, csr_matrix, lil_matrix

from QOptCraft.basis import get_photon_basis


warnings.filterwarnings(
    "ignore",
    message=(
        "Changing the sparsity structure of a csr_matrix is expensive."
        " lil_matrix is more efficient."
    ),
)


def get_algebra_basis(modes: int, photons: int):
    """Return a basis for the Hilbert space with n photons and m modes.
    If the basis was saved retrieve it, otherwise the function creates
    and saves the basis to a file.

    Args:
        photons (int): number of photons.
        modes (int): number of modes.

    Returns:
        list[list[int]]: basis of the Hilbert space.
    """
    folder = Path(f"save_basis/m={modes} n={photons}")
    folder.mkdir(parents=True, exist_ok=True)

    basis_path = folder / "algebra.pkl"
    basis_image_path = folder / "image_algebra.pkl"
    basis_path.touch()  # create file if it doesn't exist
    basis_image_path.touch()
    try:
        with basis_path.open("rb") as f:
            basis = pickle.load(f)
        with basis_image_path.open("rb") as f:
            basis_image = pickle.load(f)

    except EOFError:
        basis, basis_image = _algebra_basis(modes, photons)
        with basis_path.open("wb") as f:
            pickle.dump(basis, f)
        with basis_image_path.open("wb") as f:
            pickle.dump(basis_image, f)
        print(f"Basis saved in {folder}")

    return basis, basis_image


def _algebra_basis(modes: int, photons: int):
    """Generate the basis for the algebra and image algebra."""
    basis = []
    basis_image = []
    dim_image = int(sp.special.comb(modes + photons - 1, photons))
    photon_basis = get_photon_basis(modes, photons)

    for mode_1 in range(modes):
        for mode_2 in range(mode_1 + 1):
            matrix = sym_matrix(mode_1, mode_2, modes)
            basis.append(matrix)
            basis_image.append(image_sym_matrix(mode_1, mode_2, dim_image, photon_basis))

    # Divide into two loops to separate symmetric from antisymmetric matrices
    for mode_1 in range(modes):
        for mode_2 in range(mode_1):
            matrix = antisym_matrix(mode_1, mode_2, modes)
            basis.append(matrix)
            basis_image.append(image_antisym_matrix(mode_1, mode_2, dim_image, photon_basis))

    return basis, basis_image


def sym_matrix(mode_1: int, mode_2: int, dim: int) -> NDArray:
    """Create the element of the algebra i/2(|j><k| + |k><j|)."""
    matrix = csr_matrix((dim, dim), dtype="complex64")
    matrix[mode_1, mode_2] = 0.5j
    matrix[mode_2, mode_1] += 0.5j
    return matrix


def antisym_matrix(mode_1: int, mode_2: int, dim: int) -> NDArray:
    """Create the element of the algebra 1/2(|j><k| - |k><j|)."""
    matrix = csr_matrix((dim, dim), dtype="complex64")
    matrix[mode_1, mode_2] = 0.5
    matrix[mode_2, mode_1] = -0.5

    return matrix


def image_sym_matrix(mode_1: int, mode_2: int, dim: int, basis: list[list[int]]) -> spmatrix:
    """Image of the symmetric basis matrix by the lie algebra homomorphism."""
    matrix = lil_matrix((dim, dim), dtype="complex64")  # * efficient format for loading data

    for col, fock_ in enumerate(basis):
        if fock_[mode_1] != 0:
            fock = fock_.copy()  # * We don't want to modify the basis!!
            coef = annihilation(mode_1, fock)
            coef *= creation(mode_2, fock)
            matrix[basis.index(fock), col] = 0.5j * coef

        if fock_[mode_2] != 0:
            fock = fock_.copy()
            coef = annihilation(mode_2, fock)
            coef *= creation(mode_1, fock)
            matrix[basis.index(fock), col] += 0.5j * coef

    return matrix.tocsr()


def image_antisym_matrix(
    mode_1: int, mode_2: int, dim: int, photon_basis: list[list[int]]
) -> spmatrix:
    """Image of the antisymmetric basis matrix by the lie algebra homomorphism."""
    matrix = lil_matrix((dim, dim), dtype="complex64")

    for col, fock_ in enumerate(photon_basis):
        if fock_[mode_1] != 0:
            fock = fock_.copy()
            coef = annihilation(mode_1, fock)
            coef *= creation(mode_2, fock)
            matrix[photon_basis.index(fock), col] = 0.5 * coef

        if fock_[mode_2] != 0:
            fock = fock_.copy()
            coef = annihilation(mode_2, fock)
            coef *= creation(mode_1, fock)
            matrix[photon_basis.index(fock), col] += -0.5 * coef

    return matrix.tocsr()


def creation(mode: int, state: list[int]) -> Number:
    """Creation operator acting on a specific mode. Modifies state in-place.

    Args:
        mode (int): a quantum mode.
        state (list[int]): fock basis state.
        coef (Number): coefficient of the state.

    Returns:
        tuple[list[int], Number]: created state and its coefficient.
    """
    photons = state[mode]
    coef = np.sqrt(photons + 1)
    state[mode] = photons + 1  # * modified in-place
    return coef


def annihilation(mode: int, state: list[int]) -> Number:
    """Annihilation operator acting on a specific mode.

    Args:
        mode (int): a quantum mode.
        state (list[int]): fock basis state.
        coef (Number): coefficient of the state.

    Returns:
        tuple[list[int], Number]: annihilated state and its coefficient.
    """
    photons = state[mode]
    coef = np.sqrt(photons)
    state[mode] = photons - 1  # * modified in-place
    return coef
