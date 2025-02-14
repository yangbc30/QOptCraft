import pickle
import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import spmatrix, lil_matrix

from .photon import photon_basis, BasisPhoton
from qoptcraft.operators import creation_fock, annihilation_fock
from qoptcraft import config


BasisAlgebra = list[spmatrix]

warnings.filterwarnings(
    "ignore",
    message=(
        "Changing the sparsity structure of a csr_matrix is expensive."
        " lil_matrix is more efficient."
    ),
)

SQRT_2_INV = 1 / np.sqrt(2)


def _saved_image_algebra_basis(modes: int, photons: int) -> BasisAlgebra:
    """Return a basis for the Hilbert space with n photons and m modes.
    If the basis was saved retrieve it, otherwise the function creates
    and saves the basis to a file.

    Args:
        photons (int): number of photons.
        modes (int): number of modes.

    Returns:
        BasisAlgebra: basis of the image algebra.
    """
    folder_path = config.SAVE_DATA_PATH / f"m={modes} n={photons}"
    folder_path.mkdir(parents=True, exist_ok=True)

    basis_image_path = folder_path / "image_algebra.pkl"
    basis_image_path.touch()  # create file if it doesn't exist
    try:
        with basis_image_path.open("rb") as f:
            basis_image = pickle.load(f)
    except EOFError:
        basis_image = image_algebra_basis(modes, photons, cache=False)
        with basis_image_path.open("wb") as f:
            pickle.dump(basis_image, f)
        print(f"Image algebra basis saved in {folder_path}")

    return basis_image


def unitary_algebra_basis(dim: int) -> BasisAlgebra:
    """Basis of the unitary algebra of dim x dim anti-hermitian matrices."""
    basis = []
    for i in range(dim):
        basis.append(sym_matrix(i, i, dim))
        for j in range(i):
            basis.append(sym_matrix(i, j, dim))
            basis.append(antisym_matrix(i, j, dim))
    return basis


def image_algebra_basis(modes: int, photons: int, cache: bool = True) -> BasisAlgebra:
    """Generate the basis for the algebra and image algebra.

    Args:
        modes (int): number of modes.
        photons (int): number of photons.
        cache (bool, optional): if True uses a cached version of the basis to
            avoid computing it again. Defaults to True.

    Returns:
        BasisAlgebra: _description_
    """
    basis = []
    photonic_basis = photon_basis(modes, photons)

    if cache:
        return _saved_image_algebra_basis(modes, photons)

    for i in range(modes):
        basis.append(image_photon_number(i, photonic_basis))
        for j in range(i):
            basis.append(image_sym_matrix(i, j, photonic_basis))
            basis.append(image_antisym_matrix(i, j, photonic_basis))
    return basis


def sym_matrix(mode_1: int, mode_2: int, dim: int) -> NDArray:
    """Create the element of the algebra i/sqrt(2)(|j><k| + |k><j|)."""
    matrix = np.zeros((dim, dim), dtype=np.complex128)
    if mode_1 == mode_2:
        matrix[mode_1, mode_1] = 1j
        return matrix
    matrix[mode_1, mode_2] = SQRT_2_INV * 1j
    matrix[mode_2, mode_1] += SQRT_2_INV * 1j
    return matrix


def antisym_matrix(mode_1: int, mode_2: int, dim: int) -> NDArray:
    """Create the element of the algebra 1/sqrt(2)(|j><k| - |k><j|)."""
    matrix = np.zeros((dim, dim), dtype=np.complex128)
    matrix[mode_1, mode_2] = SQRT_2_INV
    matrix[mode_2, mode_1] = -SQRT_2_INV
    return matrix


def image_photon_number(mode: int, photonic_basis: BasisPhoton) -> spmatrix:
    """Image of the symmetric basis matrix by the lie algebra homomorphism."""
    dim = len(photonic_basis)
    matrix = lil_matrix((dim, dim), dtype=np.complex128)  # * efficient format for loading data

    for i, fock in enumerate(photonic_basis):
        matrix[i, i] = 1j * fock[mode]
    return matrix.tocsr()


def image_sym_matrix(mode_1: int, mode_2: int, photonic_basis: BasisPhoton) -> spmatrix:
    """Image of the symmetric basis matrix by the lie algebra homomorphism."""
    dim = len(photonic_basis)
    matrix = lil_matrix((dim, dim), dtype=np.complex128)  # * efficient format for loading data

    if mode_1 == mode_2:
        raise ValueError(
            "Modes should be different. For mode_1 == mode_2 use image_photon_number()."
        )

    for col, fock_ in enumerate(photonic_basis):
        if fock_[mode_1] != 0:
            fock, coef = annihilation_fock(mode_1, fock_)
            fock, coef_ = creation_fock(mode_2, fock)
            matrix[photonic_basis.index(fock), col] = SQRT_2_INV * 1j * coef * coef_

        if fock_[mode_2] != 0:
            fock, coef = annihilation_fock(mode_2, fock_)
            fock, coef_ = creation_fock(mode_1, fock)
            matrix[photonic_basis.index(fock), col] += SQRT_2_INV * 1j * coef * coef_

    return matrix.tocsr()


def image_antisym_matrix(mode_1: int, mode_2: int, photonic_basis: BasisPhoton) -> spmatrix:
    """Image of the antisymmetric basis matrix by the lie algebra homomorphism."""
    if mode_1 == mode_2:
        raise ValueError("Antisymmetric matrix cannot have equal modes.")
    dim = len(photonic_basis)
    matrix = lil_matrix((dim, dim), dtype=np.complex128)

    for col, fock_ in enumerate(photonic_basis):
        if fock_[mode_1] != 0:
            fock, coef = annihilation_fock(mode_1, fock_)
            fock, coef_ = creation_fock(mode_2, fock)
            matrix[photonic_basis.index(fock), col] = -SQRT_2_INV * coef * coef_

        if fock_[mode_2] != 0:
            fock, coef = annihilation_fock(mode_2, fock_)
            fock, coef_ = creation_fock(mode_1, fock)
            matrix[photonic_basis.index(fock), col] += SQRT_2_INV * coef * coef_

    return matrix.tocsr()
