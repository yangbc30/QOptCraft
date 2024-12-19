import pickle
import warnings

import numpy as np
from scipy.sparse import spmatrix, lil_matrix

from .photon import get_photon_basis, BasisPhoton
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


def get_algebra_basis(modes: int, photons: int) -> tuple[BasisAlgebra, BasisAlgebra]:
    """Return a basis for the Hilbert space with n photons and m modes.
    If the basis was saved retrieve it, otherwise the function creates
    and saves the basis to a file.

    Args:
        photons (int): number of photons.
        modes (int): number of modes.

    Returns:
        BasisAlgebra, BasisAlgebra: basis of the algebra and the image algebra.
    """
    warnings.warn(
        "get_algebra_basis is deprecated. Use the new function get_image_algebra_basis.",
        DeprecationWarning,
        stacklevel=2,
    )

    folder_path = config.SAVE_DATA_PATH / f"m={modes} n={photons}"
    folder_path.mkdir(parents=True, exist_ok=True)

    basis_path = folder_path / "algebra.pkl"
    basis_image_path = folder_path / "image_algebra.pkl"
    basis_path.touch()  # create file if it doesn't exist
    basis_image_path.touch()
    try:
        with basis_path.open("rb") as f:
            basis = pickle.load(f)
        with basis_image_path.open("rb") as f:
            basis_image = pickle.load(f)

    except EOFError:
        basis, basis_image = algebra_basis(modes, photons)
        with basis_path.open("wb") as f:
            pickle.dump(basis, f)
        with basis_image_path.open("wb") as f:
            pickle.dump(basis_image, f)
        print(f"Algebra basis saved in {folder_path}")

    return basis, basis_image


def algebra_basis(modes: int, photons: int) -> tuple[BasisAlgebra, BasisAlgebra]:
    """Generate the basis for the algebra and image algebra."""
    basis = []
    basis_image = []
    photon_basis = get_photon_basis(modes, photons)

    for mode_1 in range(modes):
        for mode_2 in range(mode_1 + 1):
            matrix = sym_matrix(mode_1, mode_2, modes)
            basis.append(matrix)
            basis_image.append(image_sym_matrix(mode_1, mode_2, photon_basis))

    # Divide into two loops to separate symmetric from antisymmetric matrices
    for mode_1 in range(modes):
        for mode_2 in range(mode_1):
            matrix = antisym_matrix(mode_1, mode_2, modes)
            basis.append(matrix)
            basis_image.append(image_antisym_matrix(mode_1, mode_2, photon_basis))

    return basis, basis_image


def get_image_algebra_basis(modes: int, photons: int) -> BasisAlgebra:
    """Return a basis for the Hilbert space with n photons and m modes.
    If the basis was saved retrieve it, otherwise the function creates
    and saves the basis to a file.

    Args:
        photons (int): number of photons.
        modes (int): number of modes.

    Returns:
        BasisAlgebra, BasisAlgebra: basis of the algebra and the image algebra.
    """
    folder_path = config.SAVE_DATA_PATH / f"m={modes} n={photons}"
    folder_path.mkdir(parents=True, exist_ok=True)

    basis_image_path = folder_path / "image_algebra.pkl"
    basis_image_path.touch()  # create file if it doesn't exist
    try:
        with basis_image_path.open("rb") as f:
            basis_image = pickle.load(f)
    except EOFError:
        basis_image = image_algebra_basis(modes, photons)
        with basis_image_path.open("wb") as f:
            pickle.dump(basis_image, f)
        print(f"Image algebra basis saved in {folder_path}")

    return basis_image


def unitary_algebra_basis(dim: int) -> BasisAlgebra:
    basis = []
    for i in range(dim):
        basis.append(sym_matrix(i, i, dim))
        for j in range(i):
            basis.append(sym_matrix(i, j, dim))
            basis.append(antisym_matrix(i, j, dim))
    return basis


def image_algebra_basis(modes: int, photons: int) -> tuple[BasisAlgebra, BasisAlgebra]:
    """Generate the basis for the algebra and image algebra."""
    basis = []
    photon_basis = get_photon_basis(modes, photons)

    for i in range(modes):
        basis.append(image_photon_number(i, photon_basis))
        for j in range(i):
            basis.append(image_sym_matrix(i, j, photon_basis))
            basis.append(image_antisym_matrix(i, j, photon_basis))
    return basis


def sym_matrix(mode_1: int, mode_2: int, dim: int) -> spmatrix:
    """Create the element of the algebra i/2(|j><k| + |k><j|)."""
    matrix = np.zeros((dim, dim), dtype=np.complex128)
    if mode_1 == mode_2:
        matrix[mode_1, mode_1] = 1j
        return matrix
    matrix[mode_1, mode_2] = SQRT_2_INV * 1j
    matrix[mode_2, mode_1] += SQRT_2_INV * 1j
    return matrix


def antisym_matrix(mode_1: int, mode_2: int, dim: int) -> spmatrix:
    """Create the element of the algebra 1/2(|j><k| - |k><j|)."""
    matrix = np.zeros((dim, dim), dtype=np.complex128)
    matrix[mode_1, mode_2] = SQRT_2_INV
    matrix[mode_2, mode_1] = -SQRT_2_INV
    return matrix


def image_photon_number(mode: int, photon_basis: BasisPhoton) -> spmatrix:
    """Image of the symmetric basis matrix by the lie algebra homomorphism."""
    dim = len(photon_basis)
    matrix = lil_matrix((dim, dim), dtype=np.complex128)  # * efficient format for loading data

    for i, fock in enumerate(photon_basis):
        matrix[i, i] = 1j * fock[mode]
    return matrix.tocsr()


def image_sym_matrix(mode_1: int, mode_2: int, photon_basis: BasisPhoton) -> spmatrix:
    """Image of the symmetric basis matrix by the lie algebra homomorphism."""
    dim = len(photon_basis)
    matrix = lil_matrix((dim, dim), dtype=np.complex128)  # * efficient format for loading data

    if mode_1 == mode_2:
        for col, fock_ in enumerate(photon_basis):
            if fock_[mode_1] != 0:
                fock, coef = annihilation_fock(mode_1, fock_)
                fock, coef_ = creation_fock(mode_2, fock)
                matrix[photon_basis.index(fock), col] = 1j * coef * coef_
        return matrix.tocsr()

    for col, fock_ in enumerate(photon_basis):
        if fock_[mode_1] != 0:
            fock, coef = annihilation_fock(mode_1, fock_)
            fock, coef_ = creation_fock(mode_2, fock)
            matrix[photon_basis.index(fock), col] = SQRT_2_INV * 1j * coef * coef_

        if fock_[mode_2] != 0:
            fock, coef = annihilation_fock(mode_2, fock_)
            fock, coef_ = creation_fock(mode_1, fock)
            matrix[photon_basis.index(fock), col] += SQRT_2_INV * 1j * coef * coef_

    return matrix.tocsr()


def image_antisym_matrix(mode_1: int, mode_2: int, photon_basis: BasisPhoton) -> spmatrix:
    """Image of the antisymmetric basis matrix by the lie algebra homomorphism."""
    if mode_1 == mode_2:
        raise ValueError("Antisymmetric matrix cannot have equal modes.")
    dim = len(photon_basis)
    matrix = lil_matrix((dim, dim), dtype=np.complex128)

    for col, fock_ in enumerate(photon_basis):
        if fock_[mode_1] != 0:
            fock, coef = annihilation_fock(mode_1, fock_)
            fock, coef_ = creation_fock(mode_2, fock)
            matrix[photon_basis.index(fock), col] = -SQRT_2_INV * coef * coef_

        if fock_[mode_2] != 0:
            fock, coef = annihilation_fock(mode_2, fock_)
            fock, coef_ = creation_fock(mode_1, fock)
            matrix[photon_basis.index(fock), col] += SQRT_2_INV * coef * coef_

    return matrix.tocsr()
