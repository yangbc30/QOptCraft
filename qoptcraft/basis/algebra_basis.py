import math
import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import lil_matrix, spmatrix

from qoptcraft.math import gram_schmidt, hs_inner_product, hs_norm
from qoptcraft.operators import annihilation_fock, creation_fock
from qoptcraft.utils import saved_basis

from .hilbert_dimension import hilbert_dim
from .photon import BasisPhoton, photon_basis


BasisAlgebra = list[spmatrix]

warnings.filterwarnings(
    "ignore",
    message=(
        "Changing the sparsity structure of a csr_matrix is expensive."
        " lil_matrix is more efficient."
    ),
)

SQRT_2_INV = 1 / np.sqrt(2)


def unitary_algebra_basis(dim: int) -> BasisAlgebra:
    """Basis of the unitary algebra of dim x dim anti-hermitian matrices."""
    basis = []
    for i in range(dim):
        basis.append(sym_matrix(i, i, dim))
        for j in range(i):
            basis.append(sym_matrix(i, j, dim))
            basis.append(antisym_matrix(i, j, dim))
    return basis


@saved_basis(file_name="image_algebra")
def image_algebra_basis(
    modes: int, photons: int, orthonormal: bool = False, cache: bool = True
) -> BasisAlgebra:
    """Generate the basis for the algebra and image algebra.

    Args:
        modes (int): number of modes.
        photons (int): number of photons.
        orthonormal (bool): if True, the basis is orthonormalized.
        cache (bool, optional): if True uses a cached version of the basis to
            avoid computing it again. Defaults to True.

    Returns:
        BasisAlgebra: _description_
    """
    _ = cache  # only used by the decorator @saved_basis

    basis = []
    photonic_basis = photon_basis(modes, photons)

    for i in range(modes):
        basis.append(image_photon_number(i, photonic_basis))
        for j in range(i):
            basis.append(image_sym_matrix(i, j, photonic_basis))
            basis.append(image_antisym_matrix(i, j, photonic_basis))
    if orthonormal:
        return gram_schmidt(basis)
    return basis


def sym_matrix(mode_1: int, mode_2: int, dim: int) -> NDArray:
    """Create the element of the algebra i/sqrt(2)(|j><k| + |k><j|)."""
    matrix = np.zeros((dim, dim), dtype=np.complex128)
    if mode_1 == mode_2:
        matrix[mode_1, mode_1] = 1
        return matrix
    matrix[mode_1, mode_2] = SQRT_2_INV
    matrix[mode_2, mode_1] = SQRT_2_INV
    return matrix


def antisym_matrix(mode_1: int, mode_2: int, dim: int) -> NDArray:
    """Create the element of the algebra 1/sqrt(2)(|j><k| - |k><j|)."""
    if mode_1 == mode_2:
        raise ValueError(
            f"Modes cannot be equal for antisymmetric matrix: {mode_1 = }, {mode_2 = }."
        )
    matrix = np.zeros((dim, dim), dtype=np.complex128)
    matrix[mode_1, mode_2] = SQRT_2_INV * 1j
    matrix[mode_2, mode_1] = -SQRT_2_INV * 1j
    return matrix


def image_photon_number(mode: int, photonic_basis: BasisPhoton) -> spmatrix:
    """Image of the symmetric basis matrix by the lie algebra homomorphism."""
    dim = len(photonic_basis)
    matrix = lil_matrix((dim, dim), dtype=np.complex128)  # * efficient format for loading data

    for i, fock in enumerate(photonic_basis):
        matrix[i, i] = fock[mode]
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
            matrix[photonic_basis.index(fock), col] = SQRT_2_INV * coef * coef_

        if fock_[mode_2] != 0:
            fock, coef = annihilation_fock(mode_2, fock_)
            fock, coef_ = creation_fock(mode_1, fock)
            matrix[photonic_basis.index(fock), col] += SQRT_2_INV * coef * coef_

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
            matrix[photonic_basis.index(fock), col] = -SQRT_2_INV * 1j * coef * coef_

        if fock_[mode_2] != 0:
            fock, coef = annihilation_fock(mode_2, fock_)
            fock, coef_ = creation_fock(mode_1, fock)
            matrix[photonic_basis.index(fock), col] += SQRT_2_INV * 1j * coef * coef_

    return matrix.tocsr()


def complement_algebra_basis_orthonormal(
    modes: int, photons: int, cache: bool = True
) -> list[spmatrix] | list[NDArray]:
    basis_image = image_algebra_basis(modes, photons, orthonormal=True, cache=cache)
    dim = hilbert_dim(modes, photons)
    basis_algebra = unitary_algebra_basis(dim)  # length dim * dim

    basis_complement = []

    for i in range(dim * dim):
        for matrix_image in basis_image:
            basis_algebra[i] -= hs_inner_product(matrix_image, basis_algebra[i]) * matrix_image

        if not np.allclose(basis_algebra[i], np.zeros((dim, dim))):
            basis_complement.append(basis_algebra[i] / hs_norm(basis_algebra[i]))

            for j in range(i + 1, dim * dim):
                basis_algebra[j] -= (
                    hs_inner_product(basis_complement[-1], basis_algebra[j]) * basis_complement[-1]
                )

    basis_length = len(basis_image) + len(basis_complement)
    assert basis_length == dim * dim, (
        f"Assertion error. Orthonormal basis length is {basis_length} but should be {dim * dim}."
    )

    return basis_complement


def hermitian_matrix_from_coefs(coefs):
    dim = math.isqrt(len(coefs))

    hermitian_matrix = np.zeros((dim, dim), dtype=np.complex128)

    for coef, matrix in zip(coefs, unitary_algebra_basis(dim), strict=True):
        hermitian_matrix += coef * matrix
    return hermitian_matrix
