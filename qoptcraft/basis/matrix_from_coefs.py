import numpy as np

from qoptcraft.basis import unitary_algebra_basis, image_algebra_basis, hilbert_dim


def hermitian_matrix_from_coefs(coefs, dim: int):

    hermitian_matrix = np.zeros((dim, dim), dtype=np.complex128)

    for coef, matrix in zip(coefs, unitary_algebra_basis(dim), strict=True):
        hermitian_matrix += coef * matrix
    return hermitian_matrix


def image_matrix_from_coefs(coefs, modes: int, photons: int, orthonormal: bool = False):

    image_basis = image_algebra_basis(modes, photons, orthonormal)
    dim = hilbert_dim(modes, photons)

    tangent_matrix = np.zeros((dim, dim), dtype=np.complex128)

    for coef, matrix in zip(coefs, image_basis, strict=True):
        tangent_matrix += coef * matrix
    return tangent_matrix
