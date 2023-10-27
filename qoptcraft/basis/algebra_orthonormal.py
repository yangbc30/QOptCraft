"""Orthonormal basis of the image and the perpendicular subspaces
of the Lie algebra u(M), where M is the dimension of the Hilbert space
of states with n photons and m modes.
"""
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import spmatrix

from .algebra import get_image_algebra_basis, unitary_algebra_basis
from .hilbert_dimension import hilbert_dim
from qoptcraft.math import gram_schmidt, hs_scalar_product, hs_norm


def basis_image_orthonormal(modes: int, photons: int) -> list[spmatrix] | list[NDArray]:
    basis_image = get_image_algebra_basis(modes, photons)
    return gram_schmidt(basis_image)


def basis_complement_image_orthonormal(modes: int, photons: int) -> list[spmatrix] | list[NDArray]:
    basis_image = basis_image_orthonormal(modes, photons)
    dim = hilbert_dim(modes, photons)
    basis_algebra = unitary_algebra_basis(dim)  # length dim * dim

    basis_complement = []

    for i in range(dim * dim):
        for matrix_image in basis_image:
            basis_algebra[i] -= hs_scalar_product(matrix_image, basis_algebra[i]) * matrix_image

        if not np.allclose(basis_algebra[i], np.zeros((dim, dim))):
            basis_complement.append(basis_algebra[i] / hs_norm(basis_algebra[i]))

            for j in range(i + 1, dim * dim):
                basis_algebra[j] -= (
                    hs_scalar_product(basis_complement[-1], basis_algebra[j]) * basis_complement[-1]
                )

    basis_length = len(basis_image) + len(basis_complement)
    assert (
        basis_length == dim * dim
    ), f"Assertion error. Orthonormal basis length is {basis_length} but should be {dim*dim}."

    return basis_complement
