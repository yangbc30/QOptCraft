import math
import numpy as np

from qoptcraft.basis import unitary_algebra_basis, image_algebra_basis, hilbert_dim
from .algebra_basis import sym_matrix, antisym_matrix
from qoptcraft.math import hs_inner_product


def algebra_hm(matrix, photons: int, orthonormal: bool = False):
    """Takes a hermitian matrix and outputs its image under the differential
    of the photonic homomorphism.

    Args:
        matrix (_type_): _description_
        orthonormal (bool, optional): Orthonormal image basis. Defaults to False.

    Returns:
        _type_: _description_
    """
    modes = matrix.shape[1]

    image_dim = hilbert_dim(modes, photons)
    image_matrix = np.zeros((image_dim, image_dim), dtype=np.complex64)

    image_basis = image_algebra_basis(modes, photons, orthonormal)
    counter = 0

    for i in range(modes):
        image_matrix += hs_inner_product(matrix, sym_matrix(i, i, modes)) * image_basis[counter]
        counter += 1
        for j in range(i):
            image_matrix += hs_inner_product(matrix, sym_matrix(i, j, modes)) * image_basis[counter]
            counter += 1
            image_matrix += hs_inner_product(matrix, antisym_matrix(i, j, modes)) * image_basis[counter]
            counter += 1

    return image_matrix


def image_matrix_from_coefs(coefs, modes: int, photons: int, orthonormal: bool = False):

    image_basis = image_algebra_basis(modes, photons, orthonormal)
    dim = hilbert_dim(modes, photons)

    tangent_matrix = np.zeros((dim, dim), dtype=np.complex128)

    for coef, matrix in zip(coefs, image_basis, strict=True):
        tangent_matrix += coef * matrix
    return tangent_matrix
