"""Orthonormal basis of the image and the perpendicular subspaces
of the Lie algebra u(M), where M is the dimension of the Hilbert space
of states with n photons and m modes.
"""
import pickle

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import spmatrix

from .algebra import image_algebra_basis, unitary_algebra_basis, BasisAlgebra
from .hilbert_dimension import hilbert_dim
from qoptcraft.math import gram_schmidt, hs_scalar_product, hs_norm
from qoptcraft import config


def _saved_orthonormal_image_algebra_basis(modes: int, photons: int) -> BasisAlgebra:
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

    basis_image_path = folder_path / "orthonormal_image_algebra.pkl"
    basis_image_path.touch()  # create file if it doesn't exist
    try:
        with basis_image_path.open("rb") as f:
            basis_image = pickle.load(f)
    except EOFError:
        basis_image = basis_image_orthonormal(modes, photons, cache=False)
        with basis_image_path.open("wb") as f:
            pickle.dump(basis_image, f)
        print(f"Orthonormal image algebra basis saved in {folder_path}")

    return basis_image


def basis_image_orthonormal(modes: int, photons: int, cache: bool = True) -> list[spmatrix] | list[NDArray]:
    if cache:
        return _saved_orthonormal_image_algebra_basis(modes, photons)
    basis_image = image_algebra_basis(modes, photons)
    return gram_schmidt(basis_image)


def basis_complement_image_orthonormal(modes: int, photons: int, cache: bool = True) -> list[spmatrix] | list[NDArray]:
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
