from itertools import product
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from qoptcraft.basis import image_algebra_basis, unitary_algebra_basis
from qoptcraft.math import Matrix, gram_schmidt, hs_inner_product
from qoptcraft.state import State


def projection_density(
    state: State,
    subspace: list[NDArray] | Literal["preimage", "image", "complement", "full"] = "preimage",
    orthonormal: bool = False,
) -> Matrix:
    """Project a state onto the linear optical subalgebra or its complement.

    Args:
        state (State): a photonic quantum state.
        subspace (str): "image" or "complement".
        orthonormal (bool, optional): if true, it orthonormalizes the basis of linear optical
            hamiltonians before projecting onto it. Defaults to False.

    Returns:
        Matrix: the projected density matrix.
    """
    matrix = state.density_matrix

    if subspace == "preimage":
        unitary_basis = unitary_algebra_basis(state.modes)
        image_basis = image_algebra_basis(state.modes, state.photons, orthonormal)

        projection = np.zeros_like(unitary_basis[0])
        for unitary_matrix, image_matrix in zip(unitary_basis, image_basis, strict=True):
            projection += hs_inner_product(image_matrix, matrix) * unitary_matrix
        return projection

    if isinstance(subspace, list):
        if orthonormal:
            subspace = gram_schmidt(subspace)
        projection = np.zeros_like(matrix)
        for basis_matrix in subspace:
            projection += hs_inner_product(basis_matrix, matrix) * basis_matrix
        return projection

    image_basis = image_algebra_basis(state.modes, state.photons, orthonormal)

    projection_image = np.zeros_like(matrix)
    for basis_matrix in image_basis:
        projection_image += hs_inner_product(basis_matrix, matrix) * basis_matrix
    if subspace == "image":
        return projection_image
    if subspace == "complement" and orthonormal is True:
        return matrix - projection_image
    raise ValueError("The complement invariant only makes sense if the basis is orthonormal.")


def projection_coefs(state, orthonormal=False):
    image_basis = image_algebra_basis(state.modes, state.photons, orthonormal)

    coefs = []
    matrix = state.density_matrix
    for basis_matrix in image_basis:
        coefs.append(hs_inner_product(basis_matrix, matrix))
    return np.array(coefs)


def higher_order_projection_density(
    state: State,
    order: int,
    subspace: Literal["preimage", "image"] = "image",
    orthonormal: bool = False,
) -> Matrix:
    """Project a state onto the linear optical subalgebra or its complement.

    Args:
        state (State): a photonic quantum state.
        subspace (str): "image" or "complement".
        orthonormal (bool, optional): if true, it orthonormalizes the basis of linear optical
            hamiltonians before projecting onto it. Defaults to False.

    Returns:
        Matrix: the projected density matrix.
    """
    density_matrix = state.density_matrix

    unitary_basis = unitary_algebra_basis(state.modes)
    image_basis = image_algebra_basis(state.modes, state.photons, orthonormal)

    if subspace == "preimage":
        projection = np.zeros_like(unitary_basis[0])
        for indices in product(*[list(range(state.modes * state.modes))] * order):
            hermitian_matrix = unitary_basis[indices[0]]
            image_matrix = image_basis[indices[0]]
            for idx in indices[1:]:
                hermitian_matrix = hermitian_matrix @ unitary_basis[idx]
                image_matrix = image_matrix @ image_basis[idx]
            projection += hs_inner_product(image_matrix, density_matrix) * hermitian_matrix
    elif subspace == "image":
        projection = np.zeros_like(image_basis[0].toarray())
        for indices in product(*[list(range(state.modes * state.modes))] * order):
            image_matrix = image_basis[indices[0]]
            for idx in indices[1:]:
                image_matrix = image_matrix @ image_basis[idx]
            projection += hs_inner_product(image_matrix, density_matrix) * image_matrix
    else:
        raise ValueError(f"Only admited subspaces are preimage and image, but {subspace = }.")
    return projection
