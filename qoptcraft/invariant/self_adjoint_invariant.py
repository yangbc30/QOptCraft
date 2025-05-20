from itertools import product, groupby

import numpy as np
from numpy.typing import NDArray

from qoptcraft.basis import (
    image_algebra_basis,
    unitary_algebra_basis,
    hilbert_dim,
    hermitian_matrix_from_coefs,
)
from qoptcraft.math import hs_scalar_product, commutator
from qoptcraft.invariant import invariant_coef
from qoptcraft.utils import saved_basis


@saved_basis(file_name="subspaces_nested_commutator.pkl")
def invariant_subspaces_nested_commutator(
    modes, photons, order, orthonormal: bool = False, cache: bool = True
):
    _ = cache  # only used by the decorator @saved_basis

    operator = self_adjoint_projection(modes, photons, order, orthonormal=orthonormal)
    eigenvalues, eigenvectors = np.linalg.eigh(operator)
    eigenvectors[:, :] = eigenvectors.T  # eigenvectors will be rows instead of columns
    eigenvalues[:], eigenvectors[:] = zip(
        *sorted(zip(eigenvalues.round(6), eigenvectors.round(6), strict=True), key=lambda p: p[0]),
        strict=True,
    )

    return [
        [hermitian_matrix_from_coefs(c) for _, c in g]
        for _, g in groupby(
            zip(eigenvalues.round(6), eigenvectors.round(6), strict=True), key=lambda x: x[0]
        )
    ]


def self_adjoint_projection(modes, photons, order, orthonormal=False) -> NDArray:

    dim = hilbert_dim(modes, photons)

    scattering_basis = unitary_algebra_basis(modes)
    image_basis = image_algebra_basis(modes, photons, orthonormal)
    full_basis = unitary_algebra_basis(dim)

    operator_matrix = np.zeros((dim**2, dim**2), dtype=np.complex128)

    for i, matrix_1 in enumerate(full_basis):
        projected_matrix_1 = operator_nested_commutator(
            matrix_1, order, scattering_basis, image_basis
        )
        for j, matrix_2 in enumerate(full_basis):
            operator_matrix[i, j] = hs_scalar_product(matrix_2, projected_matrix_1)

    return operator_matrix


def operator_nested_commutator(matrix, order: int, scattering_basis, image_basis):
    """Theorem 11."""

    modes = scattering_basis[0].shape[0]

    mapped_matrix = np.zeros(image_basis[0].shape, dtype=np.complex128)

    for indices in product(*[list(range(modes * modes))] * order):
        coef = invariant_coef(indices, scattering_basis)
        if coef != 0:
            temp_matrix = commutator(image_basis[indices[0]], matrix)
            for idx in indices[1:]:
                temp_matrix = commutator(image_basis[idx], temp_matrix)
            mapped_matrix += coef * temp_matrix

    return 1j**order * mapped_matrix
