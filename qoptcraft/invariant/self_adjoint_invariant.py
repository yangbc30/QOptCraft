from itertools import groupby, product
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from qoptcraft.basis import (
    hermitian_matrix_from_coefs,
    hilbert_dim,
    image_algebra_basis,
    unitary_algebra_basis,
)
from qoptcraft.invariant import invariant_coef
from qoptcraft.math import commutator, hs_inner_product, hs_scalar_product
from qoptcraft.utils import saved_basis


@saved_basis(file_name="subspaces")
def invariant_subspaces(
    modes,
    photons,
    *,
    invariant_operator: Literal["higher_order_projection", "nested_commutator"],
    order: int,
    orthonormal: bool = False,
    cache: bool = True,
    parallelize: bool = True
):
    _ = cache  # only used by the decorator @saved_basis

    operator = self_adjoint_projection(modes, photons, invariant_operator, order, orthonormal, parallelize)
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


def self_adjoint_projection(
    modes,
    photons,
    invariant_operator: Literal["higher_order_projection", "nested_commutator"],
    order: int,
    orthonormal=False,
    parallelize: bool = True
) -> NDArray:
    dim = hilbert_dim(modes, photons)

    if invariant_operator == "nested_commutator":
        scattering_basis = unitary_algebra_basis(modes)
    image_basis = image_algebra_basis(modes, photons, orthonormal)
    full_basis = unitary_algebra_basis(dim)

    def compute_row(i):
        matrix = full_basis[i]

        if invariant_operator == "higher_order_projection":
            projected = operator_higher_order_projection(matrix, modes, order, image_basis)
        elif invariant_operator == "nested_commutator":
            projected = operator_nested_commutator(matrix, order, scattering_basis, image_basis)
        else:
            raise ValueError(f"Invalid invariant_operator: {invariant_operator}")

        row = np.array([hs_scalar_product(matrix, projected) for matrix in full_basis])
        return i, row

    operator_matrix = np.zeros((dim**2, dim**2), dtype=np.complex128)

    if parallelize:
        from pathos.multiprocessing import ProcessingPool, cpu_count
        with ProcessingPool(nodes=cpu_count()) as pool:
            parallelized_rows = pool.map(compute_row, range(len(full_basis)))
        for i, row in parallelized_rows:
            operator_matrix[i, :] = row
    else:
        for i in range(len(full_basis)):
            operator_matrix[i, :] = compute_row(i)[1]  # first output is just i

    return operator_matrix


def operator_higher_order_projection(matrix, modes: int, order: int, image_basis):
    projection = np.zeros_like(image_basis[0].toarray())
    for indices in product(*[list(range(modes * modes))] * order):
        image_matrix = image_basis[indices[0]]
        for idx in indices[1:]:
            image_matrix = image_matrix @ image_basis[idx]
        projection += hs_inner_product(image_matrix, matrix) * image_matrix
    return projection


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

    return mapped_matrix
