from itertools import product

import numpy as np

from qoptcraft.basis import unitary_algebra_basis, image_algebra_basis, hilbert_dim
from qoptcraft.math import Matrix


def invariant_operator(modes: int, photons: int, order: int) -> Matrix:

    scattering_basis = unitary_algebra_basis(modes)
    image_basis = image_algebra_basis(modes, photons)

    def coefficient(indices):
        matrix = np.eye(modes, dtype=np.complex64)
        for i in indices:
            matrix @= scattering_basis[i]
        return np.trace(matrix)

    dim = hilbert_dim(modes, photons)
    invariant = np.zeros((dim, dim), dtype=np.complex64)

    for indices in product(*[list(range(modes * modes))] * order):
        matrix = np.eye(dim, dtype=np.complex128)
        coef = coefficient(indices)
        if coef != 0:
            for idx in indices:
                matrix @= coef * image_basis[idx]
            invariant += coef * matrix
        # print(f"{indices = }, {coef = }, {matrix = }")

    return invariant
