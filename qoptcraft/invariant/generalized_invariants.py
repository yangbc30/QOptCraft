from itertools import product

import numpy as np

from qoptcraft.state import State
from qoptcraft.basis import unitary_algebra_basis, image_algebra_basis
from qoptcraft.math import Matrix


def scalar_invariant(state: State, modes: int, photons: int, order: int) -> Matrix:

    scattering_basis = unitary_algebra_basis(modes)
    image_basis = image_algebra_basis(modes, photons)

    def coefficient(indices):
        matrix = np.eye(modes, dtype=np.complex64)
        for i in indices:
            matrix @= scattering_basis[i]
        return np.trace(matrix)

    invariant = 0

    for indices in product(*[list(range(modes * modes))] * order):
        coef = coefficient(indices)
        if coef != 0:
            for idx in indices:
                coef *= np.trace(image_basis[idx] @ state.density_matrix)
            invariant += coef

    return invariant
