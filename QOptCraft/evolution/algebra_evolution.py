from numbers import Number

import numpy as np
from numpy.typing import NDArray

from qoptcraft.basis import _photon_basis, hilbert_dim


def creation(mode: int, state: NDArray, coef: Number) -> tuple[NDArray, Number]:
    """Creation operator acting on a specific mode. Modifies state in-place.

    Args:
        mode (int): a quantum mode.
        state (list[int]): fock basis state.
        coef (Number): coefficient of the state.

    Returns:
        tuple[list[int], Number]: created state and its coefficient.
    """
    photons = state[mode]
    coef *= np.sqrt(photons + 1)
    state[mode] = photons + 1
    return state, coef


def annihilation(mode: int, state: NDArray, coef: Number) -> tuple[NDArray, Number]:
    """Annihilation operator acting on a specific mode.

    Args:
        mode (int): a quantum mode.
        state (list[int]): fock basis state.
        coef (Number): coefficient of the state.

    Returns:
        tuple[list[int], Number]: annihilated state and its coefficient.
    """
    photons = state[mode]
    coef *= np.sqrt(photons)
    state[mode] = photons - 1
    return state, coef


def d_phi(matrix: NDArray, photons: int) -> NDArray:
    modes = matrix.shape[0]
    dim = hilbert_dim(modes, photons)

    img_matrix = np.zeros((dim, dim), dtype=complex)
    basis_canon = np.identity(dim, dtype=complex)

    basis = _photon_basis(modes, photons)

    assert len(basis) == dim

    for p in range(dim):
        p_array_M = np.array(basis_canon[p])

        for q, basis_vector in enumerate(basis):
            for j in range(modes):
                for k in range(modes):
                    # Array subject to the operators
                    q_array_aux = np.array(basis_vector)

                    # Multiplier
                    mult = matrix[j, k]
                    # These two functions update q_array_aux and mult
                    q_array_aux, mult = creation(k, q_array_aux, mult)
                    q_array_aux, mult = annihilation(j, q_array_aux, mult)

                    for r in range(dim):
                        if (basis[r] == q_array_aux).all():
                            index = r
                            break

                    q_array_M = np.array(basis_canon[index])
                    img_matrix[p, q] += p_array_M.dot(q_array_M) * mult

    return img_matrix
