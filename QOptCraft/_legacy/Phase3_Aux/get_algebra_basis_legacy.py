"""Copyright 2021 Daniel Gómez Aguado

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

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


# The functions e_jk y f_jk allow to obtain the matrix basis of u(m)
def e_jk(j, k, base):
    j_array = np.array([base[j]])

    k_array = np.array([base[k]])

    ejk = 0.5j * (
        np.transpose(j_array).dot(np.conj(k_array)) + np.transpose(k_array).dot(np.conj(j_array))
    )

    return ejk


def f_jk(j, k, base):
    j_array = np.array([base[j]])

    k_array = np.array([base[k]])

    fjk = 0.5 * (
        np.transpose(j_array).dot(np.conj(k_array)) - np.transpose(k_array).dot(np.conj(j_array))
    )

    return fjk


# We transform from the u(m) matrix basis to u(M)'s
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


def matrix_u_basis_generator(m, M, photons, base_input):
    # We initialise the basis for each space
    base_group = np.identity(m, dtype=complex)
    base_img_group = np.identity(M, dtype=complex)
    base_algebra = np.zeros((m * m, m, m), dtype=complex)
    base_img_algebra = np.zeros((m * m, M, M), dtype=complex)
    # Here we will storage correlations with e_jk and f_jk, for a better organisation
    base_algebra_e = np.zeros((m * m, m, m), dtype=complex)
    base_algebra_f = np.zeros((m * m, m, m), dtype=complex)

    cont = 0

    for j in range(m):
        for k in range(m):
            # base_algebra_e[m * j + k] = e_jk(j, k, base_group)

            if k <= j:
                base_algebra[cont, ...] = e_jk(j, k, base_group)
                base_img_algebra[cont, ...] = d_phi(base_algebra[cont], photons)

                cont += 1

    # The separator's functions indicate the switch from e_jk to f_jk,
    # after the m*m combinations have been already computed in the former
    separator_sym_antisym = cont

    for j in range(m):
        for k in range(m):
            # base_algebra_f[m * j + k] = f_jk(j, k, base_group)

            if k < j:
                base_algebra[cont] = f_jk(j, k, base_group)

                base_img_algebra[cont] = d_phi(base_algebra[cont], photons)

                cont += 1

    return (
        base_algebra,
        base_img_algebra,
        base_algebra_e,
        base_algebra_f,
        separator_sym_antisym,
        base_group,
        base_img_group,
    )
