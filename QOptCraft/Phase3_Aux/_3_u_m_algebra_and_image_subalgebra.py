"""Copyright 2021 Daniel Gómez Aguado

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License."""

import os
import pickle
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
import scipy as sp

from QOptCraft.basis import photon_basis


def creation_op(mode, state, coef):
    photons = state[mode]

    coef *= np.sqrt(photons)
    state[mode] = photons - 1

    return state, coef


def annihil_op(mode, state, coef):
    photons = state[mode]

    coef *= np.sqrt(photons + 1)
    state[mode] = photons + 1

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


def get_basis(photons: int, modes: int) -> list[list[int]]:
    """Return a basis for the Hilbert space with n photons and m modes.
    If the basis was saved retrieve it, otherwise the function creates
    and saves the basis to a text file.

    Args:
        photons (int): number of photons.
        modes (int): number of modes.

    Returns:
        list[list[int]]: basis of the Hilbert space.
    """
    basis_path = os.path.join("save_basis", f"m={modes} n={photons}")
    try:
        with open(basis_path) as f:
            basis = pickle.load(f)

    except FileNotFoundError:
        print("Basis not found.\nGenerating basis...")
        basis = photon_basis(modes, photons)
        with open(basis_path, "w") as f:
            pickle.dump(basis, f)
        print(f"Basis saved in {basis_path}.")

    return basis


# We transform from the u(m) matrix basis to u(M)'s
def d_phi(matrix: NDArray, photons: int) -> NDArray:
    modes = len(matrix)
    dim = int(sp.special.comb(modes + photons - 1, photons))

    img_matrix = np.zeros((dim, dim), dtype=complex)
    basis_canon = np.identity(dim, dtype=complex)

    basis = photon_basis(photons, modes)

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
                    q_array_aux, mult = creation_op(k, q_array_aux, mult)
                    q_array_aux, mult = annihil_op(j, q_array_aux, mult)

                    for r in range(dim):
                        if (basis[r] == q_array_aux).all():
                            index = r
                            break

                    q_array_M = np.array(basis_canon[index])
                    img_matrix[p, q] += p_array_M.dot(q_array_M) * mult

    return img_matrix


def algebra_basis_sparse(modes: int, dim_img: int, photons):
    """Given"""
    # We initialise the basis for each space
    basis_group = np.identity(modes, dtype=complex)
    basis_img_group = np.identity(dim_img, dtype=complex)
    basis_algebra = []
    basis_img_algebra = []

    # Here we will storage correlations with e_jk and f_jk, for a better organisation
    basis_sym_algebra = []
    basis_antisym_algebra = []
    cont = 0
    for j in range(modes):
        for k in range(modes):
            basis_sym_algebra.append(sp.sparse.csr_matrix(e_jk(j, k, basis_group)))
            if k <= j:
                basis_algebra.append(sp.sparse.csr_matrix(e_jk(j, k, basis_group)))
                basis_img_algebra.append(
                    sp.sparse.csr_matrix(d_phi(basis_algebra[cont].toarray(), photons))
                )
                cont += 1
    # The separator's functions indicate the switch from e_jk to f_jk,
    # after the m*m combinations have been already computed in the former
    separator_e_f = cont
    for j in range(modes):
        for k in range(modes):
            basis_antisym_algebra.append(sp.sparse.csr_matrix(f_jk(j, k, basis_group)))
            if k < j:
                basis_algebra.append(sp.sparse.csr_matrix(f_jk(j, k, basis_group)))
                basis_img_algebra.append(
                    sp.sparse.csr_matrix(d_phi(basis_algebra[cont].toarray(), photons))
                )
                cont += 1
    return (
        basis_algebra,
        basis_img_algebra,
        basis_sym_algebra,
        basis_antisym_algebra,
        separator_e_f,
        basis_group,
        basis_img_group,
    )


def write_algebra_basis(dim: int, photons: Sequence, base_input: bool) -> None:
    num_photons = sum(photons)

    folder_path = os.path.join("save_basis", f"m={dim} n={num_photons}")
    try:
        os.makedirs(folder_path)
    except FileExistsError:
        print("This basis has already been computed, do you want to overwrite it?")
        while True:
            user_input = input("Press y/n: ")

            if user_input.lower() in ["yes", "y"]:
                break
            elif user_input.lower() in ["no", "n"]:
                print("Program finished")
                return
            else:
                continue

    # Here we will storage correlations with e_jk and f_jk, for a better organisation
    base_algebra = []
    base_img_algebra = []

    # Here we will storage correlations with e_jk and f_jk, for a better organisation
    # base_algebra_sym = []
    # base_algebra_antisym = []

    cont = 0
    for j in range(dim):
        for k in range(j + 1):
            # base_algebra_sym.append(sp.sparse.csr_matrix(sym_algebra_basis(j, k, dim)))
            base_algebra.append(sp.sparse.csr_matrix(sym_algebra_basis(j, k, dim)))
            base_img_algebra.append(
                sp.sparse.csr_matrix(d_phi(base_algebra[cont].toarray(), photons))
            )
            cont += 1

    # The separator's functions indicate the switch from e_jk to f_jk,
    # after the m*m combinations have been already computed in the former
    separator_sym_antisym = cont

    for j in range(dim):
        for k in range(j):
            # base_algebra_antisym.append(sp.sparse.csr_matrix(antisym_algebra_basis(j, k, dim)))
            base_algebra.append(sp.sparse.csr_matrix(antisym_algebra_basis(j, k, dim)))
            base_img_algebra.append(
                sp.sparse.csr_matrix(d_phi(base_algebra[cont].toarray(), photons))
            )
            cont += 1

    with open(os.path.join(folder_path, "algebra.pkl"), "wb") as f:
        pickle.dump(base_algebra, f)

    with open(os.path.join(folder_path, "phi_algebra.pkl"), "wb") as f:
        pickle.dump(base_img_algebra, f)

    with open(os.path.join(folder_path, "separator.txt"), "w") as f:
        f.write(f"separator_sym_antisym = {separator_sym_antisym}")


def sym_algebra_basis(index_1: int, index_2: int, dim: int) -> np.ndarray:
    """Create the element of the algebra i/2(|j><k| + |k><j|)."""
    basis_matrix = sp.sparse.csr_matrix((dim, dim), dtype="complex64")
    basis_matrix[index_1, index_2] = 0.5j
    basis_matrix[index_2, index_1] = 0.5j

    return basis_matrix


def antisym_algebra_basis(index_1: int, index_2: int, dim: int) -> np.ndarray:
    """Create the element of the algebra 1/2(|j><k| - |k><j|)."""
    basis_matrix = sp.sparse.csr_matrix((dim, dim), dtype="complex64")
    basis_matrix[index_1, index_2] = 0.5
    basis_matrix[index_2, index_1] = -0.5

    return basis_matrix


def matrix_u_basis_generator(m, M, photons, base_input):
    # We initialise the basis for each space
    base_group = np.identity(m, dtype=complex)
    base_img_group = np.identity(M, dtype=complex)
    base_algebra = np.zeros((m * m, m, m), dtype=complex)
    base_img_algebra = np.zeros((M * M, M, M), dtype=complex)
    # Here we will storage correlations with e_jk and f_jk, for a better organisation
    base_algebra_e = np.zeros((m * m, m, m), dtype=complex)
    base_algebra_f = np.zeros((m * m, m, m), dtype=complex)

    cont = 0

    for j in range(m):
        for k in range(m):
            base_algebra_e[m * j + k] = e_jk(j, k, base_group)

            if k <= j:
                base_algebra[cont] = e_jk(j, k, base_group)
                base_img_algebra[cont] = d_phi(base_algebra[cont], photons)

                cont += 1

    # The separator's functions indicate the switch from e_jk to f_jk,
    # after the m*m combinations have been already computed in the former
    separator_sym_antisym = cont

    for j in range(m):
        for k in range(m):
            base_algebra_f[m * j + k] = f_jk(j, k, base_group)

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


"""
# We transform from the u(m) matrix basis to u(M)'s
def d_phi(base_matrix_m, num_photons, base_input):
    m = len(base_matrix_m)
    num_photons = int(np.sum(photon_state))

    if base_input is True:
        try:
            # We load the vector basis
            with open(f"m_{m}_n_{num_photons}_vec_base.txt") as vec_base_file:
                vec_base = np.loadtxt(vec_base_file, delimiter=",", dtype=complex)

        except FileNotFoundError:
            print("\nThe required vector basis file does not exist.\n")
            print("\nIt will be freshly generated instead.\n")

            # We load the combinations with the same amount of photons in order to create the vector basis
            vec_base = photon_combs_generator(m, num_photons)
            with open(f"m_{m}_n_{num_photons}_vec_base.txt", "w") as vec_base_file:
                np.savetxt(vec_base_file, vec_base, fmt="(%e)", delimiter=",")

    else:
        # We load the combinations with the same amount of photons in order to create the vector basis
        vec_base = photon_combs_generator(m, num_photons)

    # It is required to introduce photons_aux for 'photons_aux' and 'photons' not to "update" together
    global photons_aux
    global mult

    # Dimensions of the resulting matrix U:
    M = comb_evol(num_photons, m)
    # This value can be otained too with the measurement of vec_base's length

    # base_matrix_M initialization
    base_matrix_M = np.zeros((M, M), dtype=complex)

    base_matrix_M = u_m_to_u_M(m, M, vec_base, base_matrix_m)

    return base_matrix_M
"""


"""
# Specific process of subspace shift. Very similar to iH_U_operator's obtention (see '_2_3rd_evolution_method.py')
def u_m_to_u_M(modes, dim, photons, base_matrix_m):
    # base_matrix_M initialization
    base_matrix_M = np.zeros((dim, dim), dtype=complex)
    basis_canon = np.identity(dim, dtype=complex)

    basis = photon_basis(photons, modes)

    for p in range(dim):
        np.array(basis[p])

        p_array_M = np.array(basis_canon[p])

        for q in range(dim):
            for j in range(modes):
                for k in range(modes):
                    # Array subject to the operators
                    q_array_aux = np.array(basis[q])

                    # Multiplier
                    mult = base_matrix_m[j, k]

                    # These two functions update q_array_aux and mult
                    q_array_aux, mult = a(k, q_array_aux, mult)

                    q_array_aux, mult = a_dagger(j, q_array_aux, mult)

                    for r in range(dim):
                        if (basis[r] == q_array_aux).all():
                            index = r

                            break

                    q_array_M = np.array(basis_canon[index])

                    base_matrix_M[p, q] += p_array_M.dot(q_array_M) * mult

    return base_matrix_M
"""

"""
def matrix_u_basis_generator_sparse(m, M, photons, base_input):

    # We initialise the basis for each space
    base_group = np.identity(m, dtype=complex)
    base_img_group = np.identity(M, dtype=complex)
    base_algebra = []
    base_img_algebra = []
    # Here we will storage correlations with e_jk and f_jk, for a better organisation
    base_algebra_e = []
    base_algebra_f = []

    cont = 0
    for j in range(m):
        for k in range(m):
            base_algebra_e[m * j + k].append(sp.sparse.csr_matrix(e_jk(j, k, base_group)))
            if k <= j:
                base_algebra.append(sp.sparse.csr_matrix(e_jk(j, k, base_group)))
                base_img_algebra.append(
                    sp.sparse.csr_matrix(d_phi(base_algebra[cont], photons, base_input))
                )
                cont += 1

    # The separator's functions indicate the switch from e_jk to f_jk,
    # after the m*m combinations have been already computed in the former
    separator_e_f = cont
    for j in range(m):
        for k in range(m):
            base_algebra_f.append(sp.sparse.csr_matrix(f_jk(j, k, base_group)))
            if k < j:
                base_algebra.append(sp.sparse.csr_matrix(f_jk(j, k, base_group)))
                base_img_algebra.append(
                    sp.sparse.csr_matrix(d_phi(base_algebra[cont], photons, base_input))
                )
                cont += 1
    return base_algebra, base_img_algebra, base_algebra_e, base_algebra_f, separator_e_f, base_group, base_img_group
"""
