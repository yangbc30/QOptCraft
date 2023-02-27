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

# ---------------------------------------------------------------------------------------------------------------------------
# 													LIBRARIES REQUIRED
# ---------------------------------------------------------------------------------------------------------------------------


# ----------MATHEMATICAL FUNCTIONS/MATRIX OPERATIONS:----------

# NumPy instalation: in the cmd: 'py -m pip install numpy'
import numpy as np

# SciPy instalation: in the cmd: 'py -m pip install scipy'
from scipy.linalg import expm


# ----------FILE MANAGEMENT:----------

# File opening
from io import open


# ----------SYSTEM:----------

import sys


# ----------COMBINATORY:----------

from ..recur_factorial import comb_evol


# ----------ALGORITHM 2: AUXILIAR FUNCTIONS:----------

from ..Phase2_Aux._2_creation_and_destruction_operators import *


# ----------PHOTON COMB BASIS:----------

from ..photon_comb_basis import photon_combs_generator

import scipy as sp

import pickle
from typing import Sequence
import os

# ---------------------------------------------------------------------------------------------------------------------------
# 									MATRIX BASIS COMPUTATION IN SUBSPACES u(m) AND u(M)
# ---------------------------------------------------------------------------------------------------------------------------


# The functions e_jk y f_jk allow to obtain the matrix basis of u(m)
def e_jk(j, k, base):

    j_array = np.array([base[j]])

    k_array = np.array([base[k]])

    ejk = 0.5j * (np.transpose(j_array).dot(np.conj(k_array)) + np.transpose(k_array).dot(np.conj(j_array)))

    return ejk


def f_jk(j, k, base):

    j_array = np.array([base[j]])

    k_array = np.array([base[k]])

    fjk = 0.5 * (np.transpose(j_array).dot(np.conj(k_array)) - np.transpose(k_array).dot(np.conj(j_array)))

    return fjk


# We transform from the u(m) matrix basis to u(M)'s
def d_phi(base_matrix_m, photons, base_input):

    m = len(base_matrix_m)
    num_photons = int(np.sum(photons))

    if base_input == True:

        try:

            # We load the vector basis
            vec_base_file = open(f"m_{m}_n_{num_photons}_vec_base.txt", "r")

            vec_base = np.loadtxt(vec_base_file, delimiter=",", dtype=complex)

            vec_base_file.close()

        except FileNotFoundError:

            print("\nThe required vector basis file does not exist.\n")
            print("\nIt will be freshly generated instead.\n")

            # We load the combinations with the same amount of photons in order to create the vector basis
            vec_base = photon_combs_generator(m, photons)

            vec_base_file = open(f"m_{m}_n_{num_photons}_vec_base.txt", "w")

            np.savetxt(vec_base_file, vec_base, fmt="(%e)", delimiter=",")

            vec_base_file.close()

    else:

        # We load the combinations with the same amount of photons in order to create the vector basis
        vec_base = photon_combs_generator(m, photons)

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


# Specific process of subspace shift. Very similar to iH_U_operator's obtention (see '_2_3rd_evolution_method.py')
def u_m_to_u_M(m, M, vec_base, base_matrix_m):

    # base_matrix_M initialization
    base_matrix_M = np.zeros((M, M), dtype=complex)
    vec_base_canon = np.identity(M, dtype=complex)

    for p in range(M):

        p_array = np.array(vec_base[p])

        p_array_M = np.array(vec_base_canon[p])

        for q in range(M):

            q_array = np.array(vec_base[q])

            for j in range(m):

                for l in range(m):

                    # Array subject to the operators
                    q_array_aux = np.array(vec_base[q])

                    # Multiplier
                    mult = base_matrix_m[j, l]

                    # These two functions update q_array_aux and mult
                    q_array_aux, mult = a(l, q_array_aux, mult)

                    q_array_aux, mult = a_dagger(j, q_array_aux, mult)

                    for k in range(M):

                        if (vec_base[k] == q_array_aux).all():

                            index = k

                            break

                    q_array_M = np.array(vec_base_canon[index])

                    base_matrix_M[p, q] += p_array_M.dot(q_array_M) * mult

    return base_matrix_M


def matrix_u_basis_generator(m, M, photons, base_input):

    # We initialise the basis for each space
    base_U_m = np.identity(m, dtype=complex)

    base_U_M = np.identity(M, dtype=complex)

    base_u_m = np.zeros((m * m, m, m), dtype=complex)

    base_u_M = np.zeros((M * M, M, M), dtype=complex)

    # Here we will storage correlations with e_jk and f_jk, for a better organisation
    base_u_m_e = np.zeros((m * m, m, m), dtype=complex)

    base_u_m_f = np.zeros((m * m, m, m), dtype=complex)

    cont = 0

    for j in range(m):

        for k in range(m):

            base_u_m_e[m * j + k] = e_jk(j, k, base_U_m)

            if k <= j:

                base_u_m[cont] = e_jk(j, k, base_U_m)
                base_u_M[cont] = d_phi(base_u_m[cont], photons, base_input)

                cont += 1

    # The separator's functions indicate the switch from e_jk to f_jk,
    # after the m*m combinations have been already computed in the former
    separator_e_f = cont

    for j in range(m):

        for k in range(m):

            base_u_m_f[m * j + k] = f_jk(j, k, base_U_m)

            if k < j:

                base_u_m[cont] = f_jk(j, k, base_U_m)

                base_u_M[cont] = d_phi(base_u_m[cont], photons, base_input)

                cont += 1

    return base_u_m, base_u_M, base_u_m_e, base_u_m_f, separator_e_f, base_U_m, base_U_M


def matrix_u_basis_generator_sparse(m, M, photons, base_input):

    # We initialise the basis for each space
    base_U_m = np.identity(m, dtype=complex)
    base_U_M = np.identity(M, dtype=complex)
    base_u_m = []
    base_u_M = []

    # Here we will storage correlations with e_jk and f_jk, for a better organisation
    base_u_m_e = []
    base_u_m_f = []
    cont = 0
    for j in range(m):
        for k in range(m):
            base_u_m_e[m * j + k].append(sp.sparse.csr_matrix(e_jk(j, k, base_U_m)))
            if k <= j:
                base_u_m.append(sp.sparse.csr_matrix(e_jk(j, k, base_U_m)))
                base_u_M.append(sp.sparse.csr_matrix(d_phi(base_u_m[cont].toarray(), photons, base_input)))
                cont += 1
    # The separator's functions indicate the switch from e_jk to f_jk,
    # after the m*m combinations have been already computed in the former
    separator_e_f = cont
    for j in range(m):
        for k in range(m):
            base_u_m_f.append(sp.sparse.csr_matrix(f_jk(j, k, base_U_m)))
            if k < j:
                base_u_m.append(sp.sparse.csr_matrix(f_jk(j, k, base_U_m)))
                base_u_M.append(sp.sparse.csr_matrix(d_phi(base_u_m[cont].toarray(), photons, base_input)))
                cont += 1
    return base_u_m, base_u_M, base_u_m_e, base_u_m_f, separator_e_f, base_U_m, base_U_M


"""def matrix_u_basis_generator_sparse(m, M, photons, base_input):

    # We initialise the basis for each space
    base_U_m = np.identity(m, dtype=complex)
    base_U_M = np.identity(M, dtype=complex)
    base_u_m = []
    base_u_M = []
    # Here we will storage correlations with e_jk and f_jk, for a better organisation
    base_u_m_e = []
    base_u_m_f = []

    cont = 0
    for j in range(m):
        for k in range(m):
            base_u_m_e[m * j + k].append(sp.sparse.csr_matrix(e_jk(j, k, base_U_m)))
            if k <= j:
                base_u_m.append(sp.sparse.csr_matrix(e_jk(j, k, base_U_m)))
                base_u_M.append(
                    sp.sparse.csr_matrix(d_phi(base_u_m[cont], photons, base_input))
                )
                cont += 1

    # The separator's functions indicate the switch from e_jk to f_jk,
    # after the m*m combinations have been already computed in the former
    separator_e_f = cont
    for j in range(m):
        for k in range(m):
            base_u_m_f.append(sp.sparse.csr_matrix(f_jk(j, k, base_U_m)))
            if k < j:
                base_u_m.append(sp.sparse.csr_matrix(f_jk(j, k, base_U_m)))
                base_u_M.append(
                    sp.sparse.csr_matrix(d_phi(base_u_m[cont], photons, base_input))
                )
                cont += 1
    return base_u_m, base_u_M, base_u_m_e, base_u_m_f, separator_e_f, base_U_m, base_U_M"""


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
    base_u_m = []
    base_u_M = []

    # Here we will storage correlations with e_jk and f_jk, for a better organisation
    # base_u_m_sym = []
    # base_u_m_antisym = []

    cont = 0
    for j in range(dim):
        for k in range(j + 1):
            # base_u_m_sym.append(sp.sparse.csr_matrix(sym_algebra_basis(j, k, dim)))
            base_u_m.append(sp.sparse.csr_matrix(sym_algebra_basis(j, k, dim)))
            base_u_M.append(sp.sparse.csr_matrix(d_phi(base_u_m[cont].toarray(), photons, base_input)))
            cont += 1

    # The separator's functions indicate the switch from e_jk to f_jk,
    # after the m*m combinations have been already computed in the former
    separator_sym_antisym = cont

    for j in range(dim):
        for k in range(j):
            # base_u_m_antisym.append(sp.sparse.csr_matrix(antisym_algebra_basis(j, k, dim)))
            base_u_m.append(sp.sparse.csr_matrix(antisym_algebra_basis(j, k, dim)))
            base_u_M.append(sp.sparse.csr_matrix(d_phi(base_u_m[cont].toarray(), photons, base_input)))
            cont += 1

    with open(os.path.join(folder_path, "algebra.pkl"), "wb") as f:
        pickle.dump(base_u_m, f)

    with open(os.path.join(folder_path, "phi_algebra.pkl"), "wb") as f:
        pickle.dump(base_u_M, f)

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
