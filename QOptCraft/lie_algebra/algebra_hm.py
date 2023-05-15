import os
import pickle
from collections.abc import Sequence
from numbers import Number

import numpy as np
from numpy.typing import NDArray
import scipy as sp

from QOptCraft.basis import photon_basis


def creation_op(mode: int, state: list[int], coef: Number) -> tuple[list[int], Number]:
    """Creation operator acting on a specific mode.

    Args:
        mode (int): a quantum mode.
        state (list[int]): fock basis state.
        coef (Number): coefficient of the state.

    Returns:
        tuple[list[int], Number]: created state and its coefficient.
    """
    photons = state[mode]
    coef *= np.sqrt(photons)
    state[mode] = photons - 1
    return state, coef


def annihil_op(mode: int, state: list[int], coef: Number) -> tuple[list[int], Number]:
    """Annihilation operator acting on a specific mode.

    Args:
        mode (int): a quantum mode.
        state (list[int]): fock basis state.
        coef (Number): coefficient of the state.

    Returns:
        tuple[list[int], Number]: annihilated state and its coefficient.
    """
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
    try:
        with open(f"m_{modes}_n_{photons}_basis.txt") as basis_file:
            basis = np.loadtxt(basis_file, delimiter=",", dtype=complex)

    except FileNotFoundError:
        print("\nThe required vector basis file does not exist.\n")
        print("\nIt will be freshly generated instead.\n")

        basis = photon_basis(modes, photons)
        with open(f"m_{modes}_n_{photons}_vec_base.txt", "w") as basis_file:
            np.savetxt(basis_file, basis, fmt="(%e)", delimiter=",")

    return basis


# We transform from the u(m) matrix basis to u(M)'s
def algebra_hm(matrix: NDArray, photons: int) -> NDArray:
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
                    sp.sparse.csr_matrix(algebra_hm(basis_algebra[cont].toarray(), photons))
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
                    sp.sparse.csr_matrix(algebra_hm(basis_algebra[cont].toarray(), photons))
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
            base_algebra.append(sp.sparse.csr_matrix(sym_algebra_matrix(j, k, dim)))
            base_img_algebra.append(
                sp.sparse.csr_matrix(algebra_hm(base_algebra[cont].toarray(), photons, base_input))
            )
            cont += 1

    # The separator's functions indicate the switch from e_jk to f_jk,
    # after the m*m combinations have been already computed in the former
    separator_sym_antisym = cont

    for j in range(dim):
        for k in range(j):
            # base_algebra_antisym.append(sp.sparse.csr_matrix(antisym_algebra_basis(j, k, dim)))
            base_algebra.append(sp.sparse.csr_matrix(antisym_algebra_matrix(j, k, dim)))
            base_img_algebra.append(
                sp.sparse.csr_matrix(algebra_hm(base_algebra[cont].toarray(), photons, base_input))
            )
            cont += 1

    with open(os.path.join(folder_path, "algebra.pkl"), "wb") as f:
        pickle.dump(base_algebra, f)

    with open(os.path.join(folder_path, "phi_algebra.pkl"), "wb") as f:
        pickle.dump(base_img_algebra, f)

    with open(os.path.join(folder_path, "separator.txt"), "w") as f:
        f.write(f"separator_sym_antisym = {separator_sym_antisym}")


def sym_algebra_matrix(index_1: int, index_2: int, dim: int) -> np.ndarray:
    """Create the element of the algebra i/2(|j><k| + |k><j|)."""
    basis_matrix = sp.sparse.csr_matrix((dim, dim), dtype="complex64")
    basis_matrix[index_1, index_2] = 0.5j
    basis_matrix[index_2, index_1] = 0.5j

    return basis_matrix


def antisym_algebra_matrix(index_1: int, index_2: int, dim: int) -> np.ndarray:
    """Create the element of the algebra 1/2(|j><k| - |k><j|)."""
    basis_matrix = sp.sparse.csr_matrix((dim, dim), dtype="complex64")
    basis_matrix[index_1, index_2] = 0.5
    basis_matrix[index_2, index_1] = -0.5

    return basis_matrix


def image_sym_matrix(mode_1: int, mode_2: int, modes: int, photons: int) -> sp.sparse.spmatrix:
    dim = int(sp.special.comb(modes + photons - 1, photons))
    matrix = sp.sparse.csr_matrix((dim, dim), dtype="complex64")

    folder_path = os.path.join("save_basis", f"m={modes} n={photons}")
    with open(os.path.join(folder_path, "photon_basis.pkl"), "wb") as f:
        basis = pickle.load(f)

    for idx_col, vector in enumerate(basis):
        if vector[mode_1] != 0:
            new_vector, new_coef = annihil_op(vector, mode_1)
            new_vector, coef = creation_op(new_vector, mode_2)
            new_coef *= coef
            idx_new_vector = basis.index(new_vector)

            matrix[idx_new_vector, idx_col] = 0.5j * new_coef

        if vector[mode_2] != 0:
            new_vector, new_coef = annihil_op(vector, mode_2)
            new_vector, coef = creation_op(new_vector, mode_1)
            new_coef *= coef
            idx_new_vector = basis.index(new_vector)

            matrix[idx_new_vector, idx_col] = 0.5j * new_coef

    return matrix


def image_antisym_matrix(mode_1: int, mode_2: int, modes: int, photons: int) -> sp.sparse.spmatrix:
    dim = int(sp.special.comb(modes + photons - 1, photons))
    matrix = sp.sparse.csr_matrix((dim, dim), dtype="complex64")

    folder_path = os.path.join("save_basis", f"m={modes} n={photons}")
    with open(os.path.join(folder_path, "photon_basis.pkl"), "wb") as f:
        basis = pickle.load(f)

    for idx_col, vector in enumerate(basis):
        if vector[mode_1] != 0:
            new_vector, new_coef = annihil_op(vector, mode_1)
            new_vector, coef = creation_op(new_vector, mode_2)
            new_coef *= coef
            idx_new_vector = basis.index(new_vector)

            matrix[idx_new_vector, idx_col] = -0.5 * new_coef

        if vector[mode_2] != 0:
            new_vector, new_coef = annihil_op(vector, mode_2)
            new_vector, coef = creation_op(new_vector, mode_1)
            new_coef *= coef
            idx_new_vector = basis.index(new_vector)

            matrix[idx_new_vector, idx_col] = 0.5 * new_coef

    return matrix
