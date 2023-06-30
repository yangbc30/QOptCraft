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

import numpy as np


def a(num_vec, array, mult):
    n = array[num_vec]

    mult *= np.sqrt(n)
    array[num_vec] = n - 1

    return array, mult


def a_dagger(num_vec, array, mult):
    n = array[num_vec]

    mult *= np.sqrt(n + 1)
    array[num_vec] = n + 1

    return array, mult


def photon_combs_generator(m, photons):
    global photons_aux
    global vec_base
    global check

    check = 0

    num_photons = int(np.sum(photons))

    counter = np.array(photons[:], dtype=int)

    counter_sum = np.zeros(num_photons, dtype=int)

    # The last two terms are required because of the function's recursive character
    photon_combs_generator_loop(photons, num_photons, m, counter_sum, 0)

    return vec_base


# Loop whose amount of callings depend on the number of photons in each mode
def photon_combs_generator_loop(photons, num_photons, m, sum_, k):
    global photons_aux
    global vec_base
    global check

    counter = np.array(photons[:], dtype=int)

    for sum_[k] in range(m):
        if k < num_photons - 1:
            photon_combs_generator_loop(photons, num_photons, m, sum_, k + 1)

        else:
            photons_aux = np.zeros(m, dtype=complex)

            cont = 0  # IMPORTANT, we want to explore sum_[] in order

            for p in range(m):
                for q in range(counter[p]):
                    photons_aux[sum_[cont]] += 1

                    cont += 1

            if check != 0:
                vec_base = photon_comb_basis(photons_aux, vec_base)

            else:
                vec_base = np.array([np.real(photons_aux)])

                check = 1


# Required vector basis creator
def photon_comb_basis(array, vec_base):
    num_lines = len(vec_base[:, 0])  # Reads all lines

    check = 0

    for i in range(num_lines):
        lect = vec_base[i]

        if (array == lect).all():  # Reads a line
            check = 1

            break

    if check == 0:
        vec_base = np.insert(vec_base, len(vec_base), np.real(array), axis=0)

    return vec_base


# A modification of the last function which, instead of creating a vector base,
# used an already existent one for extracting which index within it corresponds to a
# chosen array. It is used in the system's first evolution method (main algorithm 2)
def photon_comb_index(array, vec_base):
    num_lines = len(vec_base[:, 0])  # Reads all lines

    index = 0

    for i in range(num_lines):
        lect = vec_base[i]

        if (array == lect).all():  # Reads a line
            break

        else:
            index += 1

    return index


# A more optimal and N-dimensional version of the former, allowing for
# probabilities (pamplitudes) per state considered in the linear combination
def state_in_basis(arrays, pamplitudes, vec_base):
    state = np.zeros(len(vec_base), dtype=complex)
    k = 0
    for element in arrays:
        # print(element, type(element))
        for ind in range(len(vec_base)):
            if (element == vec_base[ind]).all():
                basis_state = np.zeros_like(state)
                basis_state[ind] = pamplitudes[k]
                state = state + basis_state
        k = k + 1

    return state


# Extracts a particular subspace from the Fock states basis space
def subspace_basis(m, photons, subspace):
    vec_base_orig = photon_combs_generator(m, photons)
    vec_base = subspace
    for state in vec_base_orig:
        if not np.any([np.all(y) for y in [state == x for x in vec_base]]):
            vec_base.append(list(state))

    return np.array(vec_base)


def recur_factorial(n):
    if n == 1.0:
        return n

    elif n == 0.0:
        return 1.0

    elif n < 0.0:
        return "NA"

    else:
        return n * recur_factorial(n - 1)


# Factorial computation for all values of an array
def fact_array(array):
    array_2 = np.array([array])

    array_fact = np.apply_along_axis(recur_factorial, 0, array_2)

    return array_fact


# Combinatory computation (modes, photons)
def comb_evol(num_elements, num_dim):
    """
    num_elements=n, num_dim=m
    Computes the combinatory of (m+n-1,n). Variables given so the user only needs to know n and m.
    """

    sol = int(
        recur_factorial(num_elements + num_dim - 1)
        / (recur_factorial(num_elements) * recur_factorial(num_dim - 1))
    )

    return sol


# Combinatory computation
def comb_evol_no_reps(num_elements, num_dim):
    sol = int(
        recur_factorial(num_elements)
        / (recur_factorial(num_dim) * recur_factorial(num_elements - num_dim))
    )

    return sol


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
