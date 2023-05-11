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


# Main function to inherit in other algorithms
def photon_combs_generator(m: int, photons: list):
    """Given a Fock state with a certain number of modes, generate
    the basis for the Hilbert space.

    Args:
        m (int): number of optical modes.
        photons (list): a Fock state.

    Returns:
        _type_: basis of the Hilbert space.
    """

    global photons_aux
    global vec_base
    global check

    check = 0

    num_photons = int(np.sum(photons))

    # counter = np.array(photons[:], dtype=int)

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
                for _q in range(counter[p]):
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
def state_in_basis(basis, pamplitudes, vec_base) -> np.ndarray:
    """Given a vector in terms of elements of a basis and amplitudes, output the state vector."""
    state = np.zeros(len(vec_base), dtype=complex)
    for k, vector in enumerate(basis):
        # print(vector, type(vector))
        for ind in range(len(vec_base)):
            if (vector == vec_base[ind]).all():
                basis_state = np.zeros_like(state)
                basis_state[ind] = pamplitudes[k]
                state = state + basis_state
    return state


# Extracts a particular subspace from the Fock states basis space
def subspace_basis(m, photons, subspace):
    vec_base_orig = photon_combs_generator(m, photons)
    vec_base = subspace
    for state in vec_base_orig:
        if not np.any([np.all(y) for y in [state == x for x in vec_base]]):
            vec_base.append(list(state))

    return np.array(vec_base)
