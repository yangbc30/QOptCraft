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


# ----------TIME OF EXECUTION MEASUREMENT:----------

import time


# ----------MATHEMATICAL FUNCTIONS/MATRIX OPERATIONS:----------

# NumPy instalation: in the cmd: 'py -m pip install numpy'
import numpy as np


# ----------FILE MANAGEMENT:----------

# File opening


# ----------COMBINATORY:----------

from ..recur_factorial import fact_array

from .ryser_permanent import ryser_loop


# ----------PERMUTATIONS:----------

from itertools import permutations


# ---------------------------------------------------------------------------------------------------------------------------
# 										N-PHOTON OPTICAL SYSTEM EVOLUTION: SECOND METHOD
# ---------------------------------------------------------------------------------------------------------------------------


# Permanent computation
def permanent(M):
    # We assume square matrix
    N = len(M)
    # Permutations array
    perm_iterator = permutations(range(N))
    suma = 0.0

    for item in perm_iterator:
        # We initialise the product for each addend
        product = 1.0
        # Product computation
        for j in range(N):
            if N == 1:
                product *= M[j]
            else:
                product *= M[j, np.asarray(item)[j] - 1]
        suma += product

    return suma


# Permanent computation
def permanent_ryser(M):
    # We assume square matrix
    N = len(M)
    # Sum initialisation
    suma = 0.0

    for S in range(N):
        suma1 = 0.0

        # Combs
        combs = ryser_loop(N, S)
        for i in range(len(combs)):
            # We initialise the product for each addend
            product = 1.0
            # Product computation
            for j in range(N):
                # Sum initialisation (2)
                suma2 = 0.0
                for k in range(N - S):  # compute columns
                    suma2 += M[j, combs[i, k]]
                product *= suma2
            suma1 += product
        suma1 *= np.power(-1, np.abs(S))
        suma += suma1

    return np.power(-1, N) * suma


# Submatrices
def sub_matrix(M, perm1, perm2):
    N = len(perm2)

    M_sub = np.zeros((N, N), dtype=complex)

    for i in range(N):
        for j in range(N):
            M_sub[i, j] = M[perm1[i], perm2[j]]

    return M_sub


# Multiplicity m() (another way of finding the number of photons for each mode)
def m_(array, N):
    num_photons = len(array)

    m_array = np.zeros(N, dtype=int)

    cont = 0

    for i in range(N):
        suma = 0

        # We explore the array, each time comparing it with a different value
        for j in range(num_photons):
            if array[j] == cont:
                suma += 1

        m_array[i] = suma

        cont += 1

    return m_array


# Last function's inverse. NOTE: the elements's order may differ from that of the original array,
# through it is not a problem in this algorithm, as both are perceived as identical
def m_inverse(array):
    num_photons = int(np.sum(np.real(array)))

    N = len(array)

    m_array_inv = np.zeros(num_photons, dtype=int)

    cont = 0

    for i in range(N):
        # We explore the array, each time comparing it with a different value
        for _j in range(int(np.real(array)[i])):
            m_array_inv[cont] = i

            cont += 1

    return m_array_inv


# Here, we will perform the second evolution of the system method's operations. Main function to inherit in other algorithms
def evolution_2(S, photons, vec_base):
    # Initial time
    t = time.process_time_ns()

    m = len(S)
    int(np.sum(photons))

    # Number of vectors in the basis
    num_lines = len(vec_base[:, 0])

    # Array 2 required for submatrices computations:
    perm_2 = m_inverse(photons)

    # All terms will begin multiplied by this factor
    mult = complex(np.prod(fact_array(photons))) ** (-1 / 2)

    # Here each basis vector's coeficients upon |ket> are storaged:
    U_ket = np.zeros(num_lines, dtype=complex)

    for i in range(num_lines):
        # Array 1 required for submatrices computations:
        perm_1 = m_inverse(vec_base[i])

        m_array = m_(perm_1, m)

        # U·|ket> coeficients' computation by using permaments
        U_ket[i] = (
            mult
            * permanent(sub_matrix(S, perm_1, perm_2))
            * complex(np.prod(fact_array(m_array))) ** (-1 / 2)
        )

    # Computation time
    t_inc = time.process_time_ns() - t

    return U_ket, t_inc


# Here, we will perform the second evolution of the system method's operations. Main function to inherit in other algorithms
def evolution_2_ryser(S, photons, vec_base):
    # Initial time
    t = time.process_time_ns()

    m = len(S)
    int(np.sum(photons))

    # Number of vectors in the basis
    num_lines = len(vec_base[:, 0])

    # Array 2 required for submatrices computations:
    perm_2 = m_inverse(photons)

    # All terms will begin multiplied by this factor
    mult = complex(np.prod(fact_array(photons))) ** (-1 / 2)

    # Here each basis vector's coeficients upon |ket> are storaged:
    U_ket = np.zeros(num_lines, dtype=complex)

    for i in range(num_lines):
        # Array 1 required for submatrices computations:
        perm_1 = m_inverse(vec_base[i])

        m_array = m_(perm_1, m)

        # U·|ket> coeficients' computation by using permaments
        U_ket[i] = (
            mult
            * permanent_ryser(sub_matrix(S, perm_1, perm_2))
            * complex(np.prod(fact_array(m_array))) ** (-1 / 2)
        )

    # Computation time
    t_inc = time.process_time_ns() - t

    return U_ket, t_inc
