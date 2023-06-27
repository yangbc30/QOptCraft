import numpy as np

from qoptcraft._legacy.recur_factorial import fact_array
from qoptcraft.math import permanent, permanent_ryser


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
    m = len(S)
    int(np.sum(photons))

    # Number of vectors in the basis
    num_lines = len(vec_base)

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

    return U_ket


# Here, we will perform the second evolution of the system method's operations. Main function to inherit in other algorithms
def evolution_2_ryser(S, photons, vec_base):
    # Initial time
    m = len(S)
    int(np.sum(photons))

    # Number of vectors in the basis
    num_lines = len(vec_base)

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

    return U_ket
