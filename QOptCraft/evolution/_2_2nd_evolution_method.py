import time

import numba
import numpy as np

from qoptcraft._legacy.recur_factorial import fact_array


@numba.jit(nopython=True)
def ryser_permanent(matrix: np.ndarray) -> float or complex:
    """
    Returns the permanent of a matrix using the Ryser formula in Gray ordering.

    Parameters
    ----------
    matrix : np.ndarray
        A square matrix

    Returns
    -------
    float or complex : the permanent of the matrix

    References
    ----------
    The code is based off a Python 2 code from user 'xnor' found in
    https://codegolf.stackexchange.com/questions/97060/calculate-the-permanent-as-quickly-as-possible

    Ryser formula with Gray code:
    Nijenhuis, Albert; Wilf, Herbert S. (1978), Combinatorial Algorithms, Academic Press.

    Ryser formula
    https://en.wikipedia.org/wiki/Computing_the_permanent#Ryser_formula

    Gray code used in the algorithm
    https://en.wikipedia.org/wiki/Gray_code

    This algorithm is also re-implemented in The-Walrus library
    https://github.com/XanaduAI/thewalrus/blob/master/thewalrus/_permanent.py
    """
    dim = len(matrix)
    if dim == 0:
        return matrix.dtype.type(1.0)
    # row_comb keeps the sum of previous subsets.
    row_comb = np.zeros((dim), dtype=matrix.dtype)

    permanent = 0
    old_grey = 0
    sign = +1

    binary_power_list = [2**i for i in range(dim)]
    num_loops = 2**dim

    for bin_index in range(1, num_loops):
        permanent += sign * np.prod(row_comb)

        new_grey = bin_index ^ (bin_index // 2)
        grey_diff = old_grey ^ new_grey
        grey_diff_index = binary_power_list.index(grey_diff)

        new_vector = matrix[grey_diff_index]
        direction = (old_grey > new_grey) - (old_grey < new_grey)  # same as cmp() in python 2

        for i in range(dim):
            row_comb[i] += new_vector[i] * direction

        sign = -sign
        old_grey = new_grey

    permanent += sign * np.prod(row_comb)

    return permanent * (-1) ** dim


@numba.jit(nopython=True)
def permanent(matrix: np.ndarray) -> float or complex:
    """
    Returns the permanent of a matrix using the Ryser formula in Gray ordering.

    Parameters
    ----------
    matrix : np.ndarray
        A square matrix

    Returns
    -------
    float or complex : the permanent of the matrix

    References
    ----------
    The code is based off a Python 2 code from user 'xnor' found in
    https://codegolf.stackexchange.com/questions/97060/calculate-the-permanent-as-quickly-as-possible

    Glynn algorithm for the permanent
    http://library.isical.ac.in:8080/jspui/bitstream/10263/3603/1/TH73.pdf

    Gray code used in the algorithm
    https://en.wikipedia.org/wiki/Gray_code

    This algorithm is also re-implemented in The-Walrus library
    https://github.com/XanaduAI/thewalrus/blob/master/thewalrus/_permanent.py
    """
    dim = len(matrix)
    if dim == 0:
        return matrix.dtype.type(1.0)
    # sum of each row of the matrix
    row_comb = np.sum(matrix, axis=0)

    permanent = 0
    old_grey = 0
    sign = +1

    binary_power_list = [2**i for i in range(dim)]
    num_loops = 2 ** (dim - 1)

    for bin_index in range(1, num_loops):
        permanent += sign * np.prod(row_comb)

        new_grey = bin_index ^ (bin_index // 2)
        grey_diff = old_grey ^ new_grey
        grey_diff_index = binary_power_list.index(grey_diff)

        new_vector = matrix[grey_diff_index]
        direction = 2 * (old_grey > new_grey) - 2 * (
            old_grey < new_grey
        )  # same as cmp() in python 2

        for i in range(dim):
            row_comb[i] += new_vector[i] * direction

        sign = -sign
        old_grey = new_grey

    permanent += sign * np.prod(row_comb)

    return (-1) ** dim * permanent / num_loops


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
            * ryser_permanent(sub_matrix(S, perm_1, perm_2))
            * complex(np.prod(fact_array(m_array))) ** (-1 / 2)
        )

    # Computation time
    t_inc = time.process_time_ns() - t

    return U_ket, t_inc
