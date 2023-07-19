from typing import Literal
from numbers import Number

import numpy as np
from numpy.typing import NDArray
import numba


def permanent(matrix: NDArray, method: Literal["glynn", "ryser"] = "glynn") -> Number:
    """Returns the permanent of a matrix using the Ryser formula in Gray ordering.

    Args:
        matrix (NDArray): A square matrix
        method (str): method to calculate the permanent. Options are 'glynn' and 'ryser'.
            Defaults to 'glynn'.

    Returns:
        Number: the permanent of the matrix
    """
    if method == "glynn":
        return _permanent_glynn(matrix)
    if method == "ryser":
        return _permanent_ryser(matrix)
    raise ValueError("Supported options for the permanent are 'glynn' and 'ryser'.")


@numba.jit(nopython=True)
def _permanent_glynn(matrix: NDArray) -> Number:
    """Returns the permanent of a matrix using the Ryser formula in Gray ordering.

    References:
        [1] The code is based off a Python 2 code from user 'xnor' found in
        https://codegolf.stackexchange.com/questions/97060/calculate-the-permanent-as-quickly-as-possible

        [2] Glynn algorithm for the permanent
        http://library.isical.ac.in:8080/jspui/bitstream/10263/3603/1/TH73.pdf

        [3] Gray code used in the algorithm https://en.wikipedia.org/wiki/Gray_code

        [4] This algorithm is also re-implemented in The-Walrus library
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


@numba.jit(nopython=True)
def _permanent_ryser(matrix: NDArray) -> Number:
    """Returns the permanent of a matrix using the Ryser formula in Gray ordering.

    References:
        [1] The code is based off a Python 2 code from user 'xnor' found in
        https://codegolf.stackexchange.com/questions/97060/calculate-the-permanent-as-quickly-as-possible

        [2] Ryser formula with Gray code:
        Nijenhuis, Albert; Wilf, Herbert S. (1978), Combinatorial Algorithms, Academic Press.

        [3] Ryser formula
        https://en.wikipedia.org/wiki/Computing_the_permanent#Ryser_formula

        [4] Gray code used in the algorithm
        https://en.wikipedia.org/wiki/Gray_code

        [5] This algorithm is also re-implemented in The-Walrus library
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
