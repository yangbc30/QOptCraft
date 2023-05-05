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
import scipy as sp

# ---------------------------------------------------------------------------------------------------------------------------
# 									INNER PRODUCT AND COMPARISON FUNCTIONS BETWEEN MATRICES
# ---------------------------------------------------------------------------------------------------------------------------


def mat_inner_product(U, V):
    matrix = U.conj().T @ V + V.conj().T @ U
    if isinstance(matrix, np.ndarray):
        return 0.5 * np.trace(matrix)
    if isinstance(matrix, sp.sparse.spmatrix):
        return 0.5 * np.trace(matrix.toarray())
    raise ValueError("Matrices should be either scipy sparse or numpy arrays.")


def mat_module(U):
    return np.sqrt(np.real(mat_inner_product(U, U)))


# The comparison() function asks the names in the input due to, by being obtained within the function,
# the matrices would always be referred to as 'U' and 'V'
def comparison(U, V, name1, name2, acc_d):
    print(f"\nWe will compare the matrices {name1} and {name2}:")

    prod = mat_inner_product(U - V, U - V)

    print(f"\nThe inner product of the rest between matrices is: {prod}")

    # Onscreen confirmation of equality between the matrices
    print(f"\nIs it aproximately equal to 0? {np.round(prod,acc_d)==0}")

    # 22 decimal accuracy upon computing the inner product, it can be modified
    return np.round(prod, acc_d) == 0


# We modify too the comparison() function so it doesn't print text onscreen.
# For multiple comparisons it is more convenient (see main algorithm 2a)
def comparison_noprint(U, V):
    prod = mat_inner_product(U - V, U - V)

    # 22 decimal accuracy upon computing the inner product, it can be modified
    return np.round(prod, 22) == 0


def comparison_noprint_3a(U, V):
    prod = mat_inner_product(U - V, U - V)

    # 22 decimal accuracy upon computing the inner product, it can be modified
    return prod
