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

import math

# NumPy instalation: in the cmd: 'py -m pip install numpy'
import numpy as np

# SciPy instalation: in the cmd: 'py -m pip install scipy'
import scipy as sp

# Matrix comparisons by their inner product
from ..mat_inner_product import *


# ---------------------------------------------------------------------------------------------------------------------------
# 										GRAM-SCHMIDT ORTOGONALISATION OF A BASIS
# ---------------------------------------------------------------------------------------------------------------------------


def gram_schmidt_2dmatrices(basis):
    number_elements = len(basis[:, 0, 0])
    N = len(basis[0, :, 0])

    orth_basis = np.zeros((number_elements, N, N), dtype=complex)

    for i in range(number_elements):
        orth_basis[i] = basis[i]
        for j in range(number_elements):
            if i > j:
                orth_basis[i] -= mat_inner_product(basis[i], orth_basis[j]) * orth_basis[j]
        orth_basis[i] /= np.sqrt(np.real(mat_inner_product(orth_basis[i], orth_basis[i])))

    return orth_basis


def gram_schmidt_modified_2dmatrices(basis: list[np.ndarray]) -> list[np.ndarray]:
    """It turns out that the Gram-Schmidt procedure we introduced previously suffers from numerical instability:
    Round-off errors can accumulate and destroy orthogonality of the resulting vectors.
    We introduce the modified Gram-Schmidt procedure to help remedy this issue.

    Algorithm can be found in https://www.math.uci.edu/~ttrogdon/105A/html/Lecture23.html
    """
    dim = len(basis)
    orth_basis = []

    for j in range(dim):
        orth_matrix = basis[j] / np.sqrt(np.real(mat_inner_product(basis[j], basis[j])))
        orth_basis.append(orth_matrix)
        for k in range(j + 1, dim):
            basis[k] = basis[k] - np.real(mat_inner_product(orth_basis[j], basis[k])) * orth_basis[j]

    return orth_basis
