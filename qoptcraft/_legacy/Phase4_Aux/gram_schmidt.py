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

# Matrix comparisons by their inner product
from ..mat_inner_product import *


# ---------------------------------------------------------------------------------------------------------------------------
# 										GRAM-SCHMIDT ORTOGONALISATION OF A BASIS
# ---------------------------------------------------------------------------------------------------------------------------


def gram_schmidt_2dmatrices(basis):
    number_elements = len(basis[:, 0, 0])
    N = len(basis[0, :, 0])

    orth_basis = np.zeros((number_elements, N, N), dtype=complex)

    """
	for i in range(number_elements):

		for j in range(number_elements):

			print(f"Inner product between basis[{i}] y basis[{j}]: {mat_inner_product(basis[i],basis[j])}")
	"""

    for i in range(number_elements):
        orth_basis[i] = basis[i]

        for j in range(number_elements):
            if i > j:
                orth_basis[i] -= mat_inner_product(basis[i], orth_basis[j]) * orth_basis[j]

        orth_basis[i] /= np.sqrt(np.real(mat_inner_product(orth_basis[i], orth_basis[i])))

    """
	for i in range(number_elements):

		for j in range(number_elements):

			print(f"Inner product between orth_basis[{i}] y orth_basis[{j}]: {mat_inner_product(orth_basis[i],orth_basis[j])}")
	"""

    return orth_basis
