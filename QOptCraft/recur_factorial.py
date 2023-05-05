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


# ---------------------------------------------------------------------------------------------------------------------------
# 											FACTORIAL COMPUTATION FUNCTIONS
# ---------------------------------------------------------------------------------------------------------------------------


# Factorial of a natural number computation
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
        recur_factorial(num_elements + num_dim - 1) / (recur_factorial(num_elements) * recur_factorial(num_dim - 1))
    )

    return sol


# Combinatory computation
def comb_evol_no_reps(num_elements, num_dim):
    sol = int(recur_factorial(num_elements) / (recur_factorial(num_dim) * recur_factorial(num_elements - num_dim)))

    return sol
