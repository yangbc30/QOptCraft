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
# 											CREATION AND ANNIHILATION OPERATORS
# ---------------------------------------------------------------------------------------------------------------------------


# We define the operators a (annihilation) and a_dagger (creator):
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
