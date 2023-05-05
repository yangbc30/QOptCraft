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
# 							PERMUTATION MATRIX ASSOCIATED TO A REARRANGEMENT OF VALUES
# ---------------------------------------------------------------------------------------------------------------------------


def permutation_matrix(perm_list):
    N = len(perm_list)

    I = np.identity(N, dtype=int)

    M = np.identity(N, dtype=int)

    for i in range(N):
        M[i] = I[perm_list[i]]

    return M
