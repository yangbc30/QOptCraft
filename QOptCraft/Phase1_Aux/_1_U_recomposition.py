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
# 											RECOMPOSITION OF MATRIX U FUNCTIONS
# ---------------------------------------------------------------------------------------------------------------------------


def U_recomposition(D, TmnList, N):
    # We explore TmnList in reverse order
    cont = int(N * (N - 1) / 2) - 1

    # The matrix given after the rebuild. We declare it equal to the diagonal D matrix as its initial value:
    U_init = D

    # We explore TmnList in reverse order as declared, computing
    # inverse operations with each matrix:
    for i in range(N - 1, 0, -1):
        # Case: even step i
        if i % 2 == 0:
            for j in range(1, i + 1):
                U_init = np.linalg.inv(TmnList[cont, :, :]).dot(U_init)

                cont -= 1

        # Case: odd step i
        else:
            for j in range(0, i):
                U_init = U_init.dot(TmnList[cont, :, :])

                cont -= 1

    return U_init
