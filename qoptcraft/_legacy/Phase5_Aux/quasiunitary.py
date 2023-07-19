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
from scipy.linalg import block_diag

from ..mat_inner_product import *


# ---------------------------------------------------------------------------------------------------------------------------
# 											QUASIUNITARY MATRIX CONDITION
# ---------------------------------------------------------------------------------------------------------------------------


# By undoing the commentary of line 44, the result of computing S_per_G_per_S+ is printed onscreen
# It must be equal to the G matrix if S is quasiunitary
def quasiunitary(S, totalDim, name, acc_d):
    print(
        f"\n\nIs {name} a quasiunitary matrix? We compute {name}·G·{name}+ ({name}+ is the adjoint operator matrix, or transpose conjugate, of {name}).\n"
    )

    # Computation of the matrix G: we declare its two blocks (totalDim/2-dimensional identities +1 and -1) then we join them
    I1 = np.identity(int(totalDim / 2), dtype=complex)

    I2 = -1 * np.identity(int(totalDim / 2), dtype=complex)

    G = block_diag(I1, I2)

    # We compute the result, knowing S and G
    S_per_G_per_S = S.dot(G.dot(np.transpose(np.conj(S))))

    # print(np.round(S_per_G_per_S,5))

    # We compare both matrices, for testing if S is quasiunitary
    print(f"\nIs the condition {name}·G·{name}+ = G met?\n")

    cond = comparison(S_per_G_per_S, G, "S_per_G_per_S", "G", acc_d)

    return cond
