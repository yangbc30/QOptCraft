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

import numpy as np
from scipy.linalg import block_diag


from qoptcraft.math import hs_scalar_product


def comparison(U, V, name1, name2, acc_d):
    print(f"\nWe will compare the matrices {name1} and {name2}:")

    prod = hs_scalar_product(U - V, U - V)

    print(f"\nThe inner product of the rest between matrices is: {prod}")

    # Onscreen confirmation of equality between the matrices
    print(f"\nIs it aproximately equal to 0? {np.round(prod,acc_d)==0}")

    # 22 decimal accuracy upon computing the inner product, it can be modified
    return np.round(prod, acc_d) == 0


# By undoing the commentary of line 44, the result of computing S_per_G_per_S+ is printed onscreen
# It must be equal to the G matrix if S is quasiunitary
def quasiunitary(S, totalDim, acc_d):
    I1 = np.identity(int(totalDim / 2), dtype=complex)
    I2 = -1 * np.identity(int(totalDim / 2), dtype=complex)
    G = block_diag(I1, I2)
    S_per_G_per_S = S.dot(G.dot(np.transpose(np.conj(S))))
    S_per_G_per_S = S @ G @ S.conj().T
    return comparison(S_per_G_per_S, G, "S_per_G_per_S", "G", acc_d)
