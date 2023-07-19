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

from .mat_inner_product import *


# ---------------------------------------------------------------------------------------------------------------------------
# 												UNITARY MATRIX CONDITION
# ---------------------------------------------------------------------------------------------------------------------------


# By undoing the commentary of line 37, the result of computing Uconj_per_U+ is printed onscreen
# It must be equal to the identity matrix if U is unitary
def unitary(U, N, name, acc_d):
    print(f"\n\nIs {name} truly an unitary matrix?")

    print(f"We compute {name}+·{name}.\n")

    Uconj_per_U = np.transpose(np.conj(U)).dot(U)

    # print(np.round(Uconj_per_U))

    # len(U[0]) gives the length of the U row. Due to U being a N-dimensional square matrix, it corresponds to the columns too
    I = np.identity(N, dtype=complex)

    # We compare both matrices, for testing if U is unitary
    print(f"\nIs the condition {name}+·{name} = I met?\n")

    cond = comparison(Uconj_per_U, I, "Uconj_per_U", "I", acc_d)

    return cond
