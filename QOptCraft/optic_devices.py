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

import cmath

# NumPy instalation: in the cmd: 'py -m pip install numpy'
import numpy as np
# SciPy instalation: in the cmd: 'py -m pip install scipy'
from scipy.linalg import block_diag

# ---------------------------------------------------------------------------------------------------------------------------
# 												OPTIC DEVICE FUNCTIONS
# ---------------------------------------------------------------------------------------------------------------------------


def split(phi, N, m, n):
    T = np.identity(N, dtype=complex)

    # We add the additional elements to our identity matrix in
    # the dimensions m, n (which correspond to the interaction)
    T[m, m] = cmath.cos(phi)  # cos, sin in radians
    T[m, n] = -cmath.sin(phi)  # cos, sin in radians
    T[n, m] = cmath.sin(phi)  # cos, sin in radians
    T[n, n] = cmath.cos(phi)  # cos, sin in radians

    split = block_diag(T, T)

    return split


def amp(phi, N, m, n):
    T = np.identity(N, dtype=complex)
    T2 = np.zeros((N, N), dtype=complex)

    # We add the additional elements to our identity matrix in
    # the dimensions m, n (which correspond to the interaction)
    T[m, m] = cmath.cosh(phi)  # cos, sin in radians
    T2[m, n] = cmath.sinh(phi)  # cos, sin in radians
    T2[n, m] = cmath.sinh(phi)  # cos, sin in radians
    T[n, n] = cmath.cosh(phi)  # cos, sin in radians

    amp = np.block([[T, T2], [T2, T]])

    return amp
