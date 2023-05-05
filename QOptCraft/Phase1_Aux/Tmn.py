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

import cmath


# ---------------------------------------------------------------------------------------------------------------------------
# 												OPTIC DEVICE FUNCTIONS
# ---------------------------------------------------------------------------------------------------------------------------


def Tmn(theta, phi, N, m, n):
    T = np.identity(N, dtype=complex)  # First declaration of the matrix

    # We add the additional elements to our identity matrix in
    # the dimensions m, n (which correspond to the interaction)
    T[m, m] = cmath.exp(phi * 1j) * cmath.cos(theta)  # cos, sin in radians
    T[m, n] = -cmath.sin(theta)  # cos, sin in radians
    T[n, m] = cmath.exp(phi * 1j) * cmath.sin(theta)  # cos, sin in radians
    T[n, n] = cmath.cos(theta)  # cos, sin in radians

    return T


def TmnReck(theta, phi, N, m, n):
    T = np.identity(N, dtype=complex)  # First declaration of the matrix

    # We add the additional elements to our identity matrix in
    # the dimensions m, n (which correspond to the interaction)
    T[m, m] = cmath.exp(phi * 1j) * cmath.sin(theta)  # cos, sin in radians
    T[m, n] = cmath.exp(phi * 1j) * cmath.cos(theta)  # cos, sin in radians
    T[n, m] = cmath.cos(theta)  # cos, sin in radians
    T[n, n] = -cmath.sin(theta)  # cos, sin in radians

    return T
