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

import math

import numpy as np
import scipy as sp


def haar_measure(N):
    # https://arxiv.org/pdf/math-ph/0609050.pdf
    z = (sp.randn(N, N) + 1j * sp.randn(N, N)) / sp.sqrt(2.0)

    q, r = np.linalg.qr(z)  # QR factorization

    d = sp.diagonal(r)

    ph = d / sp.absolute(d)

    q = sp.multiply(q, ph, q)

    return q


def matrix_generation_general_auto(N1, N2):
    U = (sp.randn(N1, N2) + 1j * sp.randn(N1, N2)) / sp.sqrt(2.0)

    return U


# We create a modification of our function dft_matrix(), which doesn't ask for a N input but it is given as
# a parameter for the function instead. It is more convenient for loops running in an interval of dimensions,
# as well as not printing text onscreen for a better performance
def dft_matrix_auto(N):
    omega = np.exp(-2.0 * math.pi * 1j / float(N))

    A = np.zeros((N, N), dtype=complex)

    for i in range(N):
        for j in range(N):
            A[i, j] = omega ** (i * j)

    return A


def qft_matrix_auto(N):
    A = np.zeros((N, N), dtype=complex)

    for i in range(N):
        for j in range(N):
            A[i, j] = np.exp(-2.0 * math.pi * 1j / float(N) * i * j)

    return A / np.sqrt(N)
