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
from scipy.linalg import logm

from ..Phase2_Aux._2_logarithm_algorithms import *


# Schur's third logarithm implementation.
def logm_3_schur(A):
    # Schur decomposition

    U, Q = schur(A)

    N = len(A)

    # D diagonal matrix initialization
    D = np.zeros((N, N), dtype=complex)

    # No-null values computation
    for i in range(N):
        D[i, i] = U[i, i] / abs(U[i, i])

    LD = logm(D)

    # Matrix logarithm computation
    logm_3A = Q.dot(LD.dot(np.transpose(np.conj(Q))))

    return logm_3A, Q, D


# Iterative algorithm for the logarithm of an unitary matrix's computation
def LogU(U, iterations):
    # from the paper
    V = 0.5 * (np.matrix(U) + np.linalg.inv(np.matrix(U)).H)  # Initial V

    for _i in range(iterations):
        V = 0.5 * (V + np.linalg.inv(V).H)

    T, Q = sp.linalg.schur(V)

    D = np.diag([T[i][i] / np.absolute(T[i][i]) for i in range(len(U))])

    HU = np.matmul(np.matmul(Q, logm(D)), np.matrix(Q).H)

    # here U^\dag=UT =-U
    return HU
