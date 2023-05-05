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


# ----------TIME OF EXECUTION MEASUREMENT:----------

import time


# ----------MATHEMATICAL FUNCTIONS/MATRIX OPERATIONS:----------

# NumPy instalation: in the cmd: 'py -m pip install numpy'
import numpy as np

from numpy.linalg import eig, det, inv

# SciPy instalation: in the cmd: 'py -m pip install scipy'
from scipy.linalg import schur, logm, sqrtm

from sympy import *

# Matrix comparisons by their inner product
from ..mat_inner_product import comparison_noprint


# ---------------------------------------------------------------------------------------------------------------------------
# 											LOGARITHM OF A MATRIX FUNCTIONS
# ---------------------------------------------------------------------------------------------------------------------------


def logm_1(A):
    # Beginning of time measurement
    t = time.process_time_ns()

    # Schur decomposition
    W, T = Matrix(A).diagonalize()
    W = np.array(W, dtype=complex)
    T = np.array(T, dtype=complex)

    N = len(A)

    # D diagonal matrix initialization
    D = np.zeros((N, N), dtype=complex)

    # No-null values computation
    for i in range(N):
        D[i, i] = T[i, i] / abs(T[i, i])

    # Matrix logarithm computation
    H = W.dot(logm(D).dot(np.linalg.inv(W)))
    logm_1A = 0.5 * (H + np.transpose(np.conj(H)))

    # Total time of execution
    t_inc = time.process_time_ns() - t

    return logm_1A, t_inc


def logm_2(A):
    # Beginning of time measurement
    t = time.process_time_ns()

    # Matrix logarithm computation
    logm_2A = 0.5 * (logm(A) + np.transpose(np.conj(logm(A))))

    # Total time of execution
    t_inc = time.process_time_ns() - t

    return logm_2A, t_inc


def logm_3(A):
    # Beginning of time measurement
    t = time.process_time_ns()

    # Schur decomposition
    U, Q = schur(A)

    N = len(A)

    # D diagonal matrix initialization
    D = np.zeros((N, N), dtype=complex)

    # No-null values computation
    for i in range(N):
        D[i, i] = U[i, i] / abs(U[i, i])

    # Matrix logarithm computation
    logm_3A = Q.dot(logm(D).dot(np.transpose(np.conj(Q))))

    # Total time of execution
    t_inc = time.process_time_ns() - t

    return logm_3A, t_inc


def logm_4(A):
    # Beginning of time measurement
    t = time.process_time_ns()

    # Defining formula of this logarithm algorithm
    V = A.dot(inv(sqrtm(np.transpose(np.conj(A)).dot(A))))

    # Schur decomposition
    U, Q = schur(V)

    N = len(A)

    # D diagonal matrix initialization
    D = np.zeros((N, N), dtype=complex)

    # No-null values computation
    for i in range(N):
        D[i, i] = U[i, i] / abs(U[i, i])

    # Matrix logarithm computation
    logm_4A = Q.dot(logm(D).dot(np.transpose(np.conj(Q))))

    # Total time of execution
    t_inc = time.process_time_ns() - t

    return logm_4A, t_inc


def logm_5(A):
    # Beginning of time measurement
    t = time.process_time_ns()

    # Defining formula of this logarithm algorithm
    V1 = (A + np.transpose(np.conj(inv(A)))) / 2.0

    V = (V1 + np.transpose(np.conj(inv(V1)))) / 2.0

    # Schur decomposition
    U, Q = schur(V)

    N = len(A)

    # D diagonal matrix initialization
    D = np.zeros((N, N), dtype=complex)

    # No-null values computation
    for i in range(N):
        D[i, i] = U[i, i] / abs(U[i, i])

    # Matrix logarithm computation
    logm_5A = Q.dot(logm(D).dot(np.transpose(np.conj(Q))))

    # Total time of execution
    t_inc = time.process_time_ns() - t

    return logm_5A, t_inc
