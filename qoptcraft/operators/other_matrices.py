"""
--------------------
       LEGACY
--------------------
"""
import math

import numpy as np


def RotMat(N, offset):
    return np.eye(N, k=offset) + np.eye(N, k=-(N - offset))


# Creation of Quantum Fourier transformation matrices
def qft(dim):
    A = np.zeros((dim, dim), dtype=complex)
    omega = 2.0 * math.pi * 1j / dim
    for i in range(dim):
        for j in range(dim):
            A[i, j] = np.exp(omega * i * j)

    return A / np.sqrt(dim)


def qft_inv(dim):
    A = np.zeros((dim, dim), dtype=complex)

    for i in range(dim):
        for j in range(dim):
            A[i, j] = np.exp(-2.0 * math.pi * 1j / dim * i * j)

    return A / np.sqrt(dim)
