import math

import numpy as np
import scipy as sp


def RotMat(N, offset):
    return np.eye(N, k=offset) + np.eye(N, k=-(N - offset))


# This function below generates random non-unitary matrices. It is useful in the design of
# quasiunitary S matrices with loss
def RandM(N1, N2):
    return (sp.randn(N1, N2) + 1j * sp.randn(N1, N2)) / sp.sqrt(2.0)


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
