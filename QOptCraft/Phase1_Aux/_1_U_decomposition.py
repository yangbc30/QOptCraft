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

import warnings

import numpy as np
from scipy.optimize import fsolve

from .Tmn import *


# The functions evenFunction(z) and oddFunction(z) find the required values of theta and phi
# for each decomposition, by using two conditions: the real and imaginary values
def evenFunction(z):
    global M_current
    global N

    # Theta and phi are indexed for each iteration...
    theta = z[0]
    phi = z[1]

    T = Tmn(theta, phi, N, m - 1, n - 1)

    # ...until the desired results are written in f0 y f1 (ehich nullify
    # the real and imaginary values)
    f0 = np.real(T[N + j - i - 1, :].dot(M_current[:, j - 1]))
    f1 = np.imag(T[N + j - i - 1, :].dot(M_current[:, j - 1]))

    return np.array([f0, f1])


def oddFunction(z):
    global M_current
    global N

    theta = z[0]
    phi = z[1]

    T = Tmn(theta, phi, N, m - 1, n - 1)

    # This time, we apply the dot product in the other side of the matrix
    f0 = np.real(M_current[N - j - 1, :].dot(np.linalg.inv(T)[:, i - j - 1]))
    f1 = np.imag(M_current[N - j - 1, :].dot(np.linalg.inv(T)[:, i - j - 1]))

    return np.array([f0, f1])


def phase_adjust(phase, minn=-np.pi, maxx=np.pi):
    while phase < minn:
        phase += 2.0 * np.pi

    while phase > maxx:
        phase -= 2.0 * np.pi

    return phase


def U_decomposition(M, dim, output, name, txt=False):
    # We declare those variables as global in orden to use them in odd/evenFunction(z)
    global m
    global n
    global j
    global i

    # Easier to comprehend dimension and matrix declarations
    global N
    N = dim
    global M_current
    M_current = M

    if output is True:
        # Decomposition of U storage
        fullnameTmn = name + "_TmnList.txt"
        TmnList_file = open(fullnameTmn, "w")

        fullnameD = name + "_D.txt"
        D_file = open(fullnameD, "w")

    # Steps counter
    cont = 0

    TmnList = np.zeros((int(N * (N - 1) / 2), N, N), dtype=complex)

    for i in range(1, N):
        if i % 2 == 0:  # even
            for j in range(1, i + 1):
                m = N + j - i - 1
                n = N + j - i

                # We need to find theta and phi before anything else,
                # but how?

                # We have two given conditions: both real and imaginary
                # values must be null simultaneously for each non-diagonal index of the
                # final matrix

                # Initial values of theta and phi given for iteration solving
                zGuess = np.ones(2)

                with warnings.catch_warnings():
                    # In the main algorithm 1a, warnings regarding a lack of advancement
                    # can appear with permutations. However, given results rebuild
                    # the initial matrix perfectly, thus irrelevance is assumed
                    warnings.simplefilter("ignore")

                    sol = fsolve(evenFunction, zGuess)

                M_current = Tmn(sol[0], sol[1], N, m - 1, n - 1).dot(M_current)

                TmnList[cont, :, :] = Tmn(sol[0], sol[1], N, m - 1, n - 1)

                if output is True:
                    theta = phase_adjust(sol[0], minn=-np.pi, maxx=np.pi)
                    phi = phase_adjust(sol[1], minn=0.0, maxx=2.0 * np.pi)
                    TmnList_file.write(
                        "\nMatriz " + name + f"_T{m}{n} (even): Theta = {theta}, Phi = {phi}\n"
                    )

                    np.savetxt(TmnList_file, TmnList[cont, :, :], delimiter=",")

                cont += 1

        else:  # odd
            for j in range(0, i):
                m = i - j
                n = i - j + 1

                zGuess = np.ones(2)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    sol = fsolve(oddFunction, zGuess)

                # The computation changes compared to the even case
                M_current = M_current.dot(np.linalg.inv(Tmn(sol[0], sol[1], N, m - 1, n - 1)))

                TmnList[cont, :, :] = Tmn(sol[0], sol[1], N, m - 1, n - 1)

                if output is True:
                    theta = phase_adjust(sol[0], minn=-np.pi, maxx=np.pi)
                    phi = phase_adjust(sol[1], minn=0.0, maxx=2.0 * np.pi)
                    TmnList_file.write(
                        "\nMatriz " + name + f"_T{m}{n} (odd): Theta = {theta}, Phi = {phi}\n"
                    )

                    np.savetxt(TmnList_file, TmnList[cont, :, :], delimiter=",")

                cont += 1

    if output is True:
        D_file.write("Matrix " + name + "_D:\n")

        np.savetxt(D_file, np.round(M_current, 5), delimiter=",")

        TmnList_file.close()

        D_file.close()

        if txt is True:
            print(
                f"\nThe Tmn matrices (each with their theta and phi values) have been storaged in the file '{fullnameTmn}'."
            )
            print(
                f"\nThe diagonal matrix D resulting from the decomposition process has been storaged in the file '{fullnameD}'."
            )

    if txt is True:
        print(f"\nThe {name} matrix has been decomposed in optic devices.")

    return TmnList, M_current


# The following function is not used in the actual code. It has been left regardless in case the user finds it useful
def TmnListInverse(TmnList):
    N = len(TmnList[0, 0, :])

    l = len(TmnList[:, 0, 0])

    TmnList_inv = np.zeros((l, N, N), dtype=complex)

    # We explore TmnList in reverse order as declared, computing
    # inverse operations with each matrix:
    for i in range(l):
        # In unitary matrices, the transpose matrix is identical to the inverse. Thus,
        # np.transpose is used for better performance
        TmnList_inv[i, :, :] = np.transpose(TmnList[i, :, :])

    return TmnList_inv
