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

import time

import numpy as np

from QOptCraft.utils.input_control import input_control, input_control_intsDim

# Matrix comparisons by their inner product
from QOptCraft.legacy.mat_inner_product import comparison
from QOptCraft.optic_decomposition.clemens_decomp import decomposition
from QOptCraft.optic_decomposition.reck_decomp import decomposition_reck
from ..optic_decomposition.recomposition import recomposition, recomposition_reck
from QOptCraft.legacy.read_matrix import read_matrix_from_txt
from ._7_generators import RandU


def Selements(
    file_input=True,
    U_un=False,
    file_output=True,
    filename=False,
    impl=0,
    newfile=True,
    N=False,
    acc_d=3,
    txt=False,
):
    """
    Creates/loads .txt files containing an unitary matrix and decomposes them into linear
    optics devices plus the remaining diagonal. Information is displayed on-screen.
    """

    if txt is True:
        print("\n\n===================================================")
        print("||| UNITARY MATRIX U GENERATOR AND DECOMPOSITOR |||")
        print("===================================================\n\n")

    # Input control: in case there is something wrong with given inputs, it is notified on-screen
    file_input, filename, newfile, acc_d = input_control(
        1, file_input, U_un, file_output, filename, True, acc_d, newfile
    )

    if type(impl) is not int:
        print("\nWARNING: invalid impl input (needs to be int).")

        while True:
            try:
                impl = int(
                    input(
                        "\nInput '0' for Clements' implementation\nInput any other int number for Reck's\n"
                    )
                )

                break

            except ValueError:
                print("The given value is not valid.\n")

    if newfile is True:
        N = input_control_intsDim(N, "N", 2)

    # ----------INITIAL MATRIX INPUT:----------

    # Beginning of time measurement (in nanoseconds. The final result is also presented
    # in IS units of measure, in this case seconds)
    t = time.process_time_ns()

    # If a new matrix is created, there are two options: either just create an array,
    # or save it into a .txt file
    if newfile is True:
        # A new file 'U.txt' containing an N-dimensional unitary matrix U is created
        # so it can be used in other processes
        U_un = RandU(file_output, filename, N, txt)

    elif file_input is True:
        # Loading U_un (un = unitary, for distinguishing it from the evolution
        # matrix 'U' generated in the main algorithm 2)
        # NOTE: for all commentary in this code, U_un is referred as 'U' for simplicity
        U_un = read_matrix_from_txt(filename)

        N = len(U_un[0])

    if txt is True:
        print("\nINITIAL MATRIX INPUT:\n")

        # We print the input matrix U onscreen (with a decimal precision of 3)
        print("\nInput matrix U (" + filename + ".txt):\n")

        print(np.round(U_un, acc_d))

        print(f"\nDimensions: {N} x {N}\n")

    # ----------UNITARY CHECK FOR MATRIX U:----------

    if txt is True:
        print("\n\n\n\nUNITARY CHECK FOR MATRIX U:\n")
        np.testing.assert_allclose(np.eye(N), U_un @ U_un.T.conj())

    # ----------MATRIX U DECOMPOSITION:----------

    # We initialize an array of matrices Tmn (which represent optical devices)
    TmnList = np.zeros((int(N * (N - 1) / 2), N, N), dtype=complex)

    # No-null Tmn matrices and the resulting diagonal D (with its initial offsets) are obtained
    if impl == 0:
        TmnList, D = decomposition(U_un, N, file_output, filename, txt)

        # We try the recomposition algorithm with the matrix U, by using the
        # recently computed D and Tmn matrices
        U_init = recomposition(D, TmnList, N)

    else:
        TmnList, D = decomposition_reck(U_un, N, file_output, filename, txt)

        # We try the recomposition algorithm with the matrix U, by using the recently
        # computed D and Tmn matrices
        U_init = recomposition_reck(D, TmnList, N)

    # ----------U ONSCREEN PRINTING Y AND RECONSTRUCTION ALGORITHM CHECK:----------

    if txt is True:
        print("\n\n\n\n\nU RECONSTRUCTION CHECK:\n")

        print("\nRebuild of the initial U:\n")

        print(np.round(U_init, acc_d))

        print("\nIs it equal to the initial U?")

        comparison(U_init, U_un, "U_init", "U_un", acc_d)

    # ----------TOTAL TIME REQUIRED:----------

    t_inc = time.process_time_ns() - t

    print(f"\nSelements: total time of execution (seconds): {float(t_inc/(10**(9)))}\n")

    return U_un, TmnList, D
