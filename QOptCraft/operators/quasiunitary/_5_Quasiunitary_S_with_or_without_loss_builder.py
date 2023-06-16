# ---------------------------------------------------------------------------------------------------------------------------
# 						ALGORITHM 1a: NON-UNITARY MATRIX U GENERATOR. SCATTERING MATRIX S WITH LOSS
# ---------------------------------------------------------------------------------------------------------------------------

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

from QOptCraft._legacy.input_control import decimal_precision, input_control, input_control_ints
from QOptCraft.optic_decomposition.clemens_decomp import decomposition
from .quasiunitary import quasiunitary
from QOptCraft.optic_decomposition.recomposition import *
from ._5_D_decomposition import *
from ._5_matrix_padding_functions import *
from ._5_S_with_loss_creation import *
from QOptCraft._legacy.read_matrix import read_matrix_from_txt
from QOptCraft._legacy.unitary import *
from ..other_matrices import RandM


def QuasiU(
    file_input=True,
    T=False,
    file_output=True,
    filename=False,
    newfile=True,
    N1=False,
    N2=False,
    acc_anc=8,
    acc_d=3,
    txt=False,
):
    """
    Creates/loads .txt files containing a quasiunitary matrix and decomposes them into linear optics devices plus the remaining diagonal, taking additional loss modes into account.
    Information is displayed on-screen.
    """

    if txt is True:
        print("==========================================================")
        print("||| SCATTERING MATRIX S CREATOR WITH (OR WITHOUT) LOSS |||")
        print("==========================================================\n\n")

    # Input control: in case there is something wrong with given inputs, it is notified on-screen
    file_input, filename, newfile, acc_d = input_control(
        5, file_input, T, file_output, filename, True, acc_d, newfile
    )

    if newfile is True:
        N1 = input_control_ints(N1, "N1", 1)

        N2 = input_control_ints(N2, "N2", 1)

    if acc_anc < 0 or type(acc_anc) is not int:
        if acc_anc < 0:
            print(
                "\nWARNING: a value higher than 0 for acc_anc, the precision of finding ancilla modes, is required."
            )

        elif type(acc_anc) is not int:
            print(
                "\nWARNING: invalid acc_anc ancilla precision input (needs to be int and equal or higher than 0)."
            )

        # This variable corresponds to the ancilla decimal precision. It applies to the comparisons for finding photon loss
        # In the .txt files, results contain all possible decimals
        acc_anc = decimal_precision()

    # ----------INITIAL MATRIX INPUT:----------

    if txt is True:
        print("\nINITIAL MATRIX INPUT:\n")

        print(
            "\nThe algorithm requires a file 'U.txt' in this code's directory, containing the unitary/non-unitary matrix to decompose."
        )
        print(
            "\nFor the study of unitary matrices, the file must be generated in the main algorithm 1."
        )
        print(
            "\nIn the case of non-unitary matrices, this algorithm gives the option of creating a random one, overwriting the previous file."
        )
        print(
            "\nIf no new file is created and no other file is present, the code will end without computing.\n"
        )

    # Beginning of time measurement
    t = time.process_time_ns()

    # If a new matrix is created, there are two options: either just create an array, or save it into a .txt file
    if newfile is True:
        # A new file containing an N1xN2-dimensional unitary matrix U is created
        # so it can be used in other processes
        T = RandM(file_output, filename, N1, N2, txt)

    elif file_input is True:
        # Loading T
        T = read_matrix_from_txt(filename)

        # Input matrix T's dimensions N1 and N2, from its rows and columns respectively
        N1 = len(T[:, 0])

        N2 = len(T[0, :])

    if txt is True:
        print("\nInput matrix T:\n")

        print(np.round(T, acc_d))

        print(f"\nDimensions: {N1} x {N2}\n")

    # ----------SINGULAR VALUE DECOMPOSITION (SVD) AND MATRICES maxDim x maxDim:----------

    # We find the higher and lower dimension of the matrix (or minimum and maximum as there are only two)
    # They will be required in the algorithm
    np.min([N1, N2])
    maxDim = np.max([N1, N2])

    # An specific method of numpy.linalg (np.linalg in our case) is required for the singular value decomposition
    U, D, W = np.linalg.svd(T)

    if txt is True:
        print(
            "\n\nThe singular value decomposition of T in the matrices U, D, W has been computed.\n"
        )

    # We save the dimensions of the square matrices U y W, they will be required later
    OGdimU = len(U[:, 0])
    OGdimW = len(W[:, 0])

    # Initial decompositions of U y W, by using the method of the main algorithm 1
    UList, UD = decomposition(U, OGdimU, file_output, filename + "_U", txt)
    WList, WD = decomposition(W, OGdimW, file_output, filename + "_W", txt)

    # Matrix padding loop: depending of which has the lower dimensions, either U or W square matrices
    # will go from being minDim-dimensional to maxDim-dimensional

    if N1 < N2:
        U = matrix_padding(U, maxDim)

        # Variables: matrix M, its decomposition in matrices TmnList, its original dimensions
        # (OGdimM) and the max dimension present in T
        UList = matrix_padding_TmnList(U, UList, OGdimU, maxDim)

        UD = matrix_padding(UD, maxDim)

    elif N2 < N1:
        W = matrix_padding(W, maxDim)

        WList = matrix_padding_TmnList(W, WList, OGdimW, maxDim)

        WD = matrix_padding(WD, maxDim)

    # In D's case, we transfer the values of the 1D array given by svd() to an maxDim-dimensional square matrix
    D = SVD_diagonal_adjusting(D, maxDim)

    # D is decomposed in other diagonal matrices, corresponding to identity matrices D_i for which D_i[i,i] = D[i,i]
    DList = D_decomposition(D, maxDim, filename, file_output, txt)

    # ----------S MATRIX GENERATION:----------

    # 2N-dimensional, where N = nn + na.
    # In our case, nn equals maxDim, whereas na equals ancDim, the ancilla modes

    # Ancilla modes present: for each D[i,i]!=1, a new one is required
    ancDim = 0

    for i in range(0, maxDim):
        if np.round(D[i, i], acc_anc) != 1.0:
            ancDim += 1

    # Total dimensions for all quasiunitary S matrices
    totalDim = 2 * (maxDim + ancDim)

    if txt is True:
        print("\n\n\n\n\nS MATRIX GENERATION:\n")

        print(f"\nHighest dimension of the input matrix maxDim = {maxDim}")
        print(f"\nNumber of required ancilla modes ancDim = {ancDim}")
        print(f"\nTotal dimension of the resulting matrix totalDim = {totalDim}")

    # S matrices building: SU, SD and SW correspond to only the transformations of
    # U, W and D componentes respectively. By multiplying those three S is obtained

    SU = S_U_W_composition(U, maxDim, ancDim)

    SD = S_D_composition(DList, maxDim, ancDim, txt)

    SW = S_U_W_composition(W, maxDim, ancDim)

    # Dot product of matrices SU·SD·SW for S's computation:
    S = SU.dot(SD.dot(SW))

    # ----------STORAGE OF MATRICES SU, SD, SW, S IN FILES:----------

    if file_output is True:
        SU_file = open(filename + "_SU.txt", "w")

        np.savetxt(SU_file, SU, delimiter=",")

        SU_file.close()

        SD_file = open(filename + "_SD.txt", "w")

        np.savetxt(SD_file, SD, delimiter=",")

        SD_file.close()

        SW_file = open(filename + "_SW.txt", "w")

        np.savetxt(SW_file, SW, delimiter=",")

        SW_file.close()

        # Storage of the quasiunitary matrix S
        S_quasiunitary_matrix_file = open(filename + "_S_quasiunitary.txt", "w")

        np.savetxt(S_quasiunitary_matrix_file, S, delimiter=",")

        S_quasiunitary_matrix_file.close()

    # New matriz S: storages the first N dimensions of the previous matrix
    # Normally, this will be the useful matrix for the following algorithms

    S_cut = S[: int(totalDim / 2), : int(totalDim / 2)]

    if file_output is True:
        S_cut_file = open(filename + "_S.txt", "w")

        np.savetxt(S_cut_file, S_cut, delimiter=",")

        S_cut_file.close()

        if txt is True:
            print(
                "\nThe resulting quasiunitary matrix S has been storaged in the file '"
                + filename
                + "_S_quasiunitary.txt'."
            )
            print(
                "\nIts components SU, SD y SW have been storaged in the '"
                + filename
                + "_SU.txt', '"
                + filename
                + "_SW.txt' y '"
                + filename
                + "_SD.txt' files respectively."
            )
            print(
                f"\nFinally, a submatrix S corresponding to the first {maxDim+ancDim} rows and columns, compatible with the following algorithms, has been storaged in the file '"
                + filename
                + "_S.txt'."
            )

    if txt is True:
        print("\nThe S matrix with loss has been obtained.")

    # ----------MULTIPLE CHECKS FOR THE SU, SD, SW, S MATRICES OBTAINED:----------

    if txt is True:
        print("\n\nSU, SD, SW, S MATRICES OBTAINED:\n")

        quasiunitary(SU, totalDim, "SU", acc_d)

        quasiunitary(SD, totalDim, "SD", acc_d)

        quasiunitary(SW, totalDim, "SW", acc_d)

        print("\n\n\nTotal matrix S:\n")
        print(np.round(S, acc_d))

        print(f"\nS submatrix (first {maxDim+ancDim} rows and columns):\n")
        print(np.round(S_cut, acc_d))

    # ----------QUASIUNITARY CHECK FOR MATRIX S:----------

    if txt is True:
        print("\n\n\n\n\nQUASIUNITARY CHECK FOR MATRIX S:\n")

        cond = quasiunitary(S, totalDim, "S", acc_d)

        if cond is False:
            print(
                "\nThe algorithm found a problem with this result. It's advised to try running it again with a higher decimal accuracy for finding ancilla modes.\n"
            )

    # ----------UNITARY CHECK FOR SUBMATRIX S[:{int(totalDim/2)},:{int(totalDim/2)}]:----------

    if txt is True:
        print(f"\n\n\n\n\nUNITARY CHECK FOR SUBMATRIX S[:{int(totalDim/2)},:{int(totalDim/2)}]:\n")

        unitary(S_cut, int(totalDim / 2), "S_cut", acc_d)

    # ----------TOTAL TIME REQUIRED:----------

    t_inc = time.process_time_ns() - t

    print(f"\nPhase1a: total time of execution (seconds): {float(t_inc/(10**(9)))}\n")

    return T, S, S_cut, UList, UD, WList, WD, D, DList
