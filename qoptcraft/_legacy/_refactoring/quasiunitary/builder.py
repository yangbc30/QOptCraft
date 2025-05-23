"""NON-UNITARY MATRIX U GENERATOR. SCATTERING MATRIX S WITH LOSS"""

import numpy as np

from qoptcraft.optical_elements import clemens_decomp
from .quasiunitary import quasiunitary
from .padding import *
from .with_loss import *


def SVD_diagonal_adjusting(M, maxDim):
    return np.append((M, [1] * (maxDim - len(M))))


def D_decomposition(M, dim):
    DList = np.zeros((dim, dim, dim), dtype=complex)

    for i in range(0, dim):
        I = np.identity(dim, dtype=complex)
        # Matrix D_i creation consists on replacing the identity
        # matrix element [i,i] for D_i of the original matrix D (here, M)
        I[i, i] = M[i, i]
        DList[i, :, :] = I
    return DList


def quasiunitary(S, totalDim, acc_d):
    I1 = np.identity(int(totalDim / 2), dtype=complex)
    I2 = -1 * np.identity(int(totalDim / 2), dtype=complex)
    G = block_diag(I1, I2)
    S_per_G_per_S = S.dot(G.dot(np.transpose(np.conj(S))))
    S_per_G_per_S = S @ G @ S.conj().T
    return comparison(S_per_G_per_S, G, "S_per_G_per_S", "G", acc_d)


def QuasiU(
    T=False,
    file_output=True,
    filename=False,
    N1=False,
    N2=False,
    acc_anc=8,
    txt=False,
):
    """Creates/loads .txt files containing a quasiunitary matrix and decomposes
    them into linear optics devices plus the remaining diagonal, taking additional
    loss modes into account.
    """
    np.min([N1, N2])
    maxDim = np.max([N1, N2])

    # An specific method of numpy.linalg (np.linalg in our case) is required for the singular value clemens_decomp
    U, D, W = np.linalg.svd(T)

    # We save the dimensions of the square matrices U y W, they will be required later
    OGdimU = len(U[:, 0])
    OGdimW = len(W[:, 0])

    # Initial clemens_decomps of U y W, by using the method of the main algorithm 1
    UList, UD = clemens_decomp(U, OGdimU, file_output, filename + "_U", txt)
    WList, WD = clemens_decomp(W, OGdimW, file_output, filename + "_W", txt)

    # Matrix padding loop: depending of which has the lower dimensions, either U or W square matrices
    # will go from being minDim-dimensional to maxDim-dimensional

    if N1 < N2:
        U = matrix_padding(U, maxDim)

        # Variables: matrix M, its clemens_decomp in matrices TmnList, its original dimensions
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
    DList = D_clemens_decomp(D, maxDim, filename, file_output, txt)

    ancDim = 0

    for i in range(0, maxDim):
        if np.round(D[i, i], acc_anc) != 1.0:
            ancDim += 1

    # Total dimensions for all quasiunitary S matrices
    totalDim = 2 * (maxDim + ancDim)

    SU = S_U_W_composition(U, maxDim, ancDim)
    SD = S_D_composition(DList, maxDim, ancDim, txt)
    SW = S_U_W_composition(W, maxDim, ancDim)

    # Dot product of matrices SU·SD·SW for S's computation:
    S = SU.dot(SD.dot(SW))

    S_cut = S[: int(totalDim / 2), : int(totalDim / 2)]

    return T, S, S_cut, UList, UD, WList, WD, D, DList
