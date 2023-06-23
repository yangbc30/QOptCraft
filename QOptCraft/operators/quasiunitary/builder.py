"""NON-UNITARY MATRIX U GENERATOR. SCATTERING MATRIX S WITH LOSS"""

import numpy as np

from qoptcraft.optic_decomposition.clemens_decomp import decomposition
from .quasiunitary import quasiunitary
from qoptcraft.optic_decomposition.recomposition import *
from .diagonal_decomp import *
from .padding import *
from .with_loss import *
from qoptcraft._legacy.unitary import *


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

    # An specific method of numpy.linalg (np.linalg in our case) is required for the singular value decomposition
    U, D, W = np.linalg.svd(T)

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
