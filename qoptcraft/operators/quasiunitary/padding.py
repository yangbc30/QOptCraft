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


# Extends the size of an square matrix to the maximum dimension maxDim given
def matrix_padding(M, maxDim):
    # Array to be added as column
    column_to_be_added = np.zeros(len(M), dtype=complex)

    # Adding column to numpy array
    for _i in range(0, maxDim - len(M)):
        M = np.column_stack((M, column_to_be_added))

    for i in range(len(M), maxDim):
        # Array to be added as row
        row_to_be_added = np.zeros(maxDim, dtype=complex)
        row_to_be_added[i] = 1.0

        M = np.vstack((M, row_to_be_added))

    return M


# Akin to the previous functions, but extends the size of a list of matrices
def matrix_padding_TmnList(M, TmnList, OGdimM, maxDim):
    TmnListExp = np.zeros((int(OGdimM * (OGdimM - 1) / 2), maxDim, maxDim), dtype=complex)

    for cont in range(0, int(OGdimM * (OGdimM - 1) / 2)):
        TmnListExp[cont, :, :] = matrix_padding(TmnList[cont, :, :], maxDim)

    return TmnListExp


# Creates the matrix version of the 1D-array containing D's diagonal values
def SVD_diagonal_adjusting(M, maxDim):
    D = np.zeros((maxDim, maxDim), dtype=complex)

    for i in range(0, len(M)):
        D[i, i] = M[i]

    for i in range(len(M), maxDim):
        D[i, i] = 1.0

    return D
