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
import scipy as sp


# S_U and S_W matrices creation: we obtain four submatrices, then place them in diagonal
def S_U_W_composition(UW, maxDim, ancDim):
    2 * (maxDim + ancDim)

    I = np.identity(ancDim, dtype=complex)

    UW_conj = np.conj(UW)

    # U/W matrix + ancilla modes identity + U/W conjugate matrix + ancilla modes identity
    S_UW = sp.linalg.block_diag(UW, I, UW_conj, I)

    return S_UW


# This function computes the dot product between all S_D_i matrices,
# so the S_D matrix wanted is found
def S_D_composition(DList, maxDim, ancDim, txt=False):
    global index_na

    # This index is indicative of the correspondence between original and loss modes
    # It is stated here and will be manipulated in the creation of the multiple S_D_i matrices
    index_na = 0

    totalDim = 2 * (maxDim + ancDim)

    # SDList initialization
    SDList = np.zeros((len(DList[:, 0, 0]), totalDim, totalDim), dtype=complex)

    SD = np.identity(totalDim, dtype=complex)

    if ancDim != 0:
        for cont in range(0, len(DList[:, 0, 0])):
            SDList[cont, :, :] = S_D(cont, DList[cont, cont, cont], maxDim, ancDim, totalDim)

            SD = SD.dot(SDList[cont, :, :])

    else:
        if txt is True:
            print(
                "\nDue to the lack of required ancilla modes, the resulting S matrix will consist of only the input matrix and its conjugate in the diagonal.\n"
            )

    return SD


# S_D_i matrices creation: se apoya en el empleo de matrices identidad sobre las que se modifican valores
def S_D(index, car, maxDim, ancDim, totalDim):
    global S_D_index

    global index_na

    S_D_index = np.identity(totalDim, dtype=complex)

    # If car (D[i,i]) is not equal to 1.0, the totalDim-dimensional identity matrix is not enough: some elements need to be modified
    if np.round(car, 10) != 1.0:
        # In this auxiliar function, the modified elements of the matrix D_i have been organised
        S_D_values(index, car, maxDim, ancDim, totalDim)

        index_na += 1

    return S_D_index


# S_D_i creation has three versions: if car (D_i[i,i]) es always equal to 1, it is just the identity matrix
# On the contrary, loss elements are added if D_i[i,i] is above or below 1.0.
def S_D_values(index, car, maxDim, ancDim, totalDim):
    global S_D_index

    global index_na

    S_D_index[index, index] = car
    S_D_index[index_na + maxDim, index_na + maxDim] = car

    S_D_index[index + int(totalDim / 2), index + int(totalDim / 2)] = car
    S_D_index[index_na + maxDim + int(totalDim / 2), index_na + maxDim + int(totalDim / 2)] = car

    if car < 1:
        S_D_index[index, index_na + maxDim] = sp.sqrt(1 - car**2)
        S_D_index[index_na + maxDim, index] = -sp.sqrt(1 - car**2)

        S_D_index[index + int(totalDim / 2), index_na + maxDim + int(totalDim / 2)] = sp.sqrt(1 - car**2)
        S_D_index[index_na + maxDim + int(totalDim / 2), index + int(totalDim / 2)] = -sp.sqrt(1 - car**2)

    elif car > 1:
        S_D_index[int(totalDim / 2) + maxDim + index_na, index] = sp.sqrt(car**2 - 1)
        S_D_index[index, int(totalDim / 2) + maxDim + index_na] = sp.sqrt(car**2 - 1)

        S_D_index[int(totalDim / 2) + index, maxDim + index_na] = sp.sqrt(car**2 - 1)
        S_D_index[maxDim + index_na, int(totalDim / 2) + index] = sp.sqrt(car**2 - 1)
