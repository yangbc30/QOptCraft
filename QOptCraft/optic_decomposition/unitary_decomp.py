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

from QOptCraft.optic_decomposition.clemens_decomp import decomposition
from QOptCraft.optic_decomposition.reck_decomp import decomposition_reck


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

    TmnList = np.zeros((int(N * (N - 1) / 2), N, N), dtype=complex)

    # No-null Tmn matrices and the resulting diagonal D (with its initial offsets) are obtained
    if impl == 0:
        TmnList, D = decomposition(U_un, N, file_output, filename, txt)

    else:
        TmnList, D = decomposition_reck(U_un, N, file_output, filename, txt)

    return U_un, TmnList, D
