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

# ---------------------------------------------------------------------------------------------------------------------------
# 													LIBRARIES REQUIRED
# ---------------------------------------------------------------------------------------------------------------------------


# ----------TIME OF EXECUTION MEASUREMENT:----------

import time


# ----------MATHEMATICAL FUNCTIONS/MATRIX OPERATIONS:----------

# NumPy instalation: in the cmd: 'py -m pip install numpy'
import numpy as np

# SciPy instalation: in the cmd: 'py -m pip install scipy'
import scipy as sp


# ----------FILE MANAGEMENT:----------

# File opening
from io import open


# ----------COMBINATORY:----------

from ..recur_factorial import *


# ----------ALGORITHM 2: AUXILIAR FUNCTIONS:----------

from ._2_creation_and_destruction_operators import a_dagger


# ----------PHOTON COMB BASIS:----------

from ..photon_comb_basis import photon_comb_index


# ---------------------------------------------------------------------------------------------------------------------------
# 										N-PHOTON OPTICAL SYSTEM EVOLUTION: FIRST METHOD
# ---------------------------------------------------------------------------------------------------------------------------


# Here, we will perform the first evolution of the system method's operations. Main function to inherit in other algorithms
def evolution(S, photons, vec_base_aux):
    # Initial time
    t = time.process_time_ns()

    global vec_base

    vec_base = vec_base_aux

    m = len(S)
    num_photons = int(np.sum(photons))

    # It is required to introduce photons_aux for 'photons_aux' and 'photons' not to "update" together
    global photons_aux
    global mult

    # U·|ket> to export
    global U_ket

    # U_ket initialization
    U_ket = np.zeros(comb_evol(num_photons, m), dtype=complex)

    # Vectors from the basis which will appear in the operations
    sum_ = np.zeros(num_photons, dtype=int)

    # The last two terms are required because of the function's recursive character
    evolution_loop(S, photons, num_photons, m, sum_, 0)

    # Computation time
    t_inc = time.process_time_ns() - t

    return U_ket, t_inc


# Loop whose amount of callings depend on the number of photons in each mode
def evolution_loop(S, photons, num_photons, m, sum_, k):
    # Variables to share with evolution() and successive callings of
    # evolution_loop()
    global U_ket
    global photons_aux
    global mult
    global vec_base

    counter = np.array(photons[:], dtype=int)

    for sum_[k] in range(m):
        if k < num_photons - 1:
            # System's recursivity: we need to call the loop indeterminate
            # times, depending of the amount of photons present
            evolution_loop(S, photons, num_photons, m, sum_, k + 1)

        else:
            mult = 1.0

            photons_aux = np.zeros(m, dtype=complex)

            # IMPORTANT, we want to explore sum_[] in order
            cont = 0

            # Here goes the sum algorithm. All i modes are visited a n_i
            # number of times corresponding to the amount of photons in each
            for p in range(m):
                # An S_ji·a+_j is added to the total sum, corresponding to this vector of the basis
                for q in range(counter[p]):
                    mult *= S[sum_[cont], p]

                    photons_aux, mult = a_dagger(sum_[cont], photons_aux, mult)

                    cont += 1

                mult = mult / sp.sqrt(complex(recur_factorial(photons[p])))

            # Following runs: we call the function photon_comb_index() for
            # obtaining the adequate index for each vector of the basis

            index = photon_comb_index(photons_aux, vec_base)

            # Coeficients of the applying of U upon |ket>
            U_ket[index] += mult
