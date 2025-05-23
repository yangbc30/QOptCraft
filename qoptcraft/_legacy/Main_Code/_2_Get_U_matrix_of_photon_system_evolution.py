# ---------------------------------------------------------------------------------------------------------------------------
# 									ALGORITHM 2: N-PHOTON OPTICAL SYSTEM EVOLUTION
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

# ---------------------------------------------------------------------------------------------------------------------------
# 													LIBRARIES REQUIRED
# ---------------------------------------------------------------------------------------------------------------------------


# ----------TIME OF EXECUTION MEASUREMENT:----------

import time


# ----------MATHEMATICAL FUNCTIONS/MATRIX OPERATIONS:----------

# NumPy instalation: in the cmd: 'py -m pip install numpy'
import numpy as np


# ----------FILE MANAGEMENT:----------

# File opening

from ..read_matrix import read_matrix_from_txt


# ----------COMBINATORY:----------

from ..recur_factorial import comb_evol


# ----------SYSTEM:----------


# ----------UNITARY MATRIX CONDITION----------

from ..unitary import *


# ----------INPUT CONTROL:----------

from ..input_control import input_control, input_control_ints


# ----------ALGORITHM 2: AUXILIAR FUNCTIONS:----------


from ..Phase2_Aux._2_1st_evolution_method import evolution

from ..Phase2_Aux._2_2nd_evolution_method import evolution_2, evolution_2_ryser

from ..Phase2_Aux._2_3rd_evolution_method import evolution_3, iH_U_operator


# ----------PHOTON COMB BASIS:----------

from ..photon_comb_basis import photon_combs_generator


# ---------------------------------------------------------------------------------------------------------------------------
# 														MAIN CODE
# ---------------------------------------------------------------------------------------------------------------------------


def iHStoiHU(
    file_input=True,
    iH_S=False,
    file_output=True,
    filename=False,
    n=False,
    acc_d=3,
    txt=False,
    vec_base=[[False, False], [False, False]],
):
    # Initial input control
    n = input_control_ints(n, "n", 1)

    # ----------iH_S MATRIX OF THE SYSTEM INPUT:----------

    # Load S matrix
    if file_input is True:
        iH_S = read_matrix_from_txt(filename)

    m = len(iH_S[:, 0])

    if txt is True:
        print("\niH_S MATRIX OF THE SYSTEM INPUT:\n")

        print("\nInput matrix iH_S:\n")

        print(np.round(iH_S, acc_d))

        print(f"\nDimensions: {m} x {m}\n")

    # ----------NUMBER OF PHOTONS INPUT---------

    photons = np.zeros(m)

    photons[0] = n

    # We load the combinations with the same amount of photons in order to create the vector basis
    if np.array(vec_base)[0, 0]:
        if txt:
            print("\nLoaded an external array for the Fock basis.")

    else:
        vec_base = photon_combs_generator(m, photons)

    if file_output is True:
        # We save the vector basis
        vec_base_file = open(f"m_{m}_n_{n}_vec_base.txt", "w")

        np.savetxt(vec_base_file, vec_base, fmt="(%e)", delimiter=",")

        vec_base_file.close()

    # Resulting U matrix's dimensions:
    M = comb_evol(n, m)

    t = time.process_time_ns()

    iH_U = iH_U_operator(file_output, filename, iH_S, m, M, vec_base)

    # ----------TOTAL TIME REQUIRED:----------

    t_inc = time.process_time_ns() - t

    print(f"\niHStoiHU: total time of execution (seconds): {float(t_inc / (10 ** (9)))}\n")

    return iH_U, vec_base


def StoU(
    file_input=True,
    S=False,
    file_output=True,
    filename=False,
    method=2,
    n=False,
    acc_d=3,
    txt=False,
    vec_base=[False, False],
):
    """
    Loads .txt files containing an unitary matrix (the so-called scattering matrix S). Depending on the total number of photons within the modes, a different evolution matrix U will be obtained.
    Information is displayed on-screen.
    """

    if txt is True:
        print("\n\n================================================================")
        print("||| EVOLUTION OF A PHOTON SYSTEM GIVEN A SCATTERING MATRIX S |||")
        print("================================================================\n\n")

    # Input control: in case there is something wrong with given inputs, it is notified on-screen
    file_input, filename, newfile, acc_d = input_control(
        2, file_input, S, file_output, filename, True, acc_d
    )

    if type(method) is not int:
        print("\nWARNING: invalid method input (needs to be int).")

        while True:
            try:
                method = int(
                    input(
                        "\nInput '0' (or any other number not mentioned afterwards) for computation following the quantum mechanical description\nInput '1' for computing by the standard permanents method\nInput '2' for computing by the Ryser permanents method\nInput '3' for the Hamiltonian computation method\n"
                    )
                )

                break

            except ValueError:
                print("The given value is not valid.\n")

    # Initial input control
    n = input_control_ints(n, "n", 1)

    # ----------S MATRIX OF THE SYSTEM INPUT:----------

    # Load S matrix
    if file_input is True:
        S = read_matrix_from_txt(filename)

    m = len(S[:, 0])

    if txt is True:
        print("\nS MATRIX OF THE SYSTEM INPUT:\n")

        print("\nInput matrix S:\n")

        print(np.round(S, acc_d))

        print(f"\nDimensions: {m} x {m}\n")

        # ----------UNITARY CHECK FOR MATRIX S:----------

        print("\n\n\n\nUNITARY CHECK FOR MATRIX S:\n")

        cond = unitary(S, m, filename, acc_d)

        # In case S is not unitary, the user is advised of the possible lack of value in the results
        # (see also last section: VERIFICATION OF THE PROGRAM'S SUCCESS)
        if cond is False:
            print("\nResults will be computed, albeit S (" + filename + ") is not unitary.\n")

    # ----------NUMBER OF PHOTONS INPUT---------

    photons = np.zeros(m)

    photons[0] = n

    # We load the combinations with the same amount of photons in order to create the vector basis
    if np.array(vec_base)[0, 0]:
        if txt:
            print("\nLoaded an external array for the Fock basis.")

    else:
        vec_base = photon_combs_generator(m, photons)

    if file_output is True:
        # We save the vector basis
        vec_base_file = open(f"m_{m}_n_{n}_vec_base.txt", "w")

        np.savetxt(vec_base_file, vec_base, fmt="(%e)", delimiter=",")

        vec_base_file.close()

    # ----------COMPUTATION BY THE CHOSEN METHOD:----------

    if txt is True:
        print("\n\nPhase2: beginning of the computation...")

    # Beginning of time measurement
    t = time.process_time_ns()

    # The first two algorithms were design with compatibility with just a photon vector (see the function)
    # photon_introd_one_input() in '_2_photon_input.py'). Thus, it is necesary to execute a loop involving
    # all the basis, and compute values such as M which the third algorithm does by default
    if method != 3:
        M = len(vec_base)

        U = np.zeros((M, M), dtype=complex)

        t_inc = 0

        for i in range(M):
            if method == 1:
                U[i], t_inc_aux = evolution_2(S, vec_base[i], vec_base)

            elif method == 2:
                U[i], t_inc_aux = evolution_2_ryser(S, vec_base[i], vec_base)

            else:
                method = 0  # readjust so the default method is associated to '0' in the output's filename

                U[i], t_inc_aux = evolution(S, vec_base[i], vec_base)

            t_inc += t_inc_aux

        U = np.transpose(U)

    elif method == 3:
        U, t_inc = evolution_3(S, photons, vec_base, file_output, filename)

    print(f"\nComputation time of chosen method {method} (in seconds): {float(t_inc / (10**9))}")

    # ----------STORAGE OF THE U EVOLUTION MATRIX COEFICIENTS AND PROBABILITIES:----------

    if file_output is True:
        # Storage in text of the resulting matrix U
        coefs_file = open(filename + f"_m_{m}_n_{n}_coefs_method_{method}.txt", "w")

        # Storage in text of the probability matrix, which comprehends the probabilities of
        # obtaining each photon distribution from another, in the corresponding cell
        probs_file = open(filename + f"_m_{m}_n_{n}_probs_method_{method}.txt", "w")

        np.savetxt(coefs_file, U, delimiter=",")
        np.savetxt(probs_file, np.real(np.multiply(np.conj(U), U)), delimiter=",", fmt="(%e)")

    # ----------VERIFICATION OF THE PROGRAM'S SUCCESS:----------

    # We have supposedly found a solution, whose validity is subject to the next loop

    if txt is True:
        # ----------UNITARY CHECK FOR MATRIX U:----------

        print("\n\n\n\n\nUNITARY CHECK FOR MATRIX U:\n")

        cond = unitary(U, len(U), "U", acc_d)

        if cond is False:
            print("\nS (" + filename + ")'s corresponding evolution U is not unitary.\n")

        else:
            print("The resulting matrix is unitary.")

    # Further check: we initialise a variable 'sol'
    sol = True

    for i in range(len(vec_base[:, 0])):
        # We compute the probability sum for each U|vector_inicial>
        sum_probs = np.sum(np.real(np.multiply(np.conj(U), U))[:, i])

        # Condition which invalidates the solution: the sum of the probabilities given for each U|vector_inicial> is not 1.0
        if np.round(sum_probs, 10) != 1.0:
            sol = False

            break

    if sol is True:
        if txt is True:
            print("\n\nProgram successfully executed.\n")

            print("\nOutput evolution matrix:")
            print(np.round(U, acc_d))

        if file_output is True:
            probs_file.write(
                "\nFor all columns, the sum of probabilities is practically equal to 1.0. "
            )

    else:
        if txt is True:
            print(
                "\nThere are discrepances in the sum of probabilities for at least one column. Was the input matrix really unitary?\n"
            )

        if file_output is True:
            # Warning of the program's failure
            probs_file.write(
                "\nThe sum of probabilities is not equal to 1.0 for at least one of the columns of U.\nThere has been a problem with the input.\n"
            )

    if file_output is True:
        coefs_file.close()

        probs_file.close()

    # ----------TOTAL TIME REQUIRED:----------

    t_inc = time.process_time_ns() - t

    print(f"\nStoU: total time of execution (seconds): {float(t_inc / (10 ** (9)))}\n")

    return U, vec_base
