# ---------------------------------------------------------------------------------------------------------------------------
# 					ALGORITHM 3: INVERSE PROBLEM. COMPUTATION OF S FROM OPTICAL SYSTEM EVOLUTION MATRIX
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

# File opening
from itertools import permutations

# NumPy instalation: in the cmd: 'py -m pip install numpy'
import numpy as np

from ..input_control import input_control, input_control_ints, input_control_intsDim
from ..Phase3_Aux._3_permutation_matrix import *
from ..Phase3_Aux._3_S_rebuild import S_output
from ..Phase3_Aux._3_u_m_algebra_and_image_subalgebra import matrix_u_basis_generator
from ..Phase3_Aux._3_verification_of_solution_existence import eq_sys_finder, verification
from ..legacy.read_matrix import read_matrix_from_txt
from ..legacy.recur_factorial import *
from ..unitary import *


def SfromU(
    file_input=True,
    U=False,
    file_output=True,
    filename=False,
    base_input=False,
    m=False,
    n=False,
    perm=False,
    acc_d=3,
    txt=False,
):
    """
    Loads .txt files containing an evolution matrix U. Should it be buildable via linear optics elements, its scattering matrix of origin S will be rebuilt. Modes can be permutted for different ways of placing the instruments.
    Information is displayed on-screen.
    """

    if txt is True:
        print("==============================================================================")
        print("||| INVERSE PROBLEM: COMPUTATION OF S FROM OPTICAL SYSTEM EVOLUTION MATRIX |||")
        print("==============================================================================\n\n")

    # Input control: in case there is something wrong with given inputs, it is notified on-screen
    file_input, filename, newfile, acc_d = input_control(
        3, file_input, U, file_output, filename, True, acc_d
    )

    # ----------U EVOLUTION OF THE MATRIX SYSTEM INPUT:----------

    # Loading U from the file name.txt
    if file_input is True:
        U = read_matrix_from_txt(filename)

    # M value (comb_evol(n,m)=comb(m+n-1,n))
    M = len(U[:, 0])

    if txt is True:
        print("\nU EVOLUTION OF THE MATRIX SYSTEM INPUT:\n")

        # We showcase the system evolution matrix (with a decimal accuracy acc_d)
        print("\nSystem evolution matrix loaded:\n")

        print(np.round(U, acc_d))

        print(f"\nDimensions: {M} x {M}\n")

    # ----------NUMBER OF MODES AND PHOTONS INPUT:----------

    # Initial input control
    m = input_control_intsDim(m, "m", 2)

    n = input_control_ints(n, "n", 1)

    # We can rebuild m mode-dimensional matrices S given a n-photon matrix U (M-dimensional). The code only admits
    # plausible combinations, that is, that verify comb_evol(n,m)=comb(m+n-1,n)=M

    while (
        comb_evol(n, m) != M
    ):  # in the function version, n and m are properly declared since launch
        print(
            "\nThe given photon number n and modes m do not satisfy the equation M=comb_evol(n,m)=comb(m+n-1,n).\n"
        )

        try:
            m = int(input("\nNumber of modes? "))

            n = int(input("\nNumber of photons? "))

        except ValueError:
            print("The given value is not valid.\n")

    photons = np.zeros(m)

    photons[0] = n

    # ----------UNITARY CHECK FOR MATRIX U:----------

    if txt is True:
        print("\n\n\n\nUNITARY CHECK FOR MATRIX U:\n")

        cond = unitary(U, M, filename, acc_d)

        if not cond:
            print("\nResults will be computed, albeit U (" + filename + ") is not unitary.\n")

    # ----------PART 1: CAN THIS MATRIX BE BUILT BY LINEAR OPTICS INSTRUMENTS?:----------

    # Beginning of time measurement
    t = time.process_time_ns()

    # Matrix basis generation
    (
        base_u_m,
        base_u_M,
        base_u_m_e,
        base_u_m_f,
        separator_e_f,
        base_U_m,
        base_U_M,
    ) = matrix_u_basis_generator(m, M, photons, base_input)

    # We obtain the equation system
    eq_sys, eq_sys_choice, index_choice = eq_sys_finder(base_u_m, base_u_M)

    # Verification of the system's validity: in case it is computable, the solution is obtained
    # In case it is not, "None" is given instead
    sol, sol_e, sol_f, check_sol = verification(
        U,
        base_u_m,
        base_u_m_e,
        base_u_m_f,
        separator_e_f,
        base_u_M,
        eq_sys,
        eq_sys_choice,
        index_choice,
    )

    if file_output is True:
        # Saving both basis of the u(m) and u(M) subspaces
        base_u_m_file = open(f"base_u_m_{m}.txt", "w+")

        base_u_M_file = open(f"base_u__M_{M}.txt", "w+")

        for i in range(m * m):
            np.savetxt(base_u_m_file, base_u_m[i], delimiter=",")

            np.savetxt(base_u_M_file, base_u_M[i], delimiter=",")

        base_u_m_file.close()

        base_u_M_file.close()

    # ----------PART 2: S MATRIX OBTENTION:----------

    if file_output is True:
        S_recon_file_no_perms = open(filename + f"_m_{m}_n_{n}_S_recon_main.txt", "w+")

        if perm is True:
            S_recon_file = open(filename + f"_m_{m}_n_{n}_S_recon_all.txt", "w+")
            U_perms_file = open(filename + f"_m_{m}_n_{n}_S_recon_all_U.txt", "w+")

            S_recon_file.write("S reconstruction for the input U:\n")
            U_perms_file.write("Different U permutations:\n")

    # In case a solution exists, S is rebuilt with the given results
    if check_sol is True:
        S = S_output(base_u_m, base_U_m, sol_e, sol_f)

        if txt is True:
            print("\n\n\nRebuilt S matrix:\n")
            print(np.round(S, acc_d))

        if file_output is True:
            np.savetxt(S_recon_file_no_perms, S, delimiter=",")

            if perm is True:
                np.savetxt(S_recon_file, S, delimiter=",")
                np.savetxt(U_perms_file, U, delimiter=",")

            S_recon_file_no_perms.close()

    else:
        if txt is True:
            print("\nA S solution has not been found for the matrix given.\n")

        S = False

        if file_output is True:
            S_recon_file_no_perms.write("A S solution has not been found for the matrix given.\n")

    # This algorithm has the same function as the previous if, but for numerous permutations of
    # the basis vectors
    if perm is True:
        if txt is True:
            print("\n\nPermutations are being computed, wait for a while...\n")

        ############## IMPORTANT ##############
        # By default, storage of the matrices U with their vector basis permuted is omitted for a better performance
        # If required, all commentaries containing 'U_perm_file' must have their commentaries undone
        # U_perm_file=open(f"m_{m}_n_{n}_U_perms.txt","w+")

        perm_iterator = permutations(range(M))

        if file_output is True:
            S_recon_file.write("\n\nStudy of permutations:\n")

        for item in perm_iterator:
            # U_perm_file.write(f"\n\nU (permutación {np.asarray(item)}):\n")

            # We compute the permutation matrix...
            M_perm = permutation_matrix(np.asarray(item))

            # Which we apply at both sides of the matrix U_perm for the basis change
            U_perm = M_perm.dot(U.dot(np.transpose(M_perm)))

            # We verify the solution's existence again, this time for each permutation
            sol, sol_e, sol_f, check_sol = verification(
                U_perm,
                base_u_m,
                base_u_m_e,
                base_u_m_f,
                separator_e_f,
                base_u_M,
                eq_sys,
                eq_sys_choice,
                index_choice,
            )

            # ----------PART 2: S MATRIX OBTENTION:----------

            if check_sol:  # ==True, implied
                S_perm = S_output(base_u_m, base_U_m, sol_e, sol_f)

                if file_output is True:
                    S_recon_file.write(f"\n\nS (permutation {np.asarray(item)}):\n")

                    np.savetxt(S_recon_file, S_perm, delimiter=",")

                    U_perms_file.write(f"\n\nU (permutation {np.asarray(item)}):\n")

                    np.savetxt(U_perms_file, U_perm, delimiter=",")

    if file_output is True:
        if perm is True:
            if txt is True:
                print(
                    f"\nAll results have been storaged in the file '{filename}_m_{m}_n_{n}_S_recon_all.txt'.\n"
                )
                print(
                    f"\nThe corresponding permutation matrices U have been storaged in the file '{filename}_m_{m}_n_{n}_S_recon_all_U.txt'.\n"
                )

            S_recon_file.close()
            U_perms_file.close()

        elif txt is True:
            print(
                f"\nThe results have been storaged in the file '{filename}_m_{m}_n_{n}_S_recon_main.txt'.\n"
            )

    # ----------TOTAL TIME REQUIRED:----------

    t_inc = time.process_time_ns() - t

    print(f"\nSfromU: total time of execution (seconds): {float(t_inc/(10**(9)))}\n")

    return S
