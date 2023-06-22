"""ALGORITHM 4: STUDY OF UNCRAFTABLE MATRICES U BY THE TOPONOGOV THEOREM
"""

import time

import numpy as np
from scipy.linalg import expm

from qoptcraft._legacy.input_control import input_control, input_control_ints, input_control_intsDim

from qoptcraft.basis import hilbert_dim
from qoptcraft.evolution._2_3rd_evolution_method import evolution_3
from qoptcraft._legacy.Phase3_Aux._3_u_m_algebra_and_image_subalgebra import (
    matrix_u_basis_generator,
)

from qoptcraft._legacy.photon_comb_basis import photon_combs_generator
from qoptcraft._legacy.read_matrix import read_matrix_from_txt
from qoptcraft._legacy.recur_factorial import *
from qoptcraft.operators import haar_random_unitary


def Toponogov(
    file_input=True,
    U_input=False,
    file_output=True,
    filename=False,
    base_input=False,
    tries=False,
    m=False,
    n=False,
    acc_d=3,
    txt=False,
    acc_t=8,
    vec_base=[[False, False], [False, False]],
):
    """
    Loads .txt files containing a non-available for implementation evolution matrix U. Given an amount of tries and a number of modes m and photons n, by the Toponogov theorem the closer viable evolutions are found.
    Information is displayed on-screen.
    """

    print("================================================================")
    print("||| STUDY OF UNCRAFTABLE MATRICES U BY THE TOPONOGOV THEOREM |||")
    print("================================================================\n\n")

    # Input control: in case there is something wrong with given inputs, it is notified on-screen
    file_input, filename, newfile, acc_d = input_control(
        4, file_input, U_input, file_output, filename, txt, acc_d
    )

    # Initial input control
    tries = input_control_ints(tries, "tries", 1)

    # ----------U MATRIX NOT CRAFTABLE WITH OPTICAL DEVICES:----------

    # Loading U from the file name.txt
    if file_input is True:
        U_input = read_matrix_from_txt(filename)
    # U_input=qft_matrix_auto(3,-0.4999999999999998+0.8660254037844387j)*(0.8660254037844387+0.49999999999999994j)

    if txt is True:
        print("\nU_input:")
        print(np.round(U_input, acc_d))

    M = len(U_input[:, 0])

    # ----------NUMBER OF OG MODES AND LOOP STEPS INPUT:----------

    # Initial input control
    m = input_control_intsDim(m, "m", 2)

    n = input_control_ints(n, "n", 1)

    # We can rebuild m mode-dimensional matrices S given a n-photon matrix U (M-dimensional). The code only admits
    # plausible combinations, that is, that verify comb_evol(n,m)=comb(m+n-1,n)=M

    while (
        hilbert_dim(m, n) != M
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

    S_rand = haar_random_unitary(m)

    # We load the combinations with the same amount of photons in order to create the vector basis
    if np.array(vec_base)[0, 0]:
        if txt:
            print("\nLoaded an external array for the Fock basis.")

    else:
        vec_base = photon_combs_generator(m, photons)

    # vec_base=np.array(((2, 0),(0, 2),(1, 1)))

    # print(vec_base)

    # ----------SUBSPACE u(M) BASIS:----------

    base_u_M = np.zeros((m * m, M, M), dtype=complex)

    if base_input is True:
        # Loading both bases
        try:
            base_u_M_file = open(f"base_u__M_{M}.txt")

            for i in range(m * m):
                base_u_M[i] = np.loadtxt(base_u_M_file, delimiter=",", max_rows=M, dtype=complex)

                base_u_M[i] = base_u_M[i] / np.sqrt(mat_norm(base_u_M[i]))

            base_u_M_file.close()

        except FileNotFoundError:
            print("\nA file required does not exist.\nIt will be freshly generated instead.\n")

            base_u_M = matrix_u_basis_generator(m, M, photons, base_input)[1]
            base_u_M = base_u_M[: m * m]

    else:
        base_u_M = matrix_u_basis_generator(m, M, photons, base_input)[1]  # image algebra
        base_u_M = base_u_M[: m * m]

    base_u_M = gram_schmidt_2dmatrices(base_u_M)

    if txt is True:
        print(f"\nu({M}) basis:")
        print(np.round(base_u_M, acc_d))

    # ----------ITERATIONS:----------

    # Beginning of time measurement
    t = time.process_time_ns()

    U_iter = np.identity(M, dtype=complex)

    sol_array = np.zeros((1, M, M), dtype=complex)
    sol_mod = np.zeros(1)

    for k in range(tries):
        first = False

        if k > 0:
            S_rand = haar_random_unitary(m)
            U_iter = evolution_3(S_rand, photons, vec_base)[0]

        # print(f"\nU_{k}:")
        # print(np.round(U_iter,acc_d))

        while True:
            coefs = np.zeros(m * m, dtype=complex)
            logm_3T = np.zeros((M, M), dtype=complex)

            # Projection onto the loaded u(M) basis
            for i in range(m * m):
                coefs[i] = mat_inner_product(
                    logm_3(np.linalg.inv(U_iter).dot(U_input))[0], base_u_M[i]
                )
                # coefs[i]=(logm_3(np.linalg.inv(U_iter).dot(U_input))[0]).dot(base_u_M[i])
                # coefs[i]=mat_inner_product(LogU(np.linalg.inv(U_iter).dot(U_input),10),base_u_M[i])

            for i in range(m * m):
                logm_3T += coefs[i] * base_u_M[i]

            U_iter = U_iter.dot(expm(logm_3T))

            if first is True and np.abs(mod - mat_norm(U_input - U_iter)) < 10 ** (-acc_t):
                # sol_array[k]=U_iter
                # sol_mod[k]=mod
                if k == 0:
                    sol_array[0] = U_iter
                    sol_mod[0] = mod

                else:
                    boolean_arr = np.zeros(len(sol_array))

                    for l in range(len(sol_array)):
                        boolean_arr[l] = (
                            np.round(U_iter, acc_d) == np.round(sol_array[l], acc_d)
                        ).all()

                    if not (boolean_arr[:] is True).any():
                        sol_array = np.append(sol_array, np.array([U_iter]), axis=0)
                        sol_mod = np.append(sol_mod, np.array([mod]), axis=0)

                break

            first = True

            mat_norm(U_input - U_iter)
            mat_norm(logm_3(np.linalg.inv(U_iter).dot(U_input))[0])
            # mod_logm=mat_norm(LogU(np.linalg.inv(U_iter).dot(U_input),10))

        """sol_file=open(f"{filename}_3d__MAIN_sol.txt","w+")

		np.savetxt(sol_file,U_iter,delimiter=",")

		sol_file.close()"""

    if txt is True:
        print("\n\nSolution:")
        print(np.round(sol_array, acc_d))
        print("\nRespective separation lenghts:")
        print(np.round(sol_mod, acc_d))

    if file_output is True:
        toponogov_file = open(f"{filename}_toponogov_general.txt", "w+")

        for i in range(len(sol_mod)):
            toponogov_file.write(
                "\ntoponogov("
                + filename
                + f")_{i+1} (separation length from original of {sol_mod[i]}):\n"
            )

            toponogov_file_2 = open(f"{filename}_toponogov_{i+1}.txt", "w+")

            np.savetxt(toponogov_file, sol_array[i], delimiter=",")

            np.savetxt(toponogov_file_2, sol_array[i], delimiter=",")

            toponogov_file_2.close()

        toponogov_file.close()

        if txt is True:
            print(
                f"\nAll matrix arrays with their separation length have been saved in '{filename}_toponogov_general.txt'."
            )

            print(
                f"\nFor each try, a file solely containing each matrix by the names of '{filename}_toponogov_N.txt', N being the number, have been saved."
            )

    # ----------TOTAL TIME REQUIRED:----------

    t_inc = time.process_time_ns() - t

    print(f"\nToponogov: total time of execution (seconds): {float(t_inc/(10**(9)))}\n")

    return sol_array
