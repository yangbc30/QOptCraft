"""ALGORITHM 2b: LOGARITHM OF A MATRIX ALGORITHMS COMPARISON
"""
import time

from scipy.linalg import expm, logm

# Matrix comparisons by their inner product
from QOptCraft.legacy.mat_inner_product import comparison_noprint
from QOptCraft.utils.Phase2_Aux._2_logarithm_algorithms import *
from QOptCraft.utils.write_initial_matrix import dft_matrix_auto


def MatLogCompV(N1=False, N2=False, txt=False, exp=False):
    """
    An additional function intended for comparing the indexes and time values between matrix logarithms (and their exponentials), mainly 3, 4 and 5.
    Information is displayed on-screen.
    """

    if txt is True:
        print("======================================================================")
        print("||| LOGARITHM OF A MATRIX ALGORITHMS COMPARISON (WITH DFT MATRICES)|||")
        print("======================================================================\n\n")

    # ----------FUNCTION TO COMPARE CHOICE:----------

    # The user is given the possibility of comparing the different logarithms, or their exponentials. The
    # latter option exists as a test for the algorithms giving the same values of exp(log(A)), avoiding
    # possible discrepances with the different roots of logm_3/4/5 and logm
    if type(exp) is not int:
        print("\nWARNING: invalid exp input (needs to be int).")

        # We input the action of the index i over the series of dimensions
        while True:
            try:
                exp = int(
                    input(
                        "\nWill the matrix logarithms be compared (press 1 or any other non mentioned number), or their exponentials (press 2)?\n"
                    )
                )

                if exp != 2:
                    exp = 1

                break

            except ValueError:
                print("\nThe given value is not valid.\n")

    # ----------DIMENSIONS INTERVAL:----------

    # We input the interval of dimensions to be computed by the algorithm

    if (type(N1) is not int) or (type(N2) is not int):
        print("\nWARNING: invalid N1 or N2 input (both need to be int).")

        while True:
            try:
                N1 = int(input("\nInitial dimension of the loop? (it cannot be lower than 2): "))

                N2 = int(
                    input(
                        "\nFinal dimension of the loop? (it cannot be lower than the initial dimension): "
                    )
                )

                if N1 < 2 or N2 < N1:
                    print(
                        "\nThere is at least a given value not included in the possible domain.\n"
                    )

                else:
                    break

            except ValueError:
                print("\nThe given value is not valid.\n")

    # ----------COMPUTATION OF THE LOGARITHM BY USING ALL ALGORITHMS, AND COMPARISONS:----------

    # Beginning of time measurement
    t = time.process_time_ns()

    comparison_file = open(f"logarithm_comparisons_V_dim_{N1}to{N2}_{exp}.txt", "w+")

    for i in range(N1, N2 + 1):
        # We generate DFT matrices, with a "sturdy" structure which serves as the adequate test for the algorithm
        A = dft_matrix_auto(i)

        t_logm = time.process_time_ns()
        obj_A = logm(A)
        # We measure the time required for computing logm(A)
        t_logm = time.process_time_ns() - t_logm

        # The functions logm_3/4/5(A) give the output of the own function as well as its computation time
        obj_1A = logm_1(A)[0]
        obj_2A = logm_2(A)[0]
        obj_3A = logm_3(A)[0]
        obj_4A = logm_4(A)[0]
        obj_5A = logm_5(A)[0]

        if exp == 2:
            # We apply expm(M) upon the logarithms
            obj_A = expm(obj_A)
            obj_1A = expm(obj_1A)
            obj_2A = expm(obj_2A)
            obj_3A = expm(obj_3A)
            obj_4A = expm(obj_4A)
            obj_5A = expm(obj_5A)

        # We compare logm(A) with logm_3/4/5(A). Normally, they should not be equivalent
        comp_OG_y_1 = comparison_noprint(obj_A, obj_1A)
        comp_OG_y_2 = comparison_noprint(obj_A, obj_2A)
        comp_OG_y_3 = comparison_noprint(obj_A, obj_3A)
        comp_OG_y_4 = comparison_noprint(obj_A, obj_4A)
        comp_OG_y_5 = comparison_noprint(obj_A, obj_5A)

        # We compare logm_3/4/5(A) with each other
        comp_3_y_4 = comparison_noprint(obj_3A, obj_4A)
        comp_3_y_5 = comparison_noprint(obj_3A, obj_5A)
        comp_4_y_5 = comparison_noprint(obj_4A, obj_5A)

        if exp == 2:
            comparison_file.write(
                f"N = {i}: expm(logm_3(A)) = expm(logm_4(A))? {comp_3_y_4}, expm(logm_3(A)) = expm(logm_5(A))? {comp_3_y_5}, expm(logm_4(A)) = expm(logm_5(A))? {comp_4_y_5}\n"
            )

        else:
            comparison_file.write(
                f"N = {i}: logm_3(A) = logm_4(A)? {comp_3_y_4}, logm_3(A) = logm_5(A)? {comp_3_y_5}, logm_4(A) = logm_5(A)? {comp_4_y_5}\n"
            )

        if comp_OG_y_1 and comp_OG_y_2 and comp_OG_y_3 and comp_OG_y_4 and comp_OG_y_5:
            comparison_file.write("logm(A) is equal to all of the new algorithms.\n\n")

        elif comp_OG_y_1 or comp_OG_y_2 or comp_OG_y_3 or comp_OG_y_4 or comp_OG_y_5:
            comparison_file.write("logm(A) is equal to some of the new algorithms.\n\n")

        else:
            comparison_file.write("logm(A) is not equal to any of the new algorithms.\n\n")

    comparison_file.close()

    print(
        f"\nResults have been saved on a 'logarithm_comparisons_V_dim_{N1}to{N2}_{exp}.txt' file."
    )

    # ----------TOTAL TIME REQUIRED:----------

    t_inc = time.process_time_ns() - t

    print(f"\nTotal time of execution (seconds): {float(t_inc/(10**(9)))}\n")
