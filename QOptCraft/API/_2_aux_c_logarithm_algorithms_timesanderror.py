# ---------------------------------------------------------------------------------------------------------------------------
# 								ALGORITHM 2b: LOGARITHM OF A MATRIX ALGORITHMS COMPARISON
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

# NumPy instalation: in the cmd: 'py -m pip install numpy'
import numpy as np
import xlsxwriter

# SciPy instalation: in the cmd: 'py -m pip install scipy'
from scipy.linalg import expm, logm

# Matrices comparisons by their inner product
from ..legacy.mat_inner_product import mat_module
from ..Phase2_Aux._2_logarithm_algorithms import *
from ..utils.write_initial_matrix import haar_measure


def MatLogCompTnE(N1=False, N2=False, txt=False, exp=False):
    """
    An additional function intended for comparing times and error values for our matrix logarithm algorithms.
    Information is displayed on-screen.
    """

    if txt is True:
        print("===========================================================================")
        print("||| LOGARITHM OF AN UNITARY MATRIX ALGORITHMS TIME AND ERROR COMPUTATION|||")
        print("===========================================================================\n\n")

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

    if type(exp) is not int:
        print("\nWARNING: invalid exp input (needs to be int).")

        # We input the action of the index i over the series of dimensions
        while True:
            try:
                exp = int(
                    input(
                        "\nWill the matrix dimensions follow up in terms of i (press 1 or any other non mentioned number), or 2^i (press 2) for an index i?\n"
                    )
                )

                if exp != 2:
                    exp = 1

                break

            except ValueError:
                print("\nThe given value is not valid.\n")

    # MICROSOFT EXCEL INITIALIZATION
    workbook = xlsxwriter.Workbook(
        f"logarithm_comparisons_TnE_dim_{N1}to{N2}_{exp}.xlsx"
    )  # create the workbook

    cell_format = workbook.add_format({"center_across": True, "border": True, "valign": "vcenter"})
    # cell_format.set_border() (another way to assign classes)
    cell_format_number = workbook.add_format(
        {"center_across": True, "left": True, "right": True, "valign": "vcenter"}
    )
    cell_format_number_end = workbook.add_format(
        {"center_across": True, "left": True, "right": True, "bottom": True, "valign": "vcenter"}
    )
    # Create a format to use in the merged range.

    # ----------COMPUTATION OF THE LOGARITHM BY USING ALL ALGORITHMS, AND COMPARISONS:----------

    # Beginning of time measurement
    t = time.process_time_ns()

    worksheet = workbook.add_worksheet("Logarithms")  # add a sheet

    # Merge 3 cells.
    worksheet.merge_range("D4:H4", "Time (s)", cell_format)
    worksheet.merge_range("L4:P4", "Backwards Error", cell_format)

    # Excel shenanigans
    row = 5

    for i in range(N1, N2 + 1):
        # We generate DFT matrices, with a "sturdy" structure which serves as the adequate test for the algorithm
        A = haar_measure(np.power(2, i)) if exp == 2 else haar_measure(i)

        t_logm = time.process_time_ns()
        logm_A = logm(A)
        # We measure the time required for computing logm(A)
        t_logm = time.process_time_ns() - t_logm

        # The functions logm_3/4/5(A) give the output of the own function as well as its computation time
        logm_1A, time_1 = logm_1(A)
        logm_2A, time_2 = logm_2(A)
        logm_3A, time_3 = logm_3(A)
        logm_4A, time_4 = logm_4(A)
        logm_5A, time_5 = logm_5(A)

        # We apply expm(M) upon the logarithms
        expm(logm_A)
        exp_1A = expm(logm_1A)
        exp_2A = expm(logm_2A)
        exp_3A = expm(logm_3A)
        exp_4A = expm(logm_4A)
        exp_5A = expm(logm_5A)

        # We compare logm(A) with logm_3/4/5(A). Normally, they should not be equivalent
        inner_prod_1 = mat_module(exp_1A - A)
        inner_prod_2 = mat_module(exp_2A - A)
        inner_prod_3 = mat_module(exp_3A - A)
        inner_prod_4 = mat_module(exp_4A - A)
        inner_prod_5 = mat_module(exp_5A - A)

        # MICROSOFT EXCEL COMPATIBILITY
        if exp == 1:
            # Deviation from unitarity
            dev = mat_module(np.transpose(np.conj(A)).dot(A) - np.identity(i))

            scores_time = {
                "N": f"{i}",
                "Deviation": dev,
                "Algorithm 1": time_1 / (10.0**9),
                "Algorithm 2": time_2 / (10.0**9),
                "Algorithm 3": time_3 / (10.0**9),
                "Algorithm 4": time_4 / (10.0**9),
                "Algorithm 5": time_5 / (10.0**9),
            }  # your score time data (in seconds)
            scores_error = {
                "N": f"{i}",
                "Deviation": dev,
                "Algorithm 1": inner_prod_1,
                "Algorithm 2": inner_prod_2,
                "Algorithm 3": inner_prod_3,
                "Algorithm 4": inner_prod_4,
                "Algorithm 5": inner_prod_5,
            }  # your score error data

        else:
            # Deviation from unitarity
            dev = mat_module(np.transpose(np.conj(A)).dot(A) - np.identity(np.power(2, i)))

            scores_time = {
                "N": f"{np.power(2,i)}",
                "Deviation": dev,
                "Algorithm 1": time_1 / (10.0**9),
                "Algorithm 2": time_2 / (10.0**9),
                "Algorithm 3": time_3 / (10.0**9),
                "Algorithm 4": time_4 / (10.0**9),
                "Algorithm 5": time_5 / (10.0**9),
            }  # your score time data (in seconds)
            scores_error = {
                "N": f"{np.power(2,i)}",
                "Deviation": dev,
                "Algorithm 1": inner_prod_1,
                "Algorithm 2": inner_prod_2,
                "Algorithm 3": inner_prod_3,
                "Algorithm 4": inner_prod_4,
                "Algorithm 5": inner_prod_5,
            }  # your score error data

        col = 1

        # line index
        for name in scores_time:
            if row == 5:  # the loop
                worksheet.write(
                    row - 1, col, name, cell_format
                )  # write name at row 'row' and column 1
                worksheet.write(
                    row - 1, col + 8, name, cell_format
                )  # write name at row 'row' and column 1

            if i == N2:
                worksheet.write(
                    row, col, scores_time[name], cell_format_number_end
                )  # write score at last row
                worksheet.write(
                    row, col + 8, scores_error[name], cell_format_number_end
                )  # write score at last row

            else:
                worksheet.write(
                    row, col, scores_time[name], cell_format_number
                )  # write score at row 'row' and column 'col'
                worksheet.write(
                    row, col + 8, scores_error[name], cell_format_number
                )  # write score at row 'row+1' and column 'col'

            col += 1  # increment the line index

        row += 1

    workbook.close()  # close the workbook

    print(
        f"\nResults have been saved on a Workbook 'logarithm_comparisons_TnE_dim_{N1}to{N2}_{exp}.xlsx' file."
    )

    # ----------TOTAL TIME REQUIRED:----------

    t_inc = time.process_time_ns() - t

    print(f"\nTotal time of execution (seconds): {float(t_inc/(10**(9)))}\n")
