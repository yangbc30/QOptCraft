# ---------------------------------------------------------------------------------------------------------------------------
# 					ALGORITHM 2a: COMPUTATION TIME OF ALGORITHM 2 METHODS OF SYSTEM EVOLUTION COMPARISON
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

import math
import time

import numpy as np
import xlsxwriter

from ..utils.input_control import input_control_ints
from ..Phase2_Aux._2_1st_evolution_method import evolution
from ..Phase2_Aux._2_2nd_evolution_method import evolution_2, evolution_2_ryser
from ..Phase2_Aux._2_3rd_evolution_method import evolution_3
from QOptCraft.legacy.photon_comb_basis import photon_combs_generator
from ..legacy.recur_factorial import comb_evol
from ..utils.write_initial_matrix import haar_measure


def StoUEvolComp(
    file_output=True,
    m1=False,
    m2=False,
    n1=False,
    n2=False,
    txt=False,
    tries=1,
    vec_base=[[False, False], [False, False]],
    inverse=False,
    comparison_matrix="haar",
):
    """
    An additional function intended for comparing times of computation between scattering matrix evolution algorithms.
    Information is displayed on-screen.
    """

    if txt is True:
        print("==============================================================================")
        print("||| COMPUTATION TIME OF ALGORITHM 2 METHODS OF SYSTEM EVOLUTION COMPARISON |||")
        print("==============================================================================\n\n")

        # ----------INPUT OF DIMENSIONS AND NUMBER OF PHOTONS INTERVALS:----------

        print(
            "In this code, computation times of the three implemented methods in the main algorithm 2 are compared.\n\n"
        )

    if (type(m1) is not int) or (type(m2) is not int):
        print("\nWARNING: invalid m1 or m2 input (both need to be int).")

        # We input the interval of dimensions to be computed by the algorithm
        while True:
            try:
                m1 = int(input("\nInitial dimension of the loop? (it cannot be lower than 2): "))

                m2 = int(
                    input(
                        "\nFinal dimension of the loop? (it cannot be lower than the initial dimension): "
                    )
                )

                if m1 < 2 or m2 < m1:
                    print(
                        "\nThere is at least a given value not included in the possible domain.\n"
                    )

                else:
                    break

            except ValueError:
                print("\nThe given value is not valid.\n")

    if (type(n1) is not int) or (type(n2) is not int):
        print("\nWARNING: invalid n1 or n2 input (both need to be int).")

        # We input the interval of number of photons to be computed by the algorithm
        while True:
            try:
                n1 = int(
                    input("\nInitial number of photons of the loop? (it cannot be lower than 1): ")
                )

                n2 = int(
                    input(
                        "\nFinal number of photons of the loop? (it cannot be lower than the initial number of photons): "
                    )
                )

                if n1 < 1 or n2 < n1:
                    print(
                        "\nThere is at least a given value not included in the possible domain.\n"
                    )

                else:
                    break

            except ValueError:
                print("\nThe given value is not valid.\n")

    if txt is True:
        print("\nNow, we will generate random unitary matrices S for each dimension.\n")
        print(
            "Per S, the evolution of the system matrix will be computed for each number of photons.\n"
        )

        # ----------COMPARISON BETWEEN COMPUTATION TIMES FOR THE THREE EVOLUTION METHODS:----------

        print("\nCOMPUTATION TIME COMPARISON (time in seconds):")
        print(f"\nDimensions m range: [{m1}, {m2}]")
        print(f"\nPhotons n range: [{n1}, {n2}]")

    if file_output is True:
        # MICROSOFT EXCEL INITIALIZATION
        workbook = xlsxwriter.Workbook(
            f"computation_comparisons_m_{m1}to{m2}_n_{n1}to{n2}_{comparison_matrix}.xlsx"
        )  # create the workbook

        cell_format = workbook.add_format(
            {"center_across": True, "border": True, "align": "center", "valign": "vcenter"}
        )
        # cell_format.set_border() (another way to assign classes)
        cell_format_number = workbook.add_format(
            {
                "center_across": True,
                "left": True,
                "right": True,
                "align": "center",
                "valign": "vcenter",
            }
        )
        cell_format_number_end = workbook.add_format(
            {
                "center_across": True,
                "left": True,
                "right": True,
                "bottom": True,
                "align": "center",
                "valign": "vcenter",
            }
        )

    tries = input_control_ints(tries, "tries", 1)

    # Beginning of time measurement
    t = time.process_time_ns()

    for cont in range(tries):
        if file_output is True:
            worksheet = workbook.add_worksheet(f"Attempt {cont+1}")  # add a sheet

        col = 1

        for k in range(m1, m2 + 1):
            # New initial matrix
            if comparison_matrix == "haar":
                S = haar_measure(k)

            elif comparison_matrix == "qft":
                S = np.zeros((k, k), dtype=complex)
                factor = -1 if inverse is True else 1

                for i in range(k):
                    for j in range(k):
                        S[i, j] = np.exp(factor * 2.0 * math.pi * 1j / float(k) * i * j)

            while comparison_matrix != "haar" and comparison_matrix != "qft":
                print("\nThe chosen matrix for comparison is not available.")

                comparison_matrix == input(
                    "Available options selected via writing what is between parenthesis for each:\n1) Haar random matrices (haar)\n2) Quantum Fourier Transform matrices (qft)"
                )

                # New initial matrix
                if comparison_matrix == "haar":
                    S = haar_measure(k)

                elif comparison_matrix == "qft":
                    S = np.zeros((k, k), dtype=complex)
                    factor = -1 if inverse is True else 1

                    for i in range(k):
                        for j in range(k):
                            S[i, j] = np.exp(factor * 2.0 * math.pi * 1j / float(k) * i * j)

            photons = np.zeros(k, dtype=float)

            for j in range(n1, n2 + 1):
                if txt is True:
                    print(f"\n\nDimensions m = {k}, photons n = {j}")

                photons[0] = j

                # We load the combinations with the same amount of photons in order to create the vector basis
                if np.array(vec_base)[0, 0] and k == m1 and j == n1:
                    if txt:
                        print("\nLoaded an external array for the Fock basis.")

                else:
                    vec_base = photon_combs_generator(k, photons)

                M = len(vec_base)

                U = np.zeros((M, M), dtype=complex)

                t_inc_1 = 0

                for i in range(M):
                    U[i], t_inc_aux = evolution(S, vec_base[i], vec_base)

                    t_inc_1 += t_inc_aux

                U = np.transpose(U)

                t_inc_2 = 0

                for i in range(M):
                    U[i], t_inc_aux = evolution_2(S, vec_base[i], vec_base)

                    t_inc_2 += t_inc_aux

                U = np.transpose(U)

                t_inc_2b = 0

                for i in range(M):
                    U[i], t_inc_aux = evolution_2_ryser(S, vec_base[i], vec_base)

                    t_inc_2b += t_inc_aux

                U = np.transpose(U)

                U, t_inc_3 = evolution_3(S, photons, vec_base)

                if txt is True:
                    print(
                        f"\nMethod 1): {float(t_inc_1/(10**9))}         Method 2): {float(t_inc_2/(10**9))}         Method 2b): {float(t_inc_2b/(10**9))}         Method 3): {float(t_inc_3/(10**9))}\n\n"
                    )

                if file_output is True:
                    # MICROSOFT EXCEL COMPATIBILITY
                    scores = {
                        "m": f"{k}",
                        "n": f"{j}",
                        "M": f"{comb_evol(j,k)}",
                        "Method 1": t_inc_1 / (10.0**9),
                        "Method 2": t_inc_2 / (10.0**9),
                        "Method 2b": t_inc_2b / (10.0**9),
                        "Method 3": t_inc_3 / (10.0**9),
                    }  # your score data
                    i = 0
                    # line index
                    for name in scores:
                        if col == 1:  # the loop
                            worksheet.write(
                                0, i, name, cell_format
                            )  # write name at row i and column 0

                        if k == m2 and j == n2:
                            worksheet.write(
                                col, i, scores[name], cell_format_number_end
                            )  # write score at last column

                        else:
                            worksheet.write(
                                col, i, scores[name], cell_format_number
                            )  # write score at row i and column col

                        i += 1  # increment the line index

                    col += 1

    if file_output is True:
        print(
            f"\nResults have been saved on a Workbook 'computation_comparisons_m_{m1}to{m2}_n_{n1}to{n2}.xlsx' file."
        )

        workbook.close()  # close the workbook

    # ----------TOTAL TIME REQUIRED:----------

    t_inc = time.process_time_ns() - t

    print(f"\nTotal time of execution (seconds): {float(t_inc/(10**(9)))}\n")
