import time

import numpy as np

from QOptCraft.basis import hilbert_dim, get_photon_basis
from QOptCraft._legacy.input_control import input_control, input_control_ints
from QOptCraft.evolution._2_1st_evolution_method import evolution
from QOptCraft.evolution._2_2nd_evolution_method import evolution_2, evolution_2_ryser
from QOptCraft.evolution._2_3rd_evolution_method import evolution_3, iH_U_operator
from QOptCraft._legacy.photon_comb_basis import photon_combs_generator
from QOptCraft._legacy.read_matrix import read_matrix_from_txt


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
    M = hilbert_dim(m, n)

    t = time.process_time_ns()

    iH_U = iH_U_operator(file_output, filename, iH_S, m, M, vec_base)

    # ----------TOTAL TIME REQUIRED:----------

    t_inc = time.process_time_ns() - t

    print(f"\niHStoiHU: total time of execution (seconds): {float(t_inc/(10**(9)))}\n")

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
    modes = len(S)

    vec_base = get_photon_basis(modes, n)

    photons = np.zeros(modes)
    photons[0] = n

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

    return U, vec_base
