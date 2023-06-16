"""Available generators
"""

import math

import numpy as np

from QOptCraft.basis import hilbert_dim
from QOptCraft.utils.input_control import input_control_ints, input_control_intsDim
from QOptCraft.utils.Phase3_Aux._3_u_m_algebra_and_image_subalgebra import matrix_u_basis_generator
from QOptCraft._legacy.photon_comb_basis import photon_combs_generator
from QOptCraft.utils.write_initial_matrix import matrix_generation_general_auto, haar_measure
from ._2_Get_U_matrix_of_photon_system_evolution import StoU


# Creates Fock states basis of n photons for m modes
def Fock(file_output=True, m=False, n=False):
    # Initial input control
    m = input_control_intsDim(m, "m", 2)

    n = input_control_ints(n, "n", 1)

    # Main
    photons = np.zeros(m)

    photons[0] = n

    vec_base = photon_combs_generator(m, photons)

    if file_output is True:
        # We save the vector basis
        vec_base_file = open(f"m_{m}_n_{n}_vec_base.txt", "w")

        np.savetxt(vec_base_file, vec_base, fmt="(%e)", delimiter=",")

        vec_base_file.close()

        print(
            f"\nThe Fock states vector basis has been generated. It is found in the file 'm_{m}_n_{n}_vec_base.txt'.\n"
        )

    return vec_base


def AlgBasis(file_output=True, m=False, n=False, M=False):
    # Initial input control
    m = input_control_intsDim(m, "m", 2)

    n = input_control_ints(n, "n", 1)

    M = input_control_intsDim(M, "n", 2)

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

    base_u_m, base_u_M = matrix_u_basis_generator(m, M, photons, False)[:2]

    if file_output is True:
        # Saving both basis of the u(m) and u(M) subspaces
        base_u_m_file = open(f"base_u_m_{m}.txt", "w+")

        base_u_M_file = open(f"base_u__M_{M}.txt", "w+")

        for i in range(m * m):
            np.savetxt(base_u_m_file, base_u_m[i], delimiter=",")

            np.savetxt(base_u_M_file, base_u_M[i], delimiter=",")

        base_u_m_file.close()

        base_u_M_file.close()

        print(
            f"\nThe bases for the subalgebras u({m}) and u({M}) have been generated. They are found in the files 'base_u_m_{m}.txt' and 'base_u_M__{M}.txt' respectively.\n"
        )

    return base_u_m, base_u_M


# This function deploys haar_measure(N), N-dimensional unitary random matrix generator,
# also allowing an N input for support and file printing
def RandU(file_output=True, filename=False, N=False, txt=False):
    if txt is True:
        print("\n\nRANDOM UNITARY MATRIX GENERATOR (dim N x N):\n")

    while file_output is True and filename is False:
        print("\nWARNING: a new filename is required.")

        try:
            filename = input("Write the name of the file (without .txt extension): ")

        except ValueError:
            print("The given value is not valid.\n")

    N = input_control_intsDim(N, "N", 2)

    U = haar_measure(N)

    if txt is True:
        print(f"\nA new {N} x {N} random unitary matrix has been generated.")

    if file_output is True:
        matrix_file = open(filename + ".txt", "w+")

        np.savetxt(matrix_file, U, delimiter=",")

        print("\nThe new matrix is found in the file '" + filename + ".txt'.\n")

        matrix_file.close()

    return U


# This function below generates random non-unitary matrices. It is useful in the design of
# quasiunitary S matrices with loss
def RandM(file_output=True, filename=False, N1=False, N2=False, txt=False):
    if txt is True:
        print("\n\nRANDOM NON-UNITARY MATRIX GENERATOR (dim n x m):\n")

    while file_output is True and filename is False:
        print("\nWARNING: a new filename is required.")

        try:
            filename = input("Write the name of the file (without .txt extension): ")

        except ValueError:
            print("The given value is not valid.\n")

    N1 = input_control_ints(N1, "N1", 1)

    N2 = input_control_ints(N2, "N2", 1)

    U = matrix_generation_general_auto(N1, N2)

    if txt is True:
        print(f"\nA new {N1} x {N2} random matrix has been generated.")

    if file_output is True:
        matrix_file = open(filename + ".txt", "w+")

        np.savetxt(matrix_file, U, delimiter=",")

        print("\nThe new matrix is found in the file '" + filename + ".txt'.\n")

        matrix_file.close()

    return U


# Creation of discrete Fourier transformation matrices
def DFT(file_output=True, filename=False, N=False, txt=False):
    if txt is True:
        print("\n\nDFT MATRIX GENERATOR (dim N x N):\n")

    while file_output is True and filename is False:
        print("\nWARNING: a new filename is required.")

        try:
            filename = input("Write the name of the file (without .txt extension): ")

        except ValueError:
            print("The given value is not valid.\n")

    N = input_control_intsDim(N, "N", 2)

    omega = np.exp(-2.0 * math.pi * 1j / float(N))

    A = np.zeros((N, N), dtype=complex)

    for i in range(N):
        for j in range(N):
            A[i, j] = omega ** (i * j)

    if txt is True:
        print(f"\nA new {N} x {N} DFT matrix has been generated.")

    if file_output is True:
        matrix_file = open(filename + ".txt", "w+")

        np.savetxt(matrix_file, A, delimiter=",")

        print("\nThe new matrix is found in the file 'dft_matrix.txt'.\n")

        matrix_file.close()

    return A


# Creation of Quantum Fourier transformation matrices
def QFT(file_output=True, filename=False, N=False, inverse=False, txt=False):
    if txt is True:
        print("\n\nQFT MATRIX GENERATOR (dim N x N):\n")

    while file_output is True and filename is False:
        print("\nWARNING: a new filename is required.")

        try:
            filename = input("Write the name of the file (without .txt extension): ")

        except ValueError:
            print("The given value is not valid.\n")

    N = input_control_intsDim(N, "N", 2)

    A = np.zeros((N, N), dtype=complex)

    if inverse is True:
        for i in range(N):
            for j in range(N):
                A[i, j] = np.exp(-2.0 * math.pi * 1j / float(N) * i * j)

        if txt is True:
            print(f"\nA new {N} x {N} inverse QFT matrix has been generated.")

    else:
        for i in range(N):
            for j in range(N):
                A[i, j] = np.exp(2.0 * math.pi * 1j / float(N) * i * j)

        if txt is True:
            print(f"\nA new {N} x {N} QFT matrix has been generated.")

    if file_output is True:
        matrix_file = open(filename + ".txt", "w+")

        np.savetxt(matrix_file, A / np.sqrt(N), delimiter=",")

        print("\nThe new matrix is found in the file '" + filename + ".txt'.\n")

        matrix_file.close()

    return A / np.sqrt(N)


def RandImU(file_output=True, filename=False, m=False, n=False, txt=False):
    if txt is True:
        print("\n\nImU MATRIX GENERATOR (dim M x M):\n")

    while file_output is True and filename is False:
        print("\nWARNING: a new filename is required.")

        try:
            filename = input("Write the name of the file (without .txt extension): ")

        except ValueError:
            print("The given value is not valid.\n")

    # Initial input control
    m = input_control_intsDim(m, "m", 2)

    n = input_control_ints(n, "n", 1)

    M = hilbert_dim(m, n)

    S = RandU(file_output=False, filename=False, N=m, txt=False)

    ImU = StoU(file_input=False, S=S, file_output=False, filename=False, method=2, n=n, txt=False)[
        0
    ]

    if txt is True:
        print(f"\nA new {M} x {M} ImU matrix has been generated.")

    if file_output is True:
        matrix_file = open(filename + ".txt", "w+")

        np.savetxt(matrix_file, ImU, delimiter=",")

        print("\nThe new matrix is found in the file '" + filename + ".txt'.\n")

        matrix_file.close()

    return ImU
