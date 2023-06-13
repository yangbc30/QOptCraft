import time

import numpy as np
from numpy.linalg import inv
from scipy.linalg import logm, schur, sqrtm
from sympy import Matrix

from QOptCraft.utils.input_control import input_control

from QOptCraft.legacy.read_matrix import read_matrix_from_txt


def Logm1M(file_input=True, A=False, file_output=True, filename=False, txt=False, acc_d=3):
    file_input, filename, filler, acc_d = input_control(
        10, file_input, A, file_output, filename, txt, acc_d, False
    )

    if txt is True:
        print("\n\nMATRIX LOGARTHM 1\n")

    while file_input is True and filename is False:
        print("\nWARNING: a new filename is required.")

        try:
            filename = input("Write the name of the file (without .txt extension): ")

        except ValueError:
            print("The given value is not valid.\n")

    if file_input is True:
        A = read_matrix_from_txt(filename)

    if txt is True:
        print("\nInput matrix:")

        print(np.round(A, acc_d))

        # Beginning of time measurement
        t = time.process_time_ns()

    # Schur decomposition
    W, T = Matrix(A).diagonalize()
    W = np.array(W, dtype=complex)
    T = np.array(T, dtype=complex)

    N = len(A)

    # D diagonal matrix initialization
    D = np.zeros((N, N), dtype=complex)

    # No-null values computation
    for i in range(N):
        D[i, i] = T[i, i] / abs(T[i, i])

    # Matrix logarithm computation
    H = W.dot(logm(D).dot(np.linalg.inv(W)))
    logm_1A = 0.5 * (H + np.transpose(np.conj(H)))

    if file_output is True:
        with open(filename + "_Logm1.txt", "w+") as matrix_file:
            np.savetxt(matrix_file, logm_1A, delimiter=",")
            print("\nThe new matrix is found in the file '" + filename + "_Logm1.txt'.\n")

    if txt is True:
        print(f"\nThe logarithm Logm1({filename}) has been computed.")

        print("\nOutput matrix:")

        print(np.round(logm_1A, acc_d))

        # Total time of execution
        t_inc = time.process_time_ns() - t

        print(f"\nLogm1M: total time of execution (seconds): {float(t_inc/(10**(9)))}\n")

    return logm_1A


def Logm2M(file_input=True, A=False, file_output=True, filename=False, txt=False, acc_d=3):
    file_input, filename, filler, acc_d = input_control(
        10, file_input, A, file_output, filename, txt, acc_d, False
    )

    if txt is True:
        print("\n\nMATRIX LOGARTHM 2\n")

    while file_input is True and filename is False:
        print("\nWARNING: a new filename is required.")

        try:
            filename = input("Write the name of the file (without .txt extension): ")

        except ValueError:
            print("The given value is not valid.\n")

    if file_input is True:
        A = read_matrix_from_txt(filename)

    if txt is True:
        print("\nInput matrix:")

        print(np.round(A, acc_d))

        # Beginning of time measurement
        t = time.process_time_ns()

    # Matrix logarithm computation
    logm_2A = 0.5 * (logm(A) + np.transpose(np.conj(logm(A))))

    if file_output is True:
        with open(filename + "_Logm2.txt", "w+") as matrix_file:
            np.savetxt(matrix_file, logm_2A, delimiter=",")
            print("\nThe new matrix is found in the file '" + filename + "_Logm2.txt'.\n")

    if txt is True:
        print(f"\nThe logarithm Logm2({filename}) has been computed.")

        print("\nOutput matrix:")

        print(np.round(logm_2A, acc_d))

        # Total time of execution
        t_inc = time.process_time_ns() - t

        print(f"\nLogm2M: total time of execution (seconds): {float(t_inc/(10**(9)))}\n")

    return logm_2A


def Logm3M(file_input=True, A=False, file_output=True, filename=False, txt=False, acc_d=3):
    file_input, filename, filler, acc_d = input_control(
        10, file_input, A, file_output, filename, txt, acc_d, False
    )

    if txt is True:
        print("\n\nMATRIX LOGARTHM 3\n")

    while file_input is True and filename is False:
        print("\nWARNING: a new filename is required.")

        try:
            filename = input("Write the name of the file (without .txt extension): ")

        except ValueError:
            print("The given value is not valid.\n")

    if file_input is True:
        A = read_matrix_from_txt(filename)

    if txt is True:
        print("\nInput matrix:")

        print(np.round(A, acc_d))

        # Beginning of time measurement
        t = time.process_time_ns()

    # Schur decomposition
    U, Q = schur(A)

    N = len(A)

    # D diagonal matrix initialization
    D = np.zeros((N, N), dtype=complex)

    # No-null values computation
    for i in range(N):
        D[i, i] = U[i, i] / abs(U[i, i])

    # Matrix logarithm computation
    logm_3A = Q.dot(logm(D).dot(np.transpose(np.conj(Q))))

    if file_output is True:
        with open(filename + "_Logm3.txt", "w+") as matrix_file:
            np.savetxt(matrix_file, logm_3A, delimiter=",")
            print("\nThe new matrix is found in the file '" + filename + "_Logm3.txt'.\n")

    if txt is True:
        print(f"\nThe logarithm Logm3({filename}) has been computed.")

        print("\nOutput matrix:")

        print(np.round(logm_3A, acc_d))

        # Total time of execution
        t_inc = time.process_time_ns() - t

        print(f"\nLogm3M: total time of execution (seconds): {float(t_inc/(10**(9)))}\n")

    return logm_3A


def Logm4M(file_input=True, A=False, file_output=True, filename=False, txt=False, acc_d=3):
    file_input, filename, filler, acc_d = input_control(
        10, file_input, A, file_output, filename, txt, acc_d, False
    )

    if txt is True:
        print("\n\nMATRIX LOGARTHM 4\n")

    while file_input is True and filename is False:
        print("\nWARNING: a new filename is required.")

        try:
            filename = input("Write the name of the file (without .txt extension): ")

        except ValueError:
            print("The given value is not valid.\n")

    if file_input is True:
        A = read_matrix_from_txt(filename)

    if txt is True:
        print("\nInput matrix:")

        print(np.round(A, acc_d))

        # Beginning of time measurement
        t = time.process_time_ns()

    # Defining formula of this logarithm algorithm
    V = A.dot(inv(sqrtm(np.transpose(np.conj(A)).dot(A))))

    # Schur decomposition
    U, Q = schur(V)

    N = len(A)

    # D diagonal matrix initialization
    D = np.zeros((N, N), dtype=complex)

    # No-null values computation
    for i in range(N):
        D[i, i] = U[i, i] / abs(U[i, i])

    # Matrix logarithm computation
    logm_4A = Q.dot(logm(D).dot(np.transpose(np.conj(Q))))

    if file_output is True:
        with open(filename + "_Logm4.txt", "w+") as matrix_file:
            np.savetxt(matrix_file, logm_4A, delimiter=",")
            print("\nThe new matrix is found in the file '" + filename + "_Logm4.txt'.\n")

    if txt is True:
        print(f"\nThe logarithm Logm4({filename}) has been computed.")

        print("\nOutput matrix:")

        print(np.round(logm_4A, acc_d))

        # Total time of execution
        t_inc = time.process_time_ns() - t

        print(f"\nLogm4M: total time of execution (seconds): {float(t_inc/(10**(9)))}\n")

    return logm_4A


def Logm5M(file_input=True, A=False, file_output=True, filename=False, txt=False, acc_d=3):
    file_input, filename, filler, acc_d = input_control(
        10, file_input, A, file_output, filename, txt, acc_d, False
    )

    if txt is True:
        print("\n\nMATRIX LOGARTHM 5\n")

    while file_input is True and filename is False:
        print("\nWARNING: a new filename is required.")

        try:
            filename = input("Write the name of the file (without .txt extension): ")

        except ValueError:
            print("The given value is not valid.\n")

    if file_input is True:
        A = read_matrix_from_txt(filename)

    if txt is True:
        print("\nInput matrix:")

        print(np.round(A, acc_d))

        # Beginning of time measurement
        t = time.process_time_ns()

    # Defining formula of this logarithm algorithm
    V1 = (A + np.transpose(np.conj(inv(A)))) / 2.0

    V = (V1 + np.transpose(np.conj(inv(V1)))) / 2.0

    # Schur decomposition
    U, Q = schur(V)

    N = len(A)

    # D diagonal matrix initialization
    D = np.zeros((N, N), dtype=complex)

    # No-null values computation
    for i in range(N):
        D[i, i] = U[i, i] / abs(U[i, i])

    # Matrix logarithm computation
    logm_5A = Q.dot(logm(D).dot(np.transpose(np.conj(Q))))

    if file_output is True:
        with open(filename + "_Logm5.txt", "w+") as matrix_file:
            np.savetxt(matrix_file, logm_5A, delimiter=",")
            print("\nThe new matrix is found in the file '" + filename + "_Logm5.txt'.\n")

    if txt is True:
        print(f"\nThe logarithm Logm5({filename}) has been computed.")

        print("\nOutput matrix:")

        print(np.round(logm_5A, acc_d))

        # Total time of execution
        t_inc = time.process_time_ns() - t

        print(f"\nLogm5M: total time of execution (seconds): {float(t_inc/(10**(9)))}\n")

    return logm_5A
