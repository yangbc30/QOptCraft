"""API of the library
"""

from numpy.typing import NDArray
from scipy.linalg import dft


from qoptcraft.operators import haar_random_unitary, qft, qft_inv
from qoptcraft.evolution import photon_hamiltonian
from qoptcraft.basis import get_algebra_basis
from qoptcraft.entanglement import StateSchmidt
from qoptcraft.math.logarithms import logm_1, logm_2, logm_3, logm_4, logm_5

from qoptcraft._legacy.input_control import input_control, input_control_ints, input_control_intsDim
from qoptcraft.entanglement._6_basis_manipulations import *
from qoptcraft.entanglement._6_schmidt import *
from qoptcraft.optic_decomposition.unitary_decomp import Selements
from qoptcraft.evolution._2_aux_a_computation_time_evolutions_comparison import StoUEvolComp
from qoptcraft.evolution._2_aux_b_logarithm_algorithms_equalities import MatLogCompV
from qoptcraft.evolution._2_aux_c_logarithm_algorithms_timesanderror import MatLogCompTnE
from qoptcraft.evolution import StoU
from qoptcraft.evolution.inverse_problem import SfromU
from qoptcraft.topogonov import topogonov
from qoptcraft.operators.quasiunitary.builder import QuasiU
from qoptcraft.operators.other_matrices import *


# A function designed for testing qoptcraft's algorithms.
def QOCGen(
    file_output=True,
    filename=False,
    N=False,
    inverse=False,
    N1=False,
    N2=False,
    m=False,
    n=False,
    M=False,
    txt=False,
    choice=False,
):
    """
    Allows the user to generate any type of matrix covered by qoptcraft individually, incluiding unitary, random, Discrete Fourier Transform and Quantum Fourier Transform matrices. The option to generate vector basis for Fock states (relevant in Phases 2 and upwards) and subalgebra u(m), U(M) matrices (Phases 3 and 4) is given as well.
    """

    if choice != 2 and choice != 3:
        while file_output is True and filename is False:
            print("\nWARNING: a new/loaded filename is required.")

            try:
                filename = input("Write the name of the file (without .txt extension): ")

            except ValueError:
                print("The given value is not valid.\n")

    if type(choice) is not int:
        print("\nWARNING: invalid choice input (needs to be int).")

        while True:
            try:
                choice = int(
                    input(
                        "\nGenerate...\n\nRandom unitary matrix via RandU: press '0' (or any other number not mentioned afterwards).\nRandom (complex) matrix via RandM: press '1'.\nFock states vector basis of photons via Fock: press '2'.\nMatrix basis of subalgebras u(m), u(M) via AlgBasis: press '3'.\nDFT matrix via DFT: press '4'.\nQFT matrix via QFT: press '5'.\nRandom unitary craftable evolution matrix ImU via RandImU: press '6'.\n"
                    )
                )

                break

            except ValueError:
                print("The given value is not valid.\n")

    if choice == 1:
        N1 = input_control_ints(N1, "N1", 1)

        N2 = input_control_ints(N2, "N2", 1)

        # A new file 'filename.txt' containing a random N1 x N2 matrix T can be created
        # for its use in other processes
        T = RandM(file_output, filename, N1, N2, txt)

        return T

    elif choice == 2:
        # Initial input control
        m = input_control_intsDim(m, "m", 2)

        n = input_control_ints(n, "n", 1)

        vec_base = Fock(file_output, m, n)

        return vec_base

    elif choice == 3:
        # Initial input control
        modes = input_control_intsDim(m, "modes", 2)
        photons = input_control_ints(n, "photons", 1)

        basis, basis_image = get_algebra_basis(modes, photons)

        return basis, basis_image

    elif choice == 4:
        N = input_control_intsDim(N, "N", 2)
        return dft(N)

    elif choice == 5:
        N = input_control_intsDim(N, "N", 2)  # TODO: remove inputs
        return qft_inv(N) if inverse else qft(N)

    elif choice == 6:
        m = input_control_intsDim(m, "m", 2)
        n = input_control_ints(n, "n", 1)

        return RandImU(file_output, filename, m, n, txt)

    else:
        N = input_control_intsDim(N, "N", 2)

        # A new file 'filename.txt' containing a random N-dimensional unitary matrix U can be created
        # for its use in other processes
        U_un = haar_random_unitary(N)

        return U_un


# A function designed for testing qoptcraft's algorithms.
def QOCTest(
    file_output=True,
    m1=False,
    m2=False,
    n1=False,
    n2=False,
    N1=False,
    N2=False,
    tries=False,
    txt=False,
    choice=False,
    exp=False,
    vec_base=[[False, False], [False, False]],
    inverse=False,
    comparison_matrix="haar",
):
    """
    Contains functions centered about checking certain aspects of qoptcraft. For example, the validity of its logarithms, as well as speed comparisons between analogous algorithms.
    """

    if type(choice) is not int:
        print("\nWARNING: invalid choice input (needs to be int).")

        while True:
            try:
                choice = int(
                    input(
                        "\nInput '0' (or any other number not mentioned afterwards) for comparing StoU evolution methods' speed.\nInput '1' for checking equalities between the logarithm algorithms (mainly 3, 4 and 5).\nInput '2' for Excel comparisons of time and error between all matrix logarithm algorithms.\n"
                    )
                )

                break

            except ValueError:
                print("The given value is not valid.\n")

    if choice == 1:
        # Initial input control
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

        if (type(N1) is not int) or (type(N2) is not int):
            print("\nWARNING: invalid N1 or N2 input (both need to be int).")

            while True:
                try:
                    N1 = int(
                        input("\nInitial dimension of the loop? (it cannot be lower than 2): ")
                    )

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

        MatLogCompV(N1, N2, txt, exp)

    elif choice == 2:
        # Initial input control
        if (type(N1) is not int) or (type(N2) is not int):
            print("\nWARNING: invalid N1 or N2 input (both need to be int).")

            while True:
                try:
                    N1 = int(
                        input("\nInitial dimension of the loop? (it cannot be lower than 2): ")
                    )

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

        MatLogCompTnE(N1, N2, txt, exp)

    else:
        # Initial input control
        if (type(m1) is not int) or (type(m2) is not int):
            print("\nWARNING: invalid m1 or m2 input (both need to be int).")

            # We input the interval of dimensions to be computed by the algorithm
            while True:
                try:
                    m1 = int(
                        input("\nInitial dimension of the loop? (it cannot be lower than 2): ")
                    )

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
                        input(
                            "\nInitial number of photons of the loop? (it cannot be lower than 1): "
                        )
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

        tries = input_control_ints(tries, "tries", 1)

        StoUEvolComp(file_output, m1, m2, n1, n2, txt, tries, vec_base, inverse, comparison_matrix)


# A function designed for testing qoptcraft's algorithms.
def QOCLog(
    file_input=False,
    matrix: NDArray | None = None,
    file_output=True,
    filename=False,
    txt=False,
    acc_d=3,
    choice=False,
):
    """
    Contains the different logarithm algorithms, Log{i}M (i being the numeration for the functions) developed for certain phases of the library.
    """

    file_input, filename, filler, acc_d = input_control(
        10, file_input, matrix, file_output, filename, txt, acc_d, False
    )

    if type(choice) is not int:
        print("\nWARNING: invalid choice input (needs to be int).")

        while True:
            try:
                choice = int(
                    input(
                        "\nInput '1' for Logm1M.\nInput '2' for Logm2M.\nInput '3' (or any other number not mentioned afterwards) for Logm3M.\nInput '4' for Logm4M.\nInput '5' for Logm5M.\n"
                    )
                )

                break

            except ValueError:
                print("The given value is not valid.\n")

    if choice == 1:
        logmA = logm_1(matrix)

    elif choice == 2:
        logmA = logm_2(matrix)

    elif choice == 4:
        logmA = logm_4(matrix)

    elif choice == 5:
        logmA = logm_5(matrix)

    else:
        logmA = logm_3(matrix)

    return logmA


# FULL ALGORITHM
def QOptCraft(
    module=False,
    file_input=True,
    M_input=False,
    file_output=True,
    filename=False,
    impl=0,
    newfile=True,
    N=False,
    method=2,
    m=False,
    N1=False,
    N2=False,
    m1=False,
    m2=False,
    n1=False,
    n2=False,
    base_input=False,
    n=False,
    perm=False,
    tries=False,
    txt=False,
    acc_d=3,
    acc_anc=8,
    omega=False,
    M=False,
    choice=False,
    inverse=False,
    exp=False,
    acc_t=8,
    vec_base=None,
    file_input_state=True,
    file_input_matrix=True,
    state_input=False,
    filename_state=False,
    filename_matrix=False,
    fidelity=0.95,
    comparison_matrix="haar",
):
    """
    The main function, making full use of all the algorithms available.
    Its standalone subfunctions or blocks (read user guide) can be deployed
    on their own as well.
    Use the module parameter (1-10) for picking which subfunction to use:
    Selements (module=1), StoU (module=2),
    SfromU (module=3), Toponogov (module=4), QuasiU (module=5), QuasiHStoU (module=10),
    StateSchmidt (module=6).
    Use the choice parameter for subsubfunctions in QOCGen (module=7, choice=0-6),
    QOCTest (module=8, choice=0-2) or QOCLog (module=9, choice=1-5).
    More info on the remaining parameters by reading qoptcraft's user guide.
    """
    if vec_base is None:
        vec_base = [[False, False], [False, False]]

    if txt is True:
        print("\n\n===========================================================")
        print("||| qoptcraft: BUILD A LINEAR OPTICS QUANTUM COMPUTATOR |||")
        print("===========================================================\n\n")

        print("Welcome to qoptcraft, a quantum mechanics computator builder.\n")

    if type(module) is not int:
        print("\nFirst of all, a module needs to be chosen.")

        if module is not False:
            print("\nWARNING: invalid module input (needs to be int).")

        while True:
            try:
                module = int(
                    input(
                        "\nInput the right number for your algortihm of interest:\n"
                        "Selements: '1' (or any other number not mentioned afterwards)\n"
                        "StoU:'2'\nSfromU: '3'\nToponogov: '4'\nQuasiU: '5'\niHStoiHU: '6'\n"
                        "QOCGen: '7'\nQOCTest: '8'\nQOCLog: '9'"
                    )
                )

                break

            except ValueError:
                print("The given value is not valid.\n")

    if module == 7:
        M = QOCGen(file_output, filename, N, inverse, N1, N2, m, n, M, txt, choice)

        return M

    elif module == 8:
        QOCTest(
            file_output,
            m1,
            m2,
            n1,
            n2,
            N1,
            N2,
            tries,
            txt,
            choice,
            exp,
            vec_base,
            inverse,
            comparison_matrix,
        )

    elif module == 6:
        file_input_state, filename_state, _, acc_d = input_control(
            module=6,
            file_input=file_input_state,
            M_input=M_input,
            file_output=file_output,
            filename=filename_state,
            txt=txt,
            acc_d=acc_d,
        )
        file_input_matrix, filename_matrix, _, acc_d = input_control(
            module=6,
            file_input=file_input_matrix,
            M_input=M_input,
            file_output=file_output,
            filename=filename_matrix,
            txt=txt,
            acc_d=acc_d,
        )

        StateSchmidt(
            file_input_state,
            file_input_matrix,
            state_input,
            M_input,
            file_output,
            filename_state,
            filename_matrix,
            base_input,
            vec_base,
            acc_d,
            txt,
            fidelity,
        )

    else:
        file_input, filename, newfile, acc_d = input_control(
            module, file_input, M_input, file_output, filename, txt, acc_d, newfile
        )

        if module == 2:
            U, vec_base = StoU(
                file_input, M_input, file_output, filename, method, n, acc_d, txt, vec_base
            )

            return U, vec_base

        elif module == 3:
            S = SfromU(
                file_input, M_input, file_output, filename, base_input, m, n, perm, acc_d, txt
            )

            return S

        elif module == 4:
            sol_array = Toponogov(
                file_input,
                M_input,
                file_output,
                filename,
                base_input,
                tries,
                m,
                n,
                acc_d,
                txt,
                acc_t,
                vec_base,
            )

            return sol_array

        elif module == 5:
            T, S, S_cut, UList, UD, WList, WD, D, DList = QuasiU(
                file_input, M_input, file_output, filename, newfile, N1, N2, acc_anc, acc_d, txt
            )

            return T, S, S_cut, UList, UD, WList, WD, D, DList

        elif module == 10:
            iH_U, vec_base = photon_hamiltonian(M_input, n)

            return iH_U, vec_base

        elif module == 9:
            logmA = QOCLog(file_input, M_input, file_output, filename, txt, acc_d, choice)

            return logmA

        else:
            U_un, TmnList, D = Selements(
                file_input, M_input, file_output, filename, impl, newfile, N, acc_d, txt
            )

            return U_un, TmnList, D
