from itertools import permutations

# NumPy instalation: in the cmd: 'py -m pip install numpy'
import numpy as np

from qoptcraft.basis import hilbert_dim, get_algebra_basis

from qoptcraft._legacy.input_control import input_control, input_control_ints, input_control_intsDim
from qoptcraft._legacy.Phase3_Aux._3_permutation_matrix import *
from qoptcraft._legacy.Phase3_Aux._3_S_rebuild import S_output
from qoptcraft._legacy.Phase3_Aux.get_algebra_basis_legacy import (
    matrix_u_basis_generator,
)
from qoptcraft._legacy.Phase3_Aux._3_verification_of_solution_existence import (
    eq_sys_finder,
    verification,
)
from qoptcraft._legacy.read_matrix import read_matrix_from_txt
from qoptcraft._legacy.recur_factorial import *
from qoptcraft._legacy.unitary import *


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
    modes = m
    photons = n

    basis, basis_image = get_algebra_basis(modes, photons)

    # We obtain the equation system
    eq_sys, eq_sys_choice, index_choice = eq_sys_finder(basis, basis_image)

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

    S = False

    # This algorithm has the same function as the previous if, but for numerous permutations of
    # the basis vectors
    if perm is True:
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

    return S, S_perm
