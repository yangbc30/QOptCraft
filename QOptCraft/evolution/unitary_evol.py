import numpy as np

from qoptcraft.basis import get_photon_basis

from ._2_aux_a_computation_time_evolutions_comparison import *
from ._2_aux_b_logarithm_algorithms_equalities import *
from ._2_aux_c_logarithm_algorithms_timesanderror import *
from .hamiltonian_evol import *
from ._2_1st_evolution_method import *
from ._2_2nd_evolution_method import *
from ._2_3rd_evolution_method import *
from ._2_creation_and_destruction_operators import *


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


# Here, we will perform the third evolution of the system method's operations. Main function to inherit in other algorithms
def evolution_3(S, photons, vec_base, file_output=False, filename=False):
    # Initial time
    # It is required to introduce photons_aux for 'photons_aux' and 'photons' not to "update" together
    global photons_aux
    global mult

    m = len(S)
    photons = int(np.sum(photons))

    # Resulting U matrix's dimensions:
    M = hilbert_dim(m, photons)
    # This value could also be obtained by measuring vec_base's length

    # Out of the three logarithm algorithms developed in the main algorithm 2b, logm_3()
    # has been the one used. It can be switched by logm_4/5(); the result will be similar
    iH_S = logm_3(S)[0]

    if file_output is True:
        # We save the vector basis
        iH_S_file = open(f"{filename}_iH_S.txt", "w")

        np.savetxt(iH_S_file, iH_S, delimiter=",")

        iH_S_file.close()

    iH_U = photon_hamiltonian(iH_S, photons)

    # If the commentary of the following four lines is undone, the operator n will also be computed
    # and its conmutation with iH_U, which must exist, will be tested. It is by default omitted
    # for a faster pace

    U = expm(iH_U)

    return U, t_inc
