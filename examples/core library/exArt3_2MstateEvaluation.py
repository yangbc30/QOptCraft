# ---------------------------------------------------------------------------------------------------------------------------
#                                                   QOPTCRAFT TRIAL
# ---------------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------------------
#                                                   LIBRARIES REQUIRED
# ---------------------------------------------------------------------------------------------------------------------------

# ----------qoptcraft: MAIN ALGORITHM----------

from qoptcraft.Main_Code import *


# ---------------------------------------------------------------------------------------------------------------------------
#                                                       MAIN CODE
# ---------------------------------------------------------------------------------------------------------------------------

import platform

print("Version      :", platform.python_version())
print("Version tuple:", platform.python_version_tuple())
print("Compiler     :", platform.python_compiler())
print("Build        :", platform.python_build())

print("Normal :", platform.platform())
print("Aliased:", platform.platform(aliased=True))
print("Terse  :", platform.platform(terse=True))

# Information of the global function
help(qoptcraft)
help(SfromU)
help(Selements)

## Main: Schmidt rank measurement of M state

# We load the matrix (one obtained through Toponogov in a previous experiment)
filename = "Rotated_toponogov_1"
U = read_matrix_from_txt(filename)

# We introduce an initial subspace basis, containing the elements we are interested in for our vector
state_basis = subspace_basis(
    5,
    [1, 0, 1, 1, 1],
    [[0, 1, 1, 1, 1], [1, 0, 1, 1, 1], [1, 1, 0, 1, 1], [1, 1, 1, 0, 1], [1, 1, 1, 1, 0]],
)

# State obtaination. Input: |rho> = 1*|1,1,1,1,0>. Output: will have a combination of plenty of elements in the Fock basis of 4 photons.
input_state = state_in_basis(
    [[1, 1, 1, 1, 0]], np.array([1]), state_basis
)  # The state in the Hilbert space basis we use
output_state = np.matmul(U, input_state.T)

# Fidelity parameter
fidelity = 0.99
# Obtaination of an approximated ket as the output
shortstate, weights = state_leading_fidelity(output_state, state_basis, fidelity)
short = state_in_basis(shortstate, weights, state_basis)

# Out of the five initial kets, the first two modes always carry a photon
print(shortstate)
print(weights)

# We may eliminate those from the operations since they don't bring new information to the table,
# Operations with the last three modes:
ThreeModes = []
for state in shortstate:
    ThreeModes.append([state[2:]])

# We generate a new basis for 2 photons and 3 modes (given the five initial kets we were interested in had their first two modes
# eliminated, and they always carried 2 out of the 4 photons)
basisn2m3 = photon_combs_generator(3, [1, 1, 0])
UnbalancedM = state_in_basis(ThreeModes, weights, basisn2m3)
print("Unbalanced M state:", UnbalancedM)
print("Schmidt rank vector:", schmidt_rank_vector(UnbalancedM, basisn2m3, [1, 1, 1]))
