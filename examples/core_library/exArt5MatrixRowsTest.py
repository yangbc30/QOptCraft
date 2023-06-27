# ---------------------------------------------------------------------------------------------------------------------------
#                                                   QOPTCRAFT TRIAL
# ---------------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------------------
#                                                   LIBRARIES REQUIRED
# ---------------------------------------------------------------------------------------------------------------------------

# ----------qoptcraft: MAIN ALGORITHM----------

from qoptcraft.Main_Code import *


# ---------------SYSTEM LIBRARIES--------------

import sys, itertools


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


U = read_matrix_from_txt(filename="Rotated_toponogov_1")  # Closest matrix

state_basis = subspace_basis(
    5,
    [1, 0, 1, 1, 1],
    [[0, 1, 1, 1, 1], [1, 0, 1, 1, 1], [1, 1, 0, 1, 1], [1, 1, 1, 0, 1], [1, 1, 1, 1, 0]],
)
inputs = []

# All possible inputs with 4 photons in separate modes
poszero = itertools.combinations(range(5), 4)
for positions in poszero:
    s = ["0"] * 5
    for bit in positions:
        s[bit] = "1"
    inputs.append([int(state) for state in "".join(s)])

print(f"Inputs: {inputs}\n")
# We will check, for each row of the matrix corresponding to inputs, its entanglement

# Fidelity parameter
fidelity = 0.99
for state in inputs:
    print(f"\nInitial state: {state}")
    input_state = state_in_basis(
        [state], np.array([1]), state_basis
    )  # The state in the Hilbert space basis we use
    output_state = np.matmul(U, input_state.T)

    print(
        np.count_nonzero(np.abs(output_state) ** 2),
        len(output_state[np.abs(output_state) ** 2 > 1e-5]),
        leading_terms(output_state, fidelity),
    )  # Total nonzero terms in the superposition, terms with more than a 10^-5 probability and relevant terms from the output in terms of the desired fidelity
    print(state_leading_fidelity(output_state, state_basis, fidelity))
    shortstate, weights = state_leading_fidelity(output_state, state_basis, fidelity)
    short = state_in_basis(shortstate, weights, state_basis)

    print(schmidt_rank_vector(short, state_basis, [1, 1, 1, 1, 1]))
    print(
        min(np.abs(weights) ** 2) / max(np.abs(weights))
    )  # "balance" of the state (ratio between the minimum and the maximum probability)
