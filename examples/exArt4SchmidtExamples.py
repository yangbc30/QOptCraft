# ---------------------------------------------------------------------------------------------------------------------------
# 													QOPTCRAFT TRIAL
# ---------------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------------------
# 													LIBRARIES REQUIRED
# ---------------------------------------------------------------------------------------------------------------------------

# ----------QOptCraft: MAIN ALGORITHM----------

from QOptCraft.Main_Code import *


# ---------------------------------------------------------------------------------------------------------------------------
# 														MAIN CODE
# ---------------------------------------------------------------------------------------------------------------------------

import platform

print("Version      :", platform.python_version())
print("Version tuple:", platform.python_version_tuple())
print("Compiler     :", platform.python_compiler())
print("Build        :", platform.python_build())

print("Normal :", platform.platform())
print("Aliased:", platform.platform(aliased=True))
print("Terse  :", platform.platform(terse=True))


# In this code, we will compute an huge variety of different states' Schmidt rank.
# The automatic module SchmidtState() is thought to be executed for multiple vectors
# contained within the same space, so there we will show examples of usage with the functions
# it is composed of.

# W state
# We generate the vector basis with photon_combs_generator()
basisn1m3 = photon_combs_generator(3, [1, 0, 0])
# We create the vector W itself with state_in_basis()
W = state_in_basis(
    [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)], basisn1m3
)
print("\nW state:", W)
# Schmidt rank computation (modes per partition: [1,1,1], last argument)
print("Schmidt rank vector:", schmidt_rank_vector(W, basisn1m3, [1, 1, 1]))

# M state
# Let's try now with a 2-photon basis
basisn2m3 = photon_combs_generator(3, [1, 1, 0])
M = state_in_basis(
    [[1, 1, 0], [0, 1, 1], [1, 0, 1]], [1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)], basisn2m3
)
print("\nM state:", M)
print("Schmidt rank vector:", schmidt_rank_vector(M, basisn2m3, [1, 1, 1]))

# Bell state
# It will require of a 2-modes basis
basisn1m2 = photon_combs_generator(2, [1, 0])
Bell = state_in_basis([[1, 0], [0, 1]], [1 / np.sqrt(2), 1 / np.sqrt(2)], basisn1m2)
print("\nBell:", Bell)
print("Schmidt rank vector:", schmidt_rank_vector(Bell, basisn1m2, [1, 1]))

# Example of a very sligthly entangled state
eps = 1e-3
AlmostSeparable = state_in_basis([[1, 0], [0, 1]], [np.sqrt(1 - eps), np.sqrt(eps)], basisn1m2)
print("\nSlightly entangled state:", AlmostSeparable)
print("Schmidt rank vector:", schmidt_rank_vector(AlmostSeparable, basisn1m2, [1, 1]))

# Since this particular case isn't seen in a good light by keeping the full basis, let's consider a fidelity percentage
fidelity = 0.99

# We clean the state
states, weights = state_leading_fidelity(AlmostSeparable, basisn1m2, fidelity)
cleanstate = state_in_basis(states, weights, basisn1m2)
print("No good results. We clean the state:\nSlightly entangled state (leading terms):", cleanstate)
print("Schmidt rank vector:", schmidt_rank_vector(cleanstate, basisn1m2, [1, 1]))


# Higher dimensional entanglement PRL 116, 090405 (2016)

# 1/2(|000>+|101>+|210>+|311>)  OAM 4 (|0>,|1>,|2>,|3>),2 (|0>,|1>), 2 (|0>,|1>)
# --> 1/2(|0001>|01>|01>+|0010>|01>|10>+|0100>|10>|01>+|1000>|10>|10>)
basisn3m8 = photon_combs_generator(8, [1, 1, 1, 0, 0, 0, 0, 0])
State422 = state_in_basis(
    [
        [0, 0, 0, 1, 0, 1, 0, 1],
        [0, 0, 1, 0, 0, 1, 1, 0],
        [0, 1, 0, 0, 1, 0, 0, 1],
        [1, 0, 0, 0, 1, 0, 1, 0],
    ],
    [1 / 2, 1 / 2, 1 / 2, 1 / 2],
    basisn3m8,
)
print("\nState422:", State422)
print("Schmidt rank vector:", schmidt_rank_vector(State422, basisn3m8, [4, 2, 2]))
