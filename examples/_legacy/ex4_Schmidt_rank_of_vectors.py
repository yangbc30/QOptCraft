# ---------------------------------------------------------------------------------------------------------------------------
# 													QOPTCRAFT TRIAL
# ---------------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------------------
# 													LIBRARIES REQUIRED
# ---------------------------------------------------------------------------------------------------------------------------

# ----------qoptcraft: MAIN ALGORITHM----------

from qoptcraft.Main_Code import *


# ---------------------------------------------------------------------------------------------------------------------------
# 														SYSTEM CHECK
# ---------------------------------------------------------------------------------------------------------------------------

import platform

print("Version      :", platform.python_version())
print("Version tuple:", platform.python_version_tuple())
print("Compiler     :", platform.python_compiler())
print("Build        :", platform.python_build())

print("Normal :", platform.platform())
print("Aliased:", platform.platform(aliased=True))
print("Terse  :", platform.platform(terse=True))


# ---------------------------------------------------------------------------------------------------------------------------
# 														MAIN CODE
# ---------------------------------------------------------------------------------------------------------------------------

# Information of the global function
help(qoptcraft)


# Generation of a 6-mode, 4-photonic unitary matrix S 'S_dim6.txt'.

# We first generate and decompose the unitary scattering matrix in linear optic devices:
QOptCraft(
    file_input=False, filename="S_dim6", newfile=True, file_output=True, module=1, impl=0, N=6
)

# Obtaination of "S_dim6.txt"'s evolution of n=4 photons U.
# 'S_dim6_m_6_n_4_coefs_method_2.txt' is generated.
QOptCraft(file_input=True, filename="S_dim6", file_output=True, module=2, n=4, method=2)

# NOTE: '3_vectors_in_Fock_basis_for_Schmidt_measurement.txt' needs to be in this .py file's directory.
# Normally, it should be there already.

# Measurement of three vectors in the Fock basis's Schmidt rank after passing through U.
QOptCraft(
    module=6,
    file_input_state=True,
    file_input_matrix=True,
    filename_state="3_vectors_in_Fock_basis_for_Schmidt_measurement",
    filename_matrix="S_dim6_m_6_n_4_coefs_method_2",
    base_input=False,
    acc_d=2,
    txt=False,
    fidelity=0.8,
)
