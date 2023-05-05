# ---------------------------------------------------------------------------------------------------------------------------
# 													QOPTCRAFT TRIAL
# ---------------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------------------
# 													LIBRARIES REQUIRED
# ---------------------------------------------------------------------------------------------------------------------------

# ----------QOptCraft: MAIN ALGORITHM----------

from QOptCraft.Main_Code import *


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
help(QOptCraft)


# NOTE: 'exArt5MatrixRowsTest.txt' needs to be in this .py file's directory.
# Normally, it should be there already.

# 'Rotated_toponogov_1' is generated in ExAux3EntangMatrix

state_basis = subspace_basis(
    5, [1, 0, 1, 1, 1], [[0, 1, 1, 1, 1], [1, 0, 1, 1, 1], [1, 1, 0, 1, 1], [1, 1, 1, 0, 1], [1, 1, 1, 1, 0]]
)

# Measurement of three vectors in the Fock basis's Schmidt rank after passing through U.
QOptCraft(
    module=6,
    file_input_state=True,
    file_input_matrix=True,
    filename_state="exArt5MatrixRowsTest",
    filename_matrix="Rotated_toponogov_1",
    vec_base=state_basis,
    base_input=False,
    acc_d=2,
    txt=False,
    fidelity=0.99,
)
