# ---------------------------------------------------------------------------------------------------------------------------
# 													QOPTCRAFT TRIAL 1
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


# QFT_matrix (N=3)

# We first generate the 3 x 3 QFT matrix:
QOptCraft(filename="QFT_matrix_3", module=7, choice=5, N=3)

# Is the original matrix aleady plausible?
QOptCraft(file_input=True, filename="QFT_matrix_3", txt=True, acc_d=2, module=3, m=2, n=2)

# Obtaination of "QFT_matrix_3.txt"'s closest evolution matrix U.
QOptCraft(
    file_input=True,
    filename="QFT_matrix_3",
    base_input=False,
    file_output=True,
    module=4,
    m=2,
    n=2,
    tries=20,
)

# Obtaination of "QFT_matrix_3_toponogov_2.txt"'s S-matrix.
QOptCraft(
    file_input=True,
    filename="QFT_matrix_3_toponogov_2",
    file_output=True,
    module=3,
    m=2,
    n=2,
    perm=True,
)

# Decomposition of "QFT_matrix_3_toponogov_2.txt's S-matrix".
QOptCraft(
    file_input=True,
    file_output=True,
    module=1,
    newfile=False,
    impl=0,
    filename="QFT_matrix_3_toponogov_2_m_2_n_2_S_recon_main",
)
