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

# Information of the global function
help(QOptCraft)
help(QFT)
help(SfromU)
help(Toponogov)
help(Selements)


# QFT_matrix (N=6)

# We first generate the 6 x 6 QFT matrix:
QFT(filename="QFT_matrix_6", N=6)

# Is the original matrix aleady plausible?
SfromU(file_input=True, filename="QFT_matrix_6", txt=True, acc_d=2, m=3, n=2)

# Getting "QFT_matrix_6.txt"'s closest evolution matrix U.
Toponogov(file_input=True, filename="QFT_matrix_6", base_input=False, file_output=True, m=3, n=2, tries=20)

# Getting "QFT_matrix_6_toponogov_3.txt"'s S-matrix.
SfromU(file_input=True, filename="QFT_matrix_6_toponogov_3", file_output=True, m=3, n=2)

# Decomposition of "QFT_matrix_6_toponogov_3.txt's S-matrix".
Selements(
    file_input=True, file_output=True, newfile=False, impl=0, filename="QFT_matrix_6_toponogov_3_m_3_n_2_S_recon_main"
)
