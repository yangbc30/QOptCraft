# ---------------------------------------------------------------------------------------------------------------------------
# 													QOPTCRAFT TRIAL
# ---------------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------------------
# 													LIBRARIES REQUIRED
# ---------------------------------------------------------------------------------------------------------------------------

# ----------qoptcraft: MAIN ALGORITHM----------

from qoptcraft.Main_Code import *


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
help(qoptcraft)
help(QuasiU)
help(RandM)


# Quasiunitary system S from random N1=2 x N2=2 matrix T
# NOTE: 'T_dim2x2.txt' needs to be in this .py file's directory.
# Normally, it should be there already.

# Obtaination of "T_dim2x2.txt"'s S-matrix.
QuasiU(file_input=True, filename="T_dim2x2", newfile=False, file_output=True)


# Quasiunitary system S from random N1=2 x N2=3 matrix M

# We first generate the random matrix:
RandM(filename="M_dim2x3", N1=2, N2=3)

# Obtaination of "M_dim2x3.txt"'s quasiunitary representation S.
QuasiU(file_input=True, filename="M_dim2x3", newfile=False, file_output=True)
