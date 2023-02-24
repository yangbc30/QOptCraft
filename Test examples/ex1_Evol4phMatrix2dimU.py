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


# Evolution of S (m=2, n=4)

# We first generate and decompose the unitary scattering matrix in linear optic devices:
QOptCraft(
    file_input=False,
    filename="S_dim2",
    newfile=True,
    file_output=True,
    module=1,
    impl=0,
    N=2,
)

# Obtaination of "S_dim2.txt"'s evolution of n=4 photons U.
QOptCraft(file_input=True, filename="S_dim2", file_output=True, module=2, n=4, method=2)
