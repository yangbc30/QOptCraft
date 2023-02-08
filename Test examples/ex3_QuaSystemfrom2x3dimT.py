# ---------------------------------------------------------------------------------------------------------------------------
#													QOPTCRAFT TRIAL 1
# ---------------------------------------------------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------------------------------------------------
#													LIBRARIES REQUIRED
# ---------------------------------------------------------------------------------------------------------------------------

# ----------QOptCraft: MAIN ALGORITHM----------

from QOptCraft.Main_Code import *


# ---------------------------------------------------------------------------------------------------------------------------
#														SYSTEM CHECK
# ---------------------------------------------------------------------------------------------------------------------------

import platform

print('Version      :', platform.python_version())
print('Version tuple:', platform.python_version_tuple())
print('Compiler     :', platform.python_compiler())
print('Build        :', platform.python_build())

print('Normal :', platform.platform())
print('Aliased:', platform.platform(aliased=True))
print('Terse  :', platform.platform(terse=True))


# ---------------------------------------------------------------------------------------------------------------------------
#														MAIN CODE
# ---------------------------------------------------------------------------------------------------------------------------

# Information of the global function
help(QOptCraft)


# Quasiunitary system S from random N1=2 x N2=3 matrix T

# We first generate the random matrix:
QOptCraft(filename="T_dim2x3",module=7,choice=1,N1=2,N2=3)

# Obtaination of "T_dim2x3.txt"'s quasiunitary representation S.
QOptCraft(file_input=True,filename="T_dim2x3",
	newfile=False,file_output=True,module=5)