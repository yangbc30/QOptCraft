# ---------------------------------------------------------------------------------------------------------------------------
#													QOPTCRAFT TRIAL
# ---------------------------------------------------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------------------------------------------------
#													LIBRARIES REQUIRED
# ---------------------------------------------------------------------------------------------------------------------------

# ----------QOptCraft: MAIN ALGORITHM----------

from QOptCraft.Main_Code import *


# ---------------------------------------------------------------------------------------------------------------------------
#														MAIN CODE
# ---------------------------------------------------------------------------------------------------------------------------

import platform

print('Version      :', platform.python_version())
print('Version tuple:', platform.python_version_tuple())
print('Compiler     :', platform.python_compiler())
print('Build        :', platform.python_build())

print('Normal :', platform.platform())
print('Aliased:', platform.platform(aliased=True))
print('Terse  :', platform.platform(terse=True))

# Information of the global and Toponogov functions
help(QOptCraft)
help(Toponogov)


# We declare a variable containing our state basis, using auxiliar, more advanced functions not belonging to QOptCraft()
state_basis=subspace_basis(5,[1,0,1,1,1],[[0,1,1,1,1],[1,0,1,1,1],[1,1,0,1,1],[1,1,1,0,1],[1,1,1,1,0]])
 
# Toponogov() execution (you may also do QOptCraft(module=4,M_input=RotMat(70,1)) and the remainding parameters, same as below)
Toponogov(file_input=False,U_input=RotMat(70,1),file_output=True,filename="Rotated",
	tries=10,m=5,n=4,acc_d=3,txt=False,acc_t=3,vec_base=state_basis)

# How to physically build the Toponogov matrix used:
# Generation of S matrix from U (you may also do QOptCraft(module=2) and the remainding parameters, same as below)
SfromU(file_input=True,filename=filename,file_output=True,m=5,n=4)

# Decomposition of S (you may also do QOptCraft(module=1) and the remainding parameters, same as below)
Selements(file_input=True,file_output=True,newfile=False,impl=0,filename=filename+'_m_5_n_4_S_recon_main')