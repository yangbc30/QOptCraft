'''Copyright 2021 Daniel Gómez Aguado

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.'''

# ---------------------------------------------------------------------------------------------------------------------------
#													LIBRARIES REQUIRED
# ---------------------------------------------------------------------------------------------------------------------------


# ----------MATHEMATICAL FUNCTIONS/MATRIX OPERATIONS:----------

# NumPy instalation: in the cmd: 'py -m pip install numpy'
import numpy as np


# ----------FILE MANAGEMENT:----------

# File opening
from io import open 


# ---------------------------------------------------------------------------------------------------------------------------
#											DECOMPOSITION OF MATRIX D FUNCTIONS
# ---------------------------------------------------------------------------------------------------------------------------


def D_decomposition(M,maxDim,filename,file_output,txt=False):

	if file_output==True:

		DList_file=open(filename+"_DList.txt","w")

	DList=np.zeros((maxDim,maxDim,maxDim), dtype=complex)

	for i in range(0,maxDim):

		I=np.identity(maxDim,dtype=complex)

		# Matrix D_i creation consists on replacing the identity
		# matrix element [i,i] for D_i of the original matrix D (here, M) 
		I[i,i]=M[i,i]

		DList[i,:,:]=I

		if file_output==True:

			np.savetxt(DList_file,DList[i,:,:],delimiter=",")
		
			DList_file.write("\n")

	if file_output==True:

		DList_file.close()

		if txt==True:

			print(f"\nThe list of matrices DList has been storaged in the file '"+filename+"_DList.txt'.")

	return DList