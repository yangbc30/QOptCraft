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

from numpy.lib.scimath import sqrt

# SciPy instalation: in the cmd: 'py -m pip install scipy'

# SymPy instalation: in the cmd: 'py -m pip install sympy'


# ---------------------------------------------------------------------------------------------------------------------------
#											SCATTERING MATRIX S RECONSTRUCTION
# ---------------------------------------------------------------------------------------------------------------------------


# Adjoint representation for a S matrix
def adjoint_S(index,base_u_m,sol):

	m=len(base_u_m[0])

	suma=np.zeros((m,m),dtype=complex)

	for j in range(m*m):

		suma+=sol[index,j]*base_u_m[j]

	return suma


# Main function of S rebuilding
def S_output(base_u_m,base_U_m,sol_e,sol_f):

	m=len(base_u_m[0])

	S=np.zeros((m,m),dtype=complex)

	# First of all, we obtain a no-null value of S for using it as a base for the rest computations
	for l in range(m):

		end=False

		for j in range(m):

			l_array=np.array([base_U_m[l]])

			absS=-1j*np.conj(l_array).dot(adjoint_S(m*j+j,base_u_m,sol_e).dot(np.transpose(l_array)))

			# 8 decimal accuracy, it can be modified
			if np.round(absS,8)==0:

				S[l,j]=0

			else:

				# We ignore the offset (for now...)

				l0=l
				j0=j

				end=True

				break

		if end:

			break

	# Later, we compute the total matrix. l0 y j0 serve as a support
	for l in range(m):

		for j in range(m):

			l0_array=np.array([base_U_m[l0]])

			l_array=np.array([base_U_m[l]])

			j_array=np.array([base_U_m[j]])

			# Storage of the sum in S
			S+=(np.conj(l_array).dot(adjoint_S(m*j+j0,base_u_m,sol_f).dot(np.transpose(l0_array)))-1j*np.conj(l_array).dot(adjoint_S(m*j+j0,base_u_m,sol_e).dot(np.transpose(l0_array))))/sqrt(-1j*np.conj(l0_array).dot(adjoint_S(m*j0+j0,base_u_m,sol_e).dot(np.transpose(l0_array))))*(np.transpose(l_array).dot(np.conj(j_array)))

	return S
