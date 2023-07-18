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


# ----------COMBINATORY:----------

from ..recur_factorial import comb_evol_no_reps


# ---------------------------------------------------------------------------------------------------------------------------
#											RYSER PERMANENT COMBINATORY LOOPS
# ---------------------------------------------------------------------------------------------------------------------------


# The main algorithm. Generates a series of iterations for both given matrix dimensions N and subtracted columns col
def loop(N,col,j):

	# Global variables, applied to the whole loop
	global cont
	global itt
	global base

	# Keep in mind each execution of loop will concurr in the next column
	cont+=1

	# 'looped' loop: there will be as much loops as columns are available
	if cont<(N-col): 

		# Case: first loop/column
		if cont==0:

			for i in range(j,col+1+cont):

				# Indexation
				for k in range(comb_evol_no_reps(N-1-i,N-col-1-cont)):

					itt[base[cont]+k,cont]=i

				# Addition to next index positioning
				base[cont]+=comb_evol_no_reps(N-1-i,N-col-1-cont)

				# Next column exploration
				loop(N,col,i)

				# Flow control: column reinitialization
				cont=0

		# Case: other loops/columns
		else:

			for i in range(j+1,col+1+cont):

				# Indexation
				for k in range(comb_evol_no_reps(N-1-i,N-col-1-cont)):

					itt[base[cont]+k,cont]=i

				# Addition to next index positioning
				base[cont]+=comb_evol_no_reps(N-1-i,N-col-1-cont)

				# Next column exploration
				loop(N,col,i)

				# Flow control: stopping the loops and returning to previous loop/column
				if i==col+cont:

					cont-=1

					break

	# Flow control: returning to previous loop/column
	else:

		cont-=1


def ryser_loop(N,col):

	# Global variables, applied to the whole loop
	global cont
	global itt
	global base

	# Loop initiation
	cont=-1 # Loop inner counter
	itt=np.zeros((comb_evol_no_reps(N,col),N-col),dtype=int) # Iterations array. Dimensions depend on matrix dimensions N
														# and subtracted columns col. 
	base=np.zeros(N-col,dtype=int) # itt indexes positioning array

	# Main algorithm
	loop(N,col,0) # 0 = starting column

	return itt