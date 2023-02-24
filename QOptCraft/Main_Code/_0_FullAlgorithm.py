# ---------------------------------------------------------------------------------------------------------------------------
#											QOPTCRAFT LIBRARY STANDARIZATION
# ---------------------------------------------------------------------------------------------------------------------------

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


# ----------COMBINATORY:----------

from ..recur_factorial import comb_evol


# ----------INPUT CONTROL:----------

from ..input_control import input_control, input_control_ints, input_control_intsDim, input_control_floats


# ----------INITIAL MATRIX GENERATOR:----------

from ..write_initial_matrix import *


# ----------ALGORITHM 1: AUXILIAR FUNCTIONS:----------

from ._1_Unitary_matrix_U_builder import Selements 


# ----------ALGORITHM 5: AUXILIAR FUNCTIONS:----------

from ._5_Quasiunitary_S_with_or_without_loss_builder import QuasiU


# ----------ALGORITHM 2: AUXILIAR FUNCTIONS:----------

from ._2_Get_U_matrix_of_photon_system_evolution import *


# ----------ALGORITHM 9: AUXILIAR FUNCTIONS:----------

from ._9_friendly_logarithm_algorithms import *


# ----------ALGORITHM 2_aux_a: AUXILIAR FUNCTIONS:----------

from ._2_aux_a_computation_time_evolutions_comparison import StoUEvolComp


# ----------ALGORITHM 2_aux_b: AUXILIAR FUNCTIONS:----------

from ._2_aux_b_logarithm_algorithms_equalities import MatLogCompV

from ._2_aux_c_logarithm_algorithms_timesanderror import MatLogCompTnE


# ----------ALGORITHM 3: AUXILIAR FUNCTIONS:----------

from ._3_Get_S_from_U_Inverse_problem import SfromU


# ----------ALGORITHM 4: AUXILIAR FUNCTIONS:----------

from ._4_toponogov_theorem_for_uncraftable_matrices_U import Toponogov


# ----------ALGORITHM 6: AUXILIAR FUNCTIONS:----------

from ._6_schmidt_entanglement_measurement_of_states import StateSchmidt

from ..Phase6_Aux._6_basis_manipulations import *

from ..Phase6_Aux._6_schmidt import *

# ----------ALGORITHM 7: AUXILIAR FUNCTIONS:----------

from ._7_generators import *

# ----------PHOTON BASIS FUNCTIONS:----------

from ..photon_comb_basis import subspace_basis

# ---------------------------------------------------------------------------------------------------------------------------
#														MAIN CODE
# ---------------------------------------------------------------------------------------------------------------------------

# A function designed for testing QOptCraft's algorithms.
def QOCGen(file_output=True,filename=False,N=False,inverse=False,N1=False,N2=False,m=False,n=False,M=False,txt=False,choice=False):
	"""
	Allows the user to generate any type of matrix covered by QOptCraft individually, incluiding unitary, random, Discrete Fourier Transform and Quantum Fourier Transform matrices. The option to generate vector basis for Fock states (relevant in Phases 2 and upwards) and subalgebra u(m), U(M) matrices (Phases 3 and 4) is given as well.
	"""

	if choice!=2 and choice !=3:

		while file_output==True and filename==False:

			print(f"\nWARNING: a new/loaded filename is required.")

			try:

				filename=input("Write the name of the file (without .txt extension): ")
		
			except ValueError:

				print("The given value is not valid.\n")

	if type(choice) is not int:

		print(f"\nWARNING: invalid choice input (needs to be int).")

		while True:

			try:

				choice=int(input("\nGenerate...\n\nRandom unitary matrix via RandU: press '0' (or any other number not mentioned afterwards).\nRandom (complex) matrix via RandM: press '1'.\nFock states vector basis of photons via Fock: press '2'.\nMatrix basis of subalgebras u(m), u(M) via AlgBasis: press '3'.\nDFT matrix via DFT: press '4'.\nQFT matrix via QFT: press '5'.\nRandom unitary craftable evolution matrix ImU via RandImU: press '6'.\n"))

				break

			except ValueError:

				print("The given value is not valid.\n")

	if choice==1:

		N1=input_control_ints(N1,"N1",1)

		N2=input_control_ints(N2,"N2",1)

		# A new file 'filename.txt' containing a random N1 x N2 matrix T can be created
		# for its use in other processes
		T=RandM(file_output,filename,N1,N2,txt)

		return T

	elif choice==2:

		# Initial input control
		m=input_control_intsDim(m,"m",2)

		n=input_control_ints(n,"n",1)

		vec_base=Fock(file_output,m,n)

		return vec_base

	elif choice==3:

		# Initial input control
		m=input_control_intsDim(m,"m",2)
		
		n=input_control_ints(n,"n",1)

		M=input_control_intsDim(M,"n",2)

		base_u_m, base_u_M=AlgBasis(file_output,m,n,M)

		return base_u_m, base_u_M

	elif choice==4:

		N=input_control_intsDim(N,"N",2)

		# A new file 'filename.txt' containing an N-dimensional DFT matrix is created
		# so it can be used in other processes
		DFT_M=DFT(file_output,filename,N,txt)

		return DFT_M

	elif choice==5:

		N=input_control_intsDim(N,"N",2)

		# A new file 'filename.txt' containing an N-dimensional QFT matrix is created
		# for its use in other processes
		QFT_M=QFT(file_output,filename,N,inverse,txt)

		return QFT_M

	elif choice==6:

		# Initial input control
		m=input_control_intsDim(m,"m",2)

		n=input_control_ints(n,"n",1)

		ImU=RandImU(file_output,filename,m,n,txt)

		return ImU

	else:

		N=input_control_intsDim(N,"N",2)

		# A new file 'filename.txt' containing a random N-dimensional unitary matrix U can be created
		# for its use in other processes
		U_un=RandU(file_output,filename,N,txt)

		return U_un


# A function designed for testing QOptCraft's algorithms.
def QOCTest(file_output=True,m1=False,m2=False,n1=False,n2=False,N1=False,N2=False,tries=False,txt=False,choice=False,exp=False,vec_base=[[False,False],[False,False]],inverse=False,comparison_matrix='haar'):
	"""
	Contains functions centered about checking certain aspects of QOptCraft. For example, the validity of its logarithms, as well as speed comparisons between analogous algorithms.
	"""

	if type(choice) is not int:

		print(f"\nWARNING: invalid choice input (needs to be int).")

		while True:

			try:

				choice=int(input("\nInput '0' (or any other number not mentioned afterwards) for comparing StoU evolution methods' speed.\nInput '1' for checking equalities between the logarithm algorithms (mainly 3, 4 and 5).\nInput '2' for Excel comparisons of time and error between all matrix logarithm algorithms.\n"))

				break

			except ValueError:

				print("The given value is not valid.\n")

	if choice==1:

		# Initial input control
		if type(exp) is not int:

			print(f"\nWARNING: invalid exp input (needs to be int).")

			# We input the action of the index i over the series of dimensions
			while True:

				try:

					exp=int(input("\nWill the matrix logarithms be compared (press 1 or any other non mentioned number), or their exponentials (press 2)?\n"))

					if exp!=2:

						exp=1

					break

				except ValueError:

					print("\nThe given value is not valid.\n")

		if (type(N1) is not int) or (type(N2) is not int):

			print(f"\nWARNING: invalid N1 or N2 input (both need to be int).")
		
			while True:

				try:

					N1=int(input("\nInitial dimension of the loop? (it cannot be lower than 2): "))

					N2=int(input("\nFinal dimension of the loop? (it cannot be lower than the initial dimension): "))

					if N1<2 or N2<N1:

						print("\nThere is at least a given value not included in the possible domain.\n")

					else:

						break

				except ValueError:

					print("\nThe given value is not valid.\n")

		MatLogCompV(N1,N2,txt,exp)

	elif choice==2:

		# Initial input control
		if (type(N1) is not int) or (type(N2) is not int):

			print(f"\nWARNING: invalid N1 or N2 input (both need to be int).")
		
			while True:

				try:

					N1=int(input("\nInitial dimension of the loop? (it cannot be lower than 2): "))

					N2=int(input("\nFinal dimension of the loop? (it cannot be lower than the initial dimension): "))

					if N1<2 or N2<N1:

						print("\nThere is at least a given value not included in the possible domain.\n")

					else:

						break

				except ValueError:

					print("\nThe given value is not valid.\n")

			if type(exp) is not int:

				print(f"\nWARNING: invalid exp input (needs to be int).")

				# We input the action of the index i over the series of dimensions
				while True:

					try:

						exp=int(input("\nWill the matrix dimensions follow up in terms of i (press 1 or any other non mentioned number), or 2^i (press 2) for an index i?\n"))

						if exp!=2:

							exp=1

						break

					except ValueError:

						print("\nThe given value is not valid.\n")

		MatLogCompTnE(N1,N2,txt,exp)

	else:

		# Initial input control
		if (type(m1) is not int) or (type(m2) is not int):

			print(f"\nWARNING: invalid m1 or m2 input (both need to be int).")

			# We input the interval of dimensions to be computed by the algorithm
			while True:

				try:

					m1=int(input("\nInitial dimension of the loop? (it cannot be lower than 2): "))

					m2=int(input("\nFinal dimension of the loop? (it cannot be lower than the initial dimension): "))

					if m1<2 or m2<m1:

						print("\nThere is at least a given value not included in the possible domain.\n")

					else:

						break

				except ValueError:

					print("\nThe given value is not valid.\n")

		if (type(n1) is not int) or (type(n2) is not int):

			print(f"\nWARNING: invalid n1 or n2 input (both need to be int).")

			# We input the interval of number of photons to be computed by the algorithm
			while True:

				try:

					n1=int(input("\nInitial number of photons of the loop? (it cannot be lower than 1): "))

					n2=int(input("\nFinal number of photons of the loop? (it cannot be lower than the initial number of photons): "))

					if n1<1 or n2<n1:

						print("\nThere is at least a given value not included in the possible domain.\n")

					else:

						break

				except ValueError:

					print("\nThe given value is not valid.\n")

		tries=input_control_ints(tries,"tries",1)

		StoUEvolComp(file_output,m1,m2,n1,n2,txt,tries,vec_base,inverse,comparison_matrix)


# A function designed for testing QOptCraft's algorithms.
def QOCLog(file_input=True,A=False,file_output=True,filename=False,txt=False,acc_d=3,choice=False):
	"""
	Contains the different logarithm algorithms, Log{i}M (i being the numeration for the functions) developed for certain phases of the library.
	"""

	file_input, filename, filler, acc_d=input_control(10,file_input,A,file_output,filename,txt,acc_d,False)

	if type(choice) is not int:

		print(f"\nWARNING: invalid choice input (needs to be int).")

		while True:

			try:

				choice=int(input("\nInput '1' for Logm1M.\nInput '2' for Logm2M.\nInput '3' (or any other number not mentioned afterwards) for Logm3M.\nInput '4' for Logm4M.\nInput '5' for Logm5M.\n"))

				break

			except ValueError:

				print("The given value is not valid.\n")

	if choice==1:

		logmA=Logm1M(file_input,A,file_output,filename,txt,acc_d)

	elif choice==2:

		logmA=Logm2M(file_input,A,file_output,filename,txt,acc_d)

	elif choice==4:

		logmA=Logm4M(file_input,A,file_output,filename,txt,acc_d)

	elif choice==5:

		logmA=Logm5M(file_input,A,file_output,filename,txt,acc_d)

	else:

		logmA=Logm3M(file_input,A,file_output,filename,txt,acc_d)

	return logmA


# FULL ALGORITHM
def QOptCraft(module=False,file_input=True,M_input=False,file_output=True,filename=False,impl=0,
	newfile=True,N=False,method=2,m=False,N1=False,N2=False,m1=False,m2=False,n1=False,n2=False,base_input=False,n=False,perm=False,
	tries=False,txt=False,acc_d=3,acc_anc=8,omega=False,M=False,choice=False,inverse=False,exp=False,acc_t=8,vec_base=[[False,False],[False,False]],
	file_input_state=True,file_input_matrix=True,state_input=False,filename_state=False,filename_matrix=False,fidelity=0.95,comparison_matrix='haar'):
	'''
	The main function, making full use of all the algorithms available. 
	Its standalone subfunctions or blocks (read user guide) can be deployed on their own as well.
	Use the module parameter (1-10) for picking which subfunction to use: Selements (module=1), StoU (module=2),
	SfromU (module=3), Toponogov (module=4), QuasiU (module=5), QuasiHStoU (module=10), StateSchmidt (module=6).
	Use the choice parameter for subsubfunctions in QOCGen (module=7, choice=0-6), 
	QOCTest (module=8, choice=0-2) or QOCLog (module=9, choice=1-5).
	More info on the remaining parameters by reading QOptCraft's user guide.
	'''

	if txt==True:

		print("\n\n===========================================================")
		print("||| QOptCraft: BUILD A LINEAR OPTICS QUANTUM COMPUTATOR |||")
		print("===========================================================\n\n")

		print("Welcome to QOptCraft, a quantum mechanics computator builder.\n")

	if type(module) is not int:

		print("\nFirst of all, a module needs to be chosen.")

		if module!=False:

			print(f"\nWARNING: invalid module input (needs to be int).")

		while True:

			try:

				module=int(input("\nInput the right number for your algortihm of interest:\nSelements: '1' (or any other number not mentioned afterwards)\nStoU: '2'\nSfromU: '3'\nToponogov: '4'\nQuasiU: '5'\niHStoiHU: '6'\nQOCGen: '7'\nQOCTest: '8'\nQOCLog: '9'"))

				break

			except ValueError:

				print("The given value is not valid.\n")

	if module==7:

		M=QOCGen(file_output,filename,N,inverse,N1,N2,m,n,M,txt,choice)

		return M

	elif module==8:

		QOCTest(file_output,m1,m2,n1,n2,N1,N2,tries,txt,choice,exp,vec_base,inverse,comparison_matrix)

	elif module==6:

		file_input_state, filename_state, _, acc_d=input_control(module=6,file_input=file_input_state,M_input=M_input,
			file_output=file_output,filename=filename_state,txt=txt,acc_d=acc_d)
		file_input_matrix, filename_matrix, _, acc_d=input_control(module=6,file_input=file_input_matrix,M_input=M_input,
			file_output=file_output,filename=filename_matrix,txt=txt,acc_d=acc_d)

		StateSchmidt(file_input_state,file_input_matrix,state_input,M_input,file_output,filename_state,filename_matrix,base_input,vec_base,acc_d,txt,fidelity)

	else:

		file_input, filename, newfile, acc_d=input_control(module,file_input,M_input,file_output,filename,txt,acc_d,newfile)

		if module==2:

			U, vec_base=StoU(file_input,M_input,file_output,filename,method,n,acc_d,txt,vec_base)

			return U, vec_base

		elif module==3:

			S=SfromU(file_input,M_input,file_output,filename,base_input,m,n,perm,acc_d,txt)

			return S

		elif module==4:

			sol_array=Toponogov(file_input,M_input,file_output,filename,base_input,tries,m,n,acc_d,txt,acc_t,vec_base)

			return sol_array

		elif module==5:

			T, S, S_cut, UList, UD, WList, WD, D, DList=QuasiU(file_input,M_input,file_output,filename,newfile,N1,N2,acc_anc,acc_d,txt)

			return T, S, S_cut, UList, UD, WList, WD, D, DList

		elif module==10:

			iH_U, vec_base=iHStoiHU(file_input,M_input,file_output,filename,n,acc_d,txt)

			return iH_U, vec_base

		elif module==9:

			logmA=QOCLog(file_input,M_input,file_output,filename,txt,acc_d,choice)

			return logmA

		else:

			U_un, TmnList, D=Selements(file_input,M_input,file_output,filename,impl,newfile,N,acc_d,txt)

			return U_un, TmnList, D