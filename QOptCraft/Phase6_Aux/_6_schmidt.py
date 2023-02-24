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
#                                                   LIBRARIES REQUIRED
# ---------------------------------------------------------------------------------------------------------------------------


# ----------MATHEMATICAL FUNCTIONS/MATRIX OPERATIONS:----------

# NumPy instalation: in the cmd: 'py -m pip install numpy'
import numpy as np


# ----------ALGORITHM 6: AUXILIAR FUNCTIONS:----------

from ._6_basis_manipulations import expand_basis, large_basis


# ---------------------------------------------------------------------------------------------------------------------------
#                                           SCHMIDT RANK FUNCTION
# ---------------------------------------------------------------------------------------------------------------------------

def schmidt_rank(state,subsystem1_basis,subsystem2_basis):
    
    row=0
    M=np.zeros(shape=(len(subsystem1_basis),len(subsystem2_basis)),dtype=complex)
    for element_bas1 in subsystem1_basis:
       column=0
       for element_bas2 in subsystem2_basis:
          M[row][column]= np.vdot(np.kron(element_bas1,element_bas2),state)#Matrix for the SVD  elements are projections of
          column=column+1
       row=row+1  

    u, s, vh = np.linalg.svd(M, full_matrices=True)
 
    return len(s)-np.sum(np.isclose(s,np.zeros_like(s)))                    
    # In the SVD decomposition, nonzero elements of S   numpy counzeros has problems with tolerances


def schmidt_rank_vector(state,basis,mvec):                 
    """
    Takes input states from the photonic Hilbert space into a larger space for entanglement evaluation 
    mvec gives the modes in each subsystem
    """

    n=int(np.sum(basis[0]))  # n+1-dimensional qudits
    m=len(basis[0])      #
    Ns=len(mvec)  #number of subsystems

    rank_vector=np.zeros(Ns)
    modes_in_subsystem=np.cumsum([0]+mvec)    #Adjusted to include 0 index
 
    for i in range(Ns):          # Go through each subsystem
       state_larger_space=np.array([1])  # Initializing the state for the Kronecker product

       subsystem1_basis=[]

       for k in range((n+1)**(mvec[i])):
           base_np1_string=np.base_repr(k,base=n+1)
           basis1_state= [int(digit) for digit in '0'*(mvec[i]-len(base_np1_string))+base_np1_string]  # Padded to m-1 digits in base n+1
           subsystem1_basis.append(large_basis(basis1_state,n,mvec[i]))

       subsystem2_basis=[]

       for k in range((n+1)**(m-mvec[i])):
           base_np1_string=np.base_repr(k,base=n+1)
           basis2_state= [int(digit) for digit in '0'*(m-mvec[i]-len(base_np1_string))+base_np1_string]  # Padded to m-1 digits in base n+1
           subsystem2_basis.append(large_basis(basis2_state,n,m-mvec[i]))

       ind=0 #index in the input state 
       state_larger_space=np.zeros_like(expand_basis(basis[0]))

       for phot_basis_element in basis:         
           small_subsystem=phot_basis_element[modes_in_subsystem[i]:modes_in_subsystem[i+1]]
           large_subsystem=np.delete(phot_basis_element, range(modes_in_subsystem[i],modes_in_subsystem[i+1]), axis=0)
           alpha_i=state[ind]
           # Iterates through the amplitudes
           state_larger_space=state_larger_space+alpha_i*np.kron(large_basis(small_subsystem,n,mvec[i]),large_basis(large_subsystem,n,m-mvec[i]))
           ind=ind+1

       #print(state_larger_space,subsystem1_basis,subsystem2_basis)
       #print(state_larger_space,len(subsystem1_basis),len(subsystem2_basis))

       rank_vector[i]=schmidt_rank(state_larger_space,subsystem1_basis,subsystem2_basis)
       
    return rank_vector        