# ---------------------------------------------------------------------------------------------------------------------------
# 						ALGORITHM 6: ENTANGLEMENT VALUES OF STATES WITHIN AN UNITARY SYSTEM
# ---------------------------------------------------------------------------------------------------------------------------

"""Copyright 2021 Daniel Gómez Aguado

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License."""

# ---------------------------------------------------------------------------------------------------------------------------
# 													LIBRARIES REQUIRED
# ---------------------------------------------------------------------------------------------------------------------------


# ----------TIME OF EXECUTION MEASUREMENT:----------

import time


# ----------MATHEMATICAL FUNCTIONS/MATRIX OPERATIONS:----------

# NumPy instalation: in the cmd: 'py -m pip install numpy'
import numpy as np

# SciPy instalation: in the cmd: 'py -m pip install scipy'

# Matrix comparisons by their inner product
from ..mat_inner_product import *

from ..Phase4_Aux.gram_schmidt import *


# ----------FILE MANAGEMENT:----------

# File opening

from ..read_matrix import read_matrix_from_txt, read_matrix_from_txt_general


# ----------SYSTEM:----------


# ----------INITIAL MATRIX GENERATOR:----------


# ----------COMBINATORY:----------

from ..recur_factorial import *


# ----------INPUT CONTROL:----------

from ..input_control import input_control


# ----------ALGORITHM 2: AUXILIAR FUNCTIONS:----------


# ----------PHOTON COMB BASIS:----------

from ..photon_comb_basis import photon_combs_generator, state_in_basis


# ----------ALGORITHM 3: AUXILIAR FUNCTIONS:----------


# Adjoint representation


# ----------ALGORITHM 4: AUXILIAR FUNCTIONS:----------

# Required logarithms
from ..Phase4_Aux._4_Logarithms_required import *


# ----------ALGORITHM 6: AUXILIAR FUNCTIONS:----------

from ..Phase6_Aux._6_schmidt import schmidt_rank_vector
from ..Phase6_Aux._6_basis_manipulations import (
    state_leading_terms,
    state_leading_fidelity,
)


# ---------------------------------------------------------------------------------------------------------------------------
# 														MAIN CODE
# ---------------------------------------------------------------------------------------------------------------------------


def StateSchmidt(
    file_input_state=True,
    file_input_matrix=True,
    state_input=False,
    U_input=False,
    file_output=True,
    filename_state=False,
    filename_matrix=False,
    base_input=False,
    vec_base=[[False, False], [False, False]],
    acc_d=3,
    txt=False,
    fidelity=0.95,
):
    """
    Loads a state either from a file or from inputs given directly to the function, as well as an unitary matrix under the same condition.
    First, the state gets adapted to the vec_base available (either generated or given) and multiplied by the unitary matrix operator.
    Later, now that it has interacted with the circuit, its entanglement is computed via the Schmidt rank vector.
    """

    print("==============================================================")
    print("||| ENTANGLEMENT VALUES OF STATES WITHIN AN UNITARY SYSTEM |||")
    print("==============================================================\n\n")

    # Input control: in case there is something wrong with given inputs, it is notified on-screen
    file_input_state, filename_state, _, acc_d = input_control(
        module=6,
        file_input=file_input_state,
        M_input=U_input,
        file_output=file_output,
        filename=filename_state,
        txt=txt,
        acc_d=acc_d,
    )
    file_input_matrix, filename_matrix, _, acc_d = input_control(
        module=6,
        file_input=file_input_matrix,
        M_input=U_input,
        file_output=file_output,
        filename=filename_matrix,
        txt=txt,
        acc_d=acc_d,
    )

    # ----------STATE OF INTEREST:----------

    # Loading U from the file name.txt
    if file_input_state is True:
        arrays, array_sep = read_matrix_from_txt_general(filename_state)

        # we separate each element
        state_basis_vectors = np.array(arrays[: array_sep[0]], dtype=int)
        state_prob_amplitudes = arrays[array_sep[0] : array_sep[1]]
        modes_per_partition = np.array(arrays[array_sep[1] : array_sep[2]], dtype=int)[0]

    if txt is True:
        # introduce the three elements (basis, weights and modes per partition) FROM THE SAME FILE
        print("\nOur state's basis vectors:")
        print(np.round(state_basis_vectors, acc_d))
        print("\nEach basis vector's probabilities:")
        print(np.round(state_prob_amplitudes, acc_d))
        print("\nModes per partition employed:")
        print(np.round(modes_per_partition, acc_d))

    # photonic ket obtaination (beware of how to read it!)
    if len(state_basis_vectors.shape) < 3:
        photons = state_basis_vectors[0]
    else:
        photons = state_basis_vectors[0, 0]
    m = len(photons)

    # ----------U MATRIX NOT CRAFTABLE WITH OPTICAL DEVICES:----------

    # Loading U from the file name.txt
    if file_input_matrix is True:
        U_input = read_matrix_from_txt(filename_matrix)

    if txt is True:
        print("\nU_input:")
        print(np.round(U_input, acc_d))

    len(U_input[:, 0])

    # ----------BASE IN M SPACE:----------

    # We load the combinations with the same amount of photons in order to create the vector basis
    if str(np.array(vec_base)[0, 0]) == str(False):
        vec_base = photon_combs_generator(m, photons)

    elif txt is True:
        print("\nLoaded an external array for the Fock basis.")

    if txt is True:
        print(f"\nVector basis:\n{vec_base}")

    # ---------MAIN PROCESS:----------

    if file_output is True:
        schmidt_leading_file = open(f"{filename_state}_{filename_matrix}_schmidt_leading.txt", "w+")
        schmidt_fidelity_file = open(
            f"{filename_state}_{filename_matrix}_schmidt_fidelity_{fidelity}.txt", "w+"
        )

    # Beginning of time measurement
    t = time.process_time_ns()
    # print(len(state_basis_vectors))
    for i in range(len(state_basis_vectors)):
        if len(state_basis_vectors[i].shape) < 2:
            basis_vectors = np.array([state_basis_vectors[i]])
            prob_amplitudes = np.array([state_prob_amplitudes[i]])
        else:
            basis_vectors = state_basis_vectors[i]
            prob_amplitudes = state_prob_amplitudes[i]

        input_state = state_in_basis(basis_vectors, prob_amplitudes, vec_base)
        output_state = np.matmul(U_input, input_state.T)
        pre_entanglement = schmidt_rank_vector(input_state, vec_base, list(modes_per_partition))
        post_entanglement = schmidt_rank_vector(output_state, vec_base, list(modes_per_partition))

        vec_base_leading, output_state_leading = state_leading_terms(output_state, vec_base)
        vec_base_fidelity, output_state_fidelity = state_leading_fidelity(
            output_state, vec_base, fidelity
        )
        short = state_in_basis(vec_base_fidelity, output_state_fidelity, vec_base)

        post_entanglement_leading = schmidt_rank_vector(
            output_state_leading, vec_base_leading, list(modes_per_partition)
        )
        post_entanglement_fidelity = schmidt_rank_vector(
            short, vec_base_fidelity, list(modes_per_partition)
        )

        # these are generated without the need for a function, but the current way it is better organised
        # vec_base_fidelity=vec_base[np.argsort(np.abs(output_state)**2)[-leading_terms(output_state,fidelity):]] # se ordenan los pesos de mayor a menor
        # output_state_fidelity=output_state[np.argsort(np.abs(output_state)**2)[-leading_terms(output_state,fidelity):]]

        weights = np.round(abs(output_state_leading) ** 2, 3)
        balance = min(weights) / max(weights)

        weights_f = np.round(abs(output_state_fidelity) ** 2, 3)
        balance_f = min(weights_f) / max(weights_f)

        if file_output is True:
            # Leading states (no fidelity applied)
            schmidt_leading_file.write(
                f"\nIteration {i}\nFor the input state (basis and probabilities for each):\n"
            )
            np.savetxt(schmidt_leading_file, basis_vectors, delimiter=",")
            np.savetxt(schmidt_leading_file, prob_amplitudes, delimiter=",")
            schmidt_leading_file.write(
                f"\nSchmidt rank for the input state: {pre_entanglement}\nSchmidt rank for the output state: {post_entanglement_leading}"
            )
            schmidt_leading_file.write(
                "\nThe output state's state basis and probabilities of collapse for each:\n"
            )
            np.savetxt(schmidt_leading_file, vec_base_leading, delimiter=",")
            np.savetxt(schmidt_leading_file, output_state_leading, delimiter=",")
            schmidt_leading_file.write(f"\nBalance: {balance}\n\n")

            # Leading-fidelity states (fidelity applied)
            schmidt_fidelity_file.write(
                f"\nIteration {i}\nFor the input state (basis and probabilities for each):\n"
            )
            np.savetxt(schmidt_fidelity_file, basis_vectors, delimiter=",")
            np.savetxt(schmidt_fidelity_file, prob_amplitudes, delimiter=",")
            schmidt_fidelity_file.write(
                f"\nSchmidt rank for the input state: {pre_entanglement}\nSchmidt rank for the output state (fidelity={fidelity}): {post_entanglement_fidelity}"
            )
            schmidt_fidelity_file.write(
                f"\nThe output state's state basis and probabilities of collapse for each (fidelity={fidelity}):\n"
            )
            np.savetxt(schmidt_fidelity_file, vec_base_fidelity, delimiter=",")
            np.savetxt(schmidt_fidelity_file, output_state_fidelity, delimiter=",")
            schmidt_fidelity_file.write(f"\nBalance: {balance_f}\n\n")

        if txt is True:
            print(f"\n\nIteration {i}\n")
            print(f"Considered vector: {basis_vectors}")
            print(f"Input state: {input_state}")
            print(f"Output state: {output_state}")
            print(f"Pre-circuit entanglement: {pre_entanglement}")
            print(f"Post-circuit entanglement: {post_entanglement}")
            # Nonzero elements output
            # print(f"Number of states available: {len(output_state)-np.sum(np.isclose(output_state,np.zeros_like(output_state),rtol=1e-05, atol=1e-02))}")

            print(f"Number of states available: {len(output_state_leading)}")
            print(
                f"Our array's exact basis elements and their probabilities after passing the circuit are located in the iteration {i} of '{filename_state}_leading.txt'."
            )
            print(vec_base_leading)
            print(output_state_leading)
            print(f"Post-circuit entanglement (leading states): {post_entanglement_leading}")

            print(f"Balance: {balance}")

            # print(f"Number of leading states for fidelity={fidelity}: {leading_terms(output_state,fidelity)}")   #importantes
            print(f"Number of leading states for fidelity={fidelity}: {len(output_state_fidelity)}")
            print(
                f"The fidelity={fidelity} array's exact basis elements and their probabilities after passing the circuit are located in the iteration {i} of '{filename_state}_reduction_{fidelity}.txt'."
            )
            print(vec_base_fidelity)
            print(output_state_fidelity)
            print(
                f"Post-circuit entanglement (fidelity={fidelity} in state): {post_entanglement_fidelity}"
            )

            print(f"Balance (fidelity={fidelity}): {balance_f}")

    if file_output is True:
        schmidt_leading_file.close()
        schmidt_fidelity_file.close()

    """
	if txt==True:

		print("\n\nSolution:")
		print(np.round(sol_array,acc_d))
		print("\nRespective separation lenghts:")
		print(np.round(sol_mod,acc_d))


	if file_output==True:

		toponogov_file=open(f"{filename}_toponogov_general.txt","w+")

		for i in range(len(sol_mod)):

			toponogov_file.write("\ntoponogov("+filename+f")_{i+1} (separation length from original of {sol_mod[i]}):\n")

			toponogov_file_2=open(f"{filename}_toponogov_{i+1}.txt","w+")

			np.savetxt(toponogov_file,sol_array[i],delimiter=",")

			np.savetxt(toponogov_file_2,sol_array[i],delimiter=",")

			toponogov_file_2.close()

		toponogov_file.close()

		if txt==True:

			print(f"\nAll matrix arrays with their separation length have been saved in '{filename}_toponogov_general.txt'.")

			print(f"\nFor each try, a file solely containing each matrix by the names of '{filename}_toponogov_N.txt', N being the number, have been saved.")
	"""

    # ----------TOTAL TIME REQUIRED:----------

    t_inc = time.process_time_ns() - t

    print(
        f"\nSchmidt entanglement: total time of execution (seconds): {float(t_inc / (10 ** (9)))}\n"
    )
