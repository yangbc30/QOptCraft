"""Measurement via computing the Schmidt rank of Fock states vectors after
passing a quantum through a given linear optics circuit (StateSchmidt()).
Note:
    The refactoring of this module is a work in progress.

References:
    https://physics.stackexchange.com/a/251574/102546

"""

import numpy as np

from qoptcraft._legacy.photon_comb_basis import state_in_basis


def expand_basis(state):
    """Takes the state from the "photonic" Hilbert space (with a total of n photons)
    to a (n+1)^m Hilbert space
    """
    n = int(np.sum(state))
    m = len(state)
    state_larger_space = np.array([1])  # Initializing the state for the Kronecker product
    for i in range(m):  # Iterates through all the subsystems (modes)
        qudit_i = np.zeros(n + 1)  # i-th qudit (n+1-dimensional system from 0 to n photons)
        qudit_i[int(state[i])] = 1

        state_larger_space = np.kron(
            state_larger_space, qudit_i
        )  # Composite system with one additional qudit at each round

    return state_larger_space


def large_basis(state, n, m):
    """Takes the state from the "photonic" Hilbert space (with a total of n photons)
    to a (n+1)^m Hilbert space  ##MODIFIED FOR subsets of total number photons
    """
    state_larger_space = np.array([1])  # Initializing the state
    for i in range(m):  # Iterates through all the subsystems (modes)
        qudit_i = np.zeros(n + 1)  # i-th qudit (n+1-dimensional system from 0 to n photons)
        qudit_i[int(state[i])] = 1

        state_larger_space = np.kron(
            state_larger_space, qudit_i
        )  # Composite system with one additional qudit at each round

    return state_larger_space


def leading_terms(state, ratio):
    """This function determines how many states are left relevant,
    for a particular ratio of precision
    """
    return (
        np.cumsum(np.flip(np.sort(np.abs(state) ** 2))) < ratio
    ).sum() + 1  # prob of each state growing in order


def state_leading_fidelity(state, basis, fidelity):
    nterms = leading_terms(state, fidelity)
    tol = (np.min(np.abs(state[np.argsort(np.abs(state) ** 2)[-(nterms + 1) :]]))) ** 2
    return state_leading_terms(state, basis, tol)


def state_leading_terms(state, basis, tol=1e-10):
    states = basis[list(map(lambda x: x[0], np.argwhere(np.abs(state) ** 2 > tol)))]
    probability_amplitudes = state[list(map(lambda x: x[0], np.argwhere(np.abs(state) ** 2 > tol)))]
    return states, probability_amplitudes


def schmidt_rank(state, subsystem1_basis, subsystem2_basis):
    row = 0
    matrix = np.zeros(shape=(len(subsystem1_basis), len(subsystem2_basis)), dtype=complex)
    for element_bas1 in subsystem1_basis:
        column = 0
        for element_bas2 in subsystem2_basis:
            matrix[row][column] = np.vdot(
                np.kron(element_bas1, element_bas2), state
            )  # Matrix for the SVD  elements are projections of
            column = column + 1
        row = row + 1

    U, D, V = np.linalg.svd(matrix)

    # In the SVD decomposition, nonzero elements of S
    # numpy counzeros has problems with tolerances
    return len(D) - np.sum(np.isclose(D, np.zeros_like(D)))


def schmidt_rank_vector(state, basis, mvec):
    """Takes input states from the photonic Hilbert space into a larger space
    for entanglement evaluation mvec gives the modes in each subsystem

    Note:
        The function schmidt_rank_vector evaluates the entanglement between different
        subsystems of the global state. It returns a vector where each element is the
        Schmidt rank for the bipartite system composed of the corresponding subsystem
        and the rest of the state. The user can introduce different groupings of the
        modes to define each subsystem. Internally, it is computed by taking the state
        space to a larger dimension where all the modes can carry up to n photons.
    """
    n = int(np.sum(basis[0]))  # n+1-dimensional qudits
    m = len(basis[0])
    Ns = len(mvec)  # number of subsystems

    rank_vector = np.zeros(Ns)
    modes_in_subsystem = np.cumsum([0, *mvec])  # Adjusted to include 0 index

    for i in range(Ns):  # Go through each subsystem
        state_larger_space = np.array([1])  # Initializing the state for the Kronecker product

        subsystem1_basis = []

        for k in range((n + 1) ** (mvec[i])):
            base_np1_string = np.base_repr(k, base=n + 1)
            basis1_state = [
                int(digit) for digit in "0" * (mvec[i] - len(base_np1_string)) + base_np1_string
            ]  # Padded to m-1 digits in base n+1
            subsystem1_basis.append(large_basis(basis1_state, n, mvec[i]))

        subsystem2_basis = []

        for k in range((n + 1) ** (m - mvec[i])):
            base_np1_string = np.base_repr(k, base=n + 1)
            basis2_state = [
                int(digit) for digit in "0" * (m - mvec[i] - len(base_np1_string)) + base_np1_string
            ]  # Padded to m-1 digits in base n+1
            subsystem2_basis.append(large_basis(basis2_state, n, m - mvec[i]))

        ind = 0  # index in the input state
        state_larger_space = np.zeros_like(expand_basis(basis[0]))

        for phot_basis_element in basis:
            small_subsystem = phot_basis_element[modes_in_subsystem[i] : modes_in_subsystem[i + 1]]
            large_subsystem = np.delete(
                phot_basis_element, range(modes_in_subsystem[i], modes_in_subsystem[i + 1]), axis=0
            )
            alpha_i = state[ind]
            # Iterates through the amplitudes
            state_larger_space = state_larger_space + alpha_i * np.kron(
                large_basis(small_subsystem, n, mvec[i]),
                large_basis(large_subsystem, n, m - mvec[i]),
            )
            ind = ind + 1

        rank_vector[i] = schmidt_rank(state_larger_space, subsystem1_basis, subsystem2_basis)

    return rank_vector


def schmidt_entanglement(
    U_input=False,
    filename_state=False,
    vec_base=[[False, False], [False, False]],
    fidelity=0.95,
):
    """
    Loads a state either from a file or from inputs given directly to the function, as well as a unitary matrix under the same condition.
    First, the state gets adapted to the vec_base available (either generated or given) and multiplied by the unitary matrix operator.
    Later, now that it has interacted with the circuit, its entanglement is computed via the Schmidt rank vector.
    """

    arrays, array_sep = read_matrix_from_txt_general(filename_state)
    state_basis_vectors = np.array(arrays[: array_sep[0]], dtype=int)
    state_prob_amplitudes = arrays[array_sep[0] : array_sep[1]]
    modes_per_partition = np.array(arrays[array_sep[1] : array_sep[2]], dtype=int)[0]

    for i, fock in enumerate(state_basis_vectors):
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
