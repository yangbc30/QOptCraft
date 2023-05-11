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

import numpy as np


def expand_basis(state):
    """
    Takes the state from the "photonic" Hilbert space (with a total of n photons) to a (n+1)^m Hilbert space
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


def expand(state, basis):
    state_larger_space = np.zeros_like(expand_basis(basis[0]))
    ind = 0  # index in the given basis
    for alpha_i in state:  # Iterates through the amplitudes
        state_larger_space = state_larger_space + alpha_i * expand_basis(basis[ind])
        ind = ind + 1

    return state_larger_space


def qudit_basis(d, n):  # n qudits (dimension d)
    qdbasis = []

    for k in range(d**n):
        state_string = np.base_repr(k, base=d)
        [
            int(digit) for digit in "0" * (n - len(state_string)) + state_string
        ]  # Padded to n digits in base d. "Compact notation"

    np.array([1])  # Initializing the state for the Kronecker product
    # for i in range(m):    #Iterates through all the subsystems (modes)
    #   qudit_i =np.zeros(n+1)   # i-th qudit (n+1-dimensional system from 0 to n photons)
    #   qudit_i[state[i]]=1

    #  state_larger_space=np.kron(state_larger_space,qudit_i)  #Composite system with one additional qudit at each round

    # qdbasis.append(basis_state)
    return qdbasis


def large_basis(state, n, m):
    """
    Takes the state from the "photonic" Hilbert space (with a total of n photons) to a (n+1)^m Hilbert space  ##MODIFIED FOR subsets of total number photons
    """

    state_larger_space = np.array([1])  # Initializing the state
    for i in range(m):  # Iterates through all the subsystems (modes)
        qudit_i = np.zeros(n + 1)  # i-th qudit (n+1-dimensional system from 0 to n photons)
        qudit_i[int(state[i])] = 1

        state_larger_space = np.kron(
            state_larger_space, qudit_i
        )  # Composite system with one additional qudit at each round

    return state_larger_space


def RotMat(N, offset):
    return np.eye(N, k=offset) + np.eye(N, k=-(N - offset))


def leading_terms(state, ratio):
    """
    This function determines how many states are left relevant, for a particular ratio of precision
    """

    return (np.cumsum(np.flip(np.sort(np.abs(state) ** 2))) < ratio).sum() + 1  # prob of each state growing in order


def state_leading_fidelity(state, basis, fidelity):
    nterms = leading_terms(state, fidelity)
    tol = (np.min(np.abs(state[np.argsort(np.abs(state) ** 2)[-(nterms + 1) :]]))) ** 2
    # print(tol)
    return state_leading_terms(state, basis, tol)


def state_leading_terms(state, basis, tol=1e-10):
    # print(list(map(lambda x: x[0],np.argwhere(np.abs(state)**2 > tol))))
    states = basis[list(map(lambda x: x[0], np.argwhere(np.abs(state) ** 2 > tol)))]
    probability_amplitudes = state[list(map(lambda x: x[0], np.argwhere(np.abs(state) ** 2 > tol)))]
    # print(states)
    return states, probability_amplitudes
