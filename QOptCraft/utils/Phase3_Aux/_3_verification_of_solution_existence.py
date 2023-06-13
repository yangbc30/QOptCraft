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
import sympy
from numpy.linalg import det, solve


# Adjoint representation for an U matrix
def adjoint_U(iH_U, U):
    return U.dot(iH_U.dot(np.transpose(np.conj(U))))


# A selection of linear independent equations is obtained
def eq_sys_finder(base_u_m, base_u_M):
    m = len(base_u_m[0])
    M = len(base_u_M[0])

    # Equation system initialization
    eq_sys = np.zeros((M * M, m * m), dtype=complex)

    # Storage of all the present equations in the system
    for j in range(m * m):
        for l in range(M):
            for o in range(M):
                eq_sys[M * l + o, j] = base_u_M[j, l, o]

    # Array wich storages m*m equations of eq_sys, for which we will attempt to solve the system
    # We will use np.append() in this and the following array for adding new terms
    eq_sys_choice = np.zeros((1, m * m), dtype=complex)

    # Array which storages the indexes of the chosen equations
    index_choice = np.zeros(1, dtype=int)

    cont = 0

    end = False

    # This loop searches for m*m equations of the list eq_sys for which a matrix with
    # a no-null determinant is made. That is, they are linear independent
    for l in range(M):
        for o in range(M):
            if cont > 0:
                # With this functions, we conserve the linear independent rows
                aux, inds = sympy.Matrix(eq_sys_choice).T.rref()

                # Applying inds to our two arrays, the algorithm is still ongoing until...
                eq_sys_choice = np.array(eq_sys_choice[np.array(inds)])
                index_choice = np.array(index_choice[np.array(inds)])

            # By obtaining a m*m x m*m equation system, with a no-null determinant, we have
            # computed the required system
            if len(eq_sys_choice[0]) == len(eq_sys_choice[:, 0]) and det(eq_sys_choice) != 0:
                end = True

                break

            # This simple condition saves multiple steps of null vectors being eliminated
            elif (eq_sys[M * l + o] != 0).any():
                if cont == 0:
                    eq_sys_choice[cont] = eq_sys[M * l + o]
                    index_choice[cont] = M * l + o

                else:
                    # We add the new arrays
                    eq_sys_choice = np.append(eq_sys_choice, np.array([eq_sys[M * l + o]]), axis=0)
                    index_choice = np.append(index_choice, np.array([M * l + o]), axis=0)

                cont += 1

        if end is True:
            break

    return eq_sys, eq_sys_choice, index_choice


def verification(
    U, base_u_m, base_u_m_e, base_u_m_f, sep, base_u_M, eq_sys, eq_sys_choice, index_choice
):
    m = len(base_u_m[0])
    M = len(base_u_M[0])

    # Solution arrays initialization
    sol = np.zeros((m * m, m * m), dtype=complex)
    sol_e = np.zeros((m * m, m * m), dtype=complex)
    sol_f = np.zeros((m * m, m * m), dtype=complex)

    # Saving both basis of the u(m) and u(M) subspaces

    for j in range(m * m):
        # We compute the adjoint for each matrix in the basis of u(M)
        adj_U_b_j = adjoint_U(base_u_M[j], U)
        adj_U_b_j_reshape = np.reshape(adj_U_b_j, M * M)

        # We choose the adj_U_b_j values of the indexes corresponding to the used equations
        adj_U_b_j_choice = np.array(adj_U_b_j_reshape[np.array(index_choice)])

        sol[j] = solve(eq_sys_choice, adj_U_b_j_choice)

        # Check for its validity for all possible equations?
        for l in range(M * M):
            suma = 0

            for o in range(m * m):
                suma += eq_sys[l, o] * sol[j, o]

            if np.round(suma, 8) != np.round(adj_U_b_j_reshape[l], 8):
                op = np.array([None])
                check = False

                # We return it three times for keeping consistency with the main algorithm 3
                return op, op, op, check

    # If the algorithm reaches this line, the solution exists. It is computed, giving a general solution of all
    # equations, and a separated version only applied to the e_jk and f_jk respectively, useful in the reconstruction of S
    check = True
    for j in range(m):
        for k in range(m):
            if m * j + k < sep:
                for l in range(m * m):
                    if (base_u_m_e[l] == base_u_m[m * j + k]).all():
                        sol_e[l] = sol[m * j + k]

            else:
                for l in range(m * m):
                    if (base_u_m_f[l] == base_u_m[m * j + k]).all():
                        sol_f[l] = sol[m * j + k]

                    if (base_u_m_f[l] == -base_u_m[m * j + k]).all():
                        sol_f[l] = -sol[m * j + k]

    return sol, sol_e, sol_f, check
