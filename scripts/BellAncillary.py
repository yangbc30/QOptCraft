import numpy as np

from QOptCraft import *


for extra in range(0, 10):
    for n in range(2, 5):
        m = 4 + (n - 2) + extra
        print(f"n = {n}, m = {m}")
        basis = photon_combs_generator(m, [n] + [0] * (m - 1))

        Bell = state_in_basis(
            [[1, 0, 1, 0] + [1] * (n - 2) + [0] * extra, [0, 1, 0, 1] + [1] * (n - 2) + [0] * extra],
            [1 / np.sqrt(2), 1 / np.sqrt(2)],
            basis,
        )
        densBell = np.matrix(Bell).T @ np.matrix(Bell).conj()

        inputstate = state_in_basis([[1, 0, 1, 0] + [1] * (n - 2) + [0] * extra], [1], basis)
        densinput = np.matrix(inputstate).T @ np.matrix(inputstate).conj()

        dim = len(basis)

        #   base_u_M=np.zeros((m*m, dim, dim), dtype=complex)
        base_u_m, base_u_M, base_u_m_e, base_u_m_f, separator_e_f, base_U_m, base_U_M = matrix_u_basis_generator_sparse(
            m, dim, [n] + [0] * (m - 1), False
        )
        # base_u_M = base_u_M[: m * m]

        # base_u_M = gram_schmidt_2dmatrices(base_u_M)
        orthonormal_base = gram_schmidt_modified_2dmatrices(base_u_M)

        coefs = np.zeros(m * m, dtype=complex)

        for i in range(m * m):
            coefs[i] = mat_inner_product(1j * densinput, orthonormal_base[i])

        print("Bell ", sum(abs(coefs) ** 2))

        coefs = np.zeros(m * m, dtype=complex)
        for i in range(m * m):
            coefs[i] = mat_inner_product(1j * densBell, orthonormal_base[i])

        print("Input ", sum(abs(coefs) ** 2))
