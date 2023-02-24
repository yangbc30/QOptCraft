import numpy as np

from QOptCraft.Main_Code import *


for extra in range(0, 10):
    for n in range(2, 5):
        m = 4 + (n - 2) + extra
        print(n, m)
        basis = photon_combs_generator(m, [n] + [0] * (m - 1))

        Bell = state_in_basis(
            [
                [1, 0, 1, 0] + [1] * (n - 2) + [0] * extra,
                [0, 1, 0, 1] + [1] * (n - 2) + [0] * extra,
            ],
            [1 / np.sqrt(2), 1 / np.sqrt(2)],
            basis,
        )
        densBell = np.matmul(np.matrix(Bell).T, np.matrix(Bell).conj())

        inputstate = state_in_basis(
            [[1, 0, 1, 0] + [1] * (n - 2) + [0] * extra], [1], basis
        )
        densinput = np.matmul(np.matrix(inputstate).T, np.matrix(inputstate).conj())

        M = len(basis)

        #   base_u_M=np.zeros((m*m,M,M),dtype=complex)
        base_u_M = matrix_u_basis_generator(m, M, [n] + [0] * (m - 1), False)[1]
        base_u_M = base_u_M[: m * m]

        base_u_M = gram_schmidt_2dmatrices(base_u_M)
        coefs = np.zeros(m * m, dtype=complex)

        for i in range(m * m):
            coefs[i] = mat_inner_product(1j * densinput, base_u_M[i])

        print("Bell  ", sum(abs(coefs) ** 2))

        coefs = np.zeros(m * m, dtype=complex)
        for i in range(m * m):
            coefs[i] = mat_inner_product(1j * densBell, base_u_M[i])

        print("Input ", sum(abs(coefs) ** 2))
