import numpy as np

from QOptCraft import (
    gram_schmidt_modified,
    state_in_basis,
    photon_basis,
    matrix_u_basis_generator_sparse,
    mat_inner_product,
)


for extra in range(0, 10):
    for photons in range(2, 5):
        modes = 4 + (photons - 2) + extra
        print(f"photons = {photons}, modes = {modes}")
        basis = photon_basis(photons, modes)

        Bell = state_in_basis(
            [
                [1, 0, 1, 0] + [1] * (photons - 2) + [0] * extra,
                [0, 1, 0, 1] + [1] * (photons - 2) + [0] * extra,
            ],
            [1 / np.sqrt(2), 1 / np.sqrt(2)],
            basis,
        )
        densBell = np.matrix(Bell).T @ np.matrix(Bell).conj()

        inputstate = state_in_basis([[1, 0, 1, 0] + [1] * (photons - 2) + [0] * extra], [1], basis)
        densinput = np.matrix(inputstate).T @ np.matrix(inputstate).conj()

        dim_img = len(basis)

        #   base_u_M=np.zeros((m*m, dim, dim), dtype=complex)
        (
            base_algebra,
            base_img_algebra,
            base_algebra_e,
            base_algebra_f,
            separator_sym_antisym,
            base_group,
            base_img_group,
        ) = matrix_u_basis_generator_sparse(modes, dim_img, photons, basis)
        # base_u_M = base_u_M[: m * m]

        # base_u_M = gram_schmidt_2dmatrices(base_u_M)
        orthonormal_base = gram_schmidt_modified(base_img_algebra)

        coefs = [np.zeros(modes * modes, dtype=complex)]

        for i in range(modes * modes):
            coefs[i] = mat_inner_product(1j * densinput, orthonormal_base[i])

        print("Bell ", sum(abs(coefs) ** 2))

        coefs = np.zeros(modes * modes, dtype=complex)
        for i in range(modes * modes):
            coefs[i] = mat_inner_product(1j * densBell, orthonormal_base[i])

        print("Input ", sum(abs(coefs) ** 2))
