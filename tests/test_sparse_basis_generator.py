import pytest
from numpy.testing import assert_allclose

from QOptCraft import matrix_u_basis_generator, matrix_u_basis_generator_sparse, photon_combs_generator


extra = 2
n = 2
m = 4 + (n - 2) + extra
basis = photon_combs_generator(m, [n] + [0] * (m - 1))
dim = len(basis)


@pytest.mark.parametrize(("m", "dim", "photons", "base_input"), ((m, dim, [n] + [0] * (m - 1), False),))
def test_sparse_basis_generator(m, dim, photons, base_input) -> None:
    u_m, u_M, u_m_e, u_m_f, sep, U_m, U_M = matrix_u_basis_generator(m, dim, photons, base_input)
    (
        u_m_sparse,
        u_M_sparse,
        u_m_e_sparse,
        u_m_f_sparse,
        sep_sparse,
        U_m_sparse,
        U_M_sparse,
    ) = matrix_u_basis_generator_sparse(m, dim, photons, base_input)

    for matrix, matrix_sparse in zip(u_m, u_m_sparse):
        assert_allclose(matrix, matrix_sparse.toarray())
