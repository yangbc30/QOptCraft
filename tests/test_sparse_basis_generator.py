import pytest
from numpy.testing import assert_allclose

from QOptCraft import matrix_u_basis_generator
from QOptCraft.lie_algebra import algebra_basis_sparse
from QOptCraft.basis import photon_basis


extra = 2
photons = 2
modes = 4 + (photons - 2) + extra
basis = photon_basis(photons, modes)
dim = len(basis)


@pytest.mark.parametrize(("modes", "dim", "photons", "base_input"), ((modes, dim, photons, False),))
def test_sparse_basis_generator(modes, dim, photons, base_input) -> None:
    alg_basis = matrix_u_basis_generator(modes, dim, photons, base_input)[0]
    alg_basis_sparse = algebra_basis_sparse(modes, dim, photons)[0]

    for matrix, matrix_sparse in zip(alg_basis, alg_basis_sparse):
        assert_allclose(matrix, matrix_sparse.toarray())
