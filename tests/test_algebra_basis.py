import pytest
from numpy.testing import assert_allclose

from QOptCraft import matrix_u_basis_generator
from QOptCraft.basis import _algebra_basis, _photon_basis


extra = 0
photons = 2
modes = 4 + (photons - 2) + extra
basis = _photon_basis(modes, photons)
dim = len(basis)


@pytest.mark.parametrize(("modes", "dim", "photons", "basis"), ((modes, dim, photons, basis),))
def test_sparse_basis(modes, dim, photons, basis) -> None:
    basis = matrix_u_basis_generator(modes, dim, photons, basis)[0]
    basis_sparse = _algebra_basis(modes, photons)[0]

    for matrix, matrix_sparse in zip(basis, basis_sparse):
        assert_allclose(matrix, matrix_sparse.toarray())


@pytest.mark.parametrize(("modes", "dim", "photons", "basis"), ((modes, dim, photons, False),))
def test_sparse_image_basis(modes, dim, photons, basis) -> None:
    basis_image = matrix_u_basis_generator(modes, dim, photons, basis)[1]
    basis_image_sparse = _algebra_basis(modes, photons)[1]

    for matrix, matrix_sparse in zip(basis_image, basis_image_sparse):
        assert_allclose(matrix, matrix_sparse.toarray())
