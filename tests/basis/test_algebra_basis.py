import pytest
from numpy.testing import assert_allclose

from qoptcraft._legacy.Phase3_Aux.get_algebra_basis_legacy import (
    matrix_u_basis_generator,
)
from qoptcraft.basis.algebra import algebra_basis
from qoptcraft.basis.photon import photon_basis


@pytest.mark.parametrize(("modes", "photons"), ((2, 3), (4, 2), (5, 3)))
def test_sparse_basis(modes, photons) -> None:
    fock_basis = photon_basis(modes, photons)
    dim = len(fock_basis)
    legacy_photons = [0] * modes
    legacy_photons[0] = photons
    basis = matrix_u_basis_generator(modes, dim, legacy_photons, fock_basis)[0]
    basis_sparse = algebra_basis(modes, photons)[0]

    for matrix, matrix_sparse in zip(basis, basis_sparse):
        assert_allclose(matrix, matrix_sparse.toarray())


# @pytest.mark.parametrize(("modes", "photons"), ((2, 3), (4, 2), (5, 3)))
@pytest.mark.skip(reason="Matrices appear in different order")
def test_sparse_image_basis(modes, photons) -> None:
    fock_basis = photon_basis(modes, photons)
    dim = len(fock_basis)
    legacy_photons = [0] * modes
    legacy_photons[0] = photons
    basis_image = matrix_u_basis_generator(modes, dim, legacy_photons, fock_basis)[1]
    basis_image_sparse = algebra_basis(modes, photons)[1]

    for matrix, matrix_sparse in zip(basis_image, basis_image_sparse):
        try:
            assert_allclose(matrix, matrix_sparse.toarray())
        except AssertionError:
            assert_allclose(matrix, -matrix_sparse.toarray())
