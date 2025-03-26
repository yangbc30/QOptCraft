from pathlib import Path

import pytest
from numpy.testing import assert_allclose

from qoptcraft.evolution import photon_hamiltonian
from qoptcraft.basis import photon_basis
from qoptcraft.basis.algebra_basis import (
    sym_matrix,
    antisym_matrix,
    image_sym_matrix,
    image_antisym_matrix,
)


@pytest.mark.parametrize(("modes", "photons"), ((3, 2), (2, 1), (3, 5)))
def test_sym_hamiltonian(modes, photons) -> None:
    photonic_basis = photon_basis(modes, photons)
    for mode_1 in range(modes):
        for mode_2 in range(mode_1 + 1):
            sym = sym_matrix(mode_1, mode_2, modes)
            assert_allclose(
                photon_hamiltonian(sym, photons),
                image_sym_matrix(mode_1, mode_2, photonic_basis).toarray(),
            )


@pytest.mark.parametrize(("modes", "photons"), ((3, 2), (2, 1), (3, 5)))
def test_antisym_hamiltonian(modes, photons) -> None:
    photonic_basis = photon_basis(modes, photons)
    for mode_1 in range(modes):
        for mode_2 in range(mode_1):
            matrix = antisym_matrix(mode_1, mode_2, modes)
            assert_allclose(
                photon_hamiltonian(matrix, photons),
                image_antisym_matrix(mode_1, mode_2, photonic_basis).toarray(),
            )
