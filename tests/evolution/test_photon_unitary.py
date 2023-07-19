from itertools import combinations

import numpy as np
from numpy.typing import NDArray
from numpy.testing import assert_allclose
import pytest

from qoptcraft.operators import haar_random_unitary
from qoptcraft.optical_elements import beam_splitter
from qoptcraft.evolution import photon_unitary


phaseshifter = np.array([[np.exp(1j * 0.5), 0], [0, 1]])
lifted_phaseshifter = np.diag([np.exp(1j * 0.5 * 3), np.exp(1j * 0.5 * 2), np.exp(1j * 0.5), 1])

beamsplitter = beam_splitter(np.pi / 4, 0, 2, 0, 1)
hong_ou_mandel = np.array(
    [[0.5, -1 / np.sqrt(2), 0.5], [1 / np.sqrt(2), 0, -1 / np.sqrt(2)], [0.5, 1 / np.sqrt(2), 0.5]]
)


@pytest.mark.parametrize(
    ("matrix", "photons", "method"),
    (
        (haar_random_unitary(3), 3, "heisenberg"),
        (haar_random_unitary(2), 5, "hamiltonian"),
        (haar_random_unitary(4), 3, "permanent glynn"),
        (haar_random_unitary(5), 7, "permanent ryser"),
    ),
)
def test_unitarity(matrix: NDArray, photons: int, method: str):
    photonic_matrix = photon_unitary(matrix, photons, method)
    dim = photonic_matrix.shape[0]
    assert_allclose(np.eye(dim), photonic_matrix @ photonic_matrix.conj().T, atol=1e-7)


@pytest.mark.parametrize(
    ("matrix", "method"),
    (
        (haar_random_unitary(2), "heisenberg"),
        (haar_random_unitary(3), "hamiltonian"),
        (haar_random_unitary(4), "permanent glynn"),
        (haar_random_unitary(5), "permanent ryser"),
    ),
)
def test_one_photon(matrix: NDArray, method: str):
    photonic_matrix = photon_unitary(matrix, 1, method)
    try:
        assert_allclose(matrix, photonic_matrix, atol=1e-7, verbose=False)
    except AssertionError:
        assert_allclose(matrix, -photonic_matrix, atol=1e-7, verbose=False)


@pytest.mark.parametrize(
    ("matrix", "photons", "lifted_matrix", "method"),
    (
        (phaseshifter, 3, lifted_phaseshifter, "heisenberg"),
        (phaseshifter, 3, lifted_phaseshifter, "hamiltonian"),
        (phaseshifter, 3, lifted_phaseshifter, "permanent glynn"),
        (phaseshifter, 3, lifted_phaseshifter, "permanent ryser"),
        (beamsplitter, 2, hong_ou_mandel, "heisenberg"),
        (beamsplitter, 2, hong_ou_mandel, "hamiltonian"),
        (beamsplitter, 2, hong_ou_mandel, "permanent glynn"),
        (beamsplitter, 2, hong_ou_mandel, "permanent ryser"),
    ),
)
def test_photon_unitary(matrix: NDArray, photons: int, lifted_matrix: NDArray, method: str):
    photonic_matrix = photon_unitary(matrix, photons, method)
    try:
        assert_allclose(lifted_matrix, photonic_matrix, atol=1e-7, verbose=False)
    except AssertionError:
        assert_allclose(lifted_matrix, -photonic_matrix, atol=1e-7, verbose=False)


@pytest.mark.parametrize(("modes", "photons"), ((2, 2), (2, 3), (5, 3), (4, 5)))
def test_equal(modes: int, photons: int):
    S = haar_random_unitary(modes)
    unitary_heis = photon_unitary(S, photons, method="heisenberg")
    unitary_ham = photon_unitary(S, photons, method="hamiltonian")
    unitary_glynn = photon_unitary(S, photons, method="permanent glynn")
    unitary_ryser = photon_unitary(S, photons, method="permanent ryser")

    for U_1, U_2 in combinations([unitary_heis, unitary_ham, unitary_glynn, unitary_ryser], 2):
        try:
            assert_allclose(U_1, U_2)
        except AssertionError:
            assert_allclose(U_1, -U_2)
