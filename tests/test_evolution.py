from itertools import combinations

import pytest
from numpy.testing import assert_allclose

from qoptcraft.operators import haar_random_unitary
from qoptcraft.evolution.unitary_evol import (
    photon_unitary_hamiltonian,
    photon_unitary_permanent,
    photon_unitary_permanent_ryser,
    photon_unitary,
)

@pytest.mark.parametrize(("modes", "photons"), ((2,2), (2, 3), (5, 3), (4,5)))
def test_unitary_evolution(modes: int, photons: int):
    S = haar_random_unitary(modes)
    U_qm = photon_unitary(S, photons)
    U_ham = photon_unitary_hamiltonian(S, photons)
    U_per = photon_unitary_permanent(S, photons)
    U_rys = photon_unitary_permanent_ryser(S, photons)

    for U_1, U_2 in combinations([U_qm, U_ham, U_per, U_rys], 2):
        try:
            assert_allclose(U_1, U_2)
        except AssertionError:
            assert_allclose(U_1, -U_2)