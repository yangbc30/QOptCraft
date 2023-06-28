from itertools import combinations

import pytest
from numpy.testing import assert_allclose

from qoptcraft.operators import haar_random_unitary
from qoptcraft.evolution import (
    photon_unitary,
    photon_unitary_permanent,
    photon_unitary_hamiltonian,
)


@pytest.mark.parametrize(("modes", "photons"), ((2, 2), (2, 3), (5, 3), (4, 5)))
def test_unitary_evolution(modes: int, photons: int):
    S = haar_random_unitary(modes)
    unitary = photon_unitary(S, photons)
    unitary_from_H = photon_unitary_hamiltonian(S, photons)
    unitary_glynn = photon_unitary_permanent(S, photons, method="glynn")
    unitary_ryser = photon_unitary_permanent(S, photons, method="ryser")

    for U_1, U_2 in combinations([unitary, unitary_from_H, unitary_glynn, unitary_ryser], 2):
        try:
            assert_allclose(U_1, U_2)
        except AssertionError:
            assert_allclose(U_1, -U_2)
