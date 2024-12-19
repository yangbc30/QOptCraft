from math import isclose

import pytest

from qoptcraft.state import Fock, PureState
from qoptcraft.operators import haar_random_unitary
from qoptcraft.basis import photon_basis
from qoptcraft.evolution import photon_unitary
from qoptcraft.invariant import photon_invariant

modes = 3
photons = 5


@pytest.mark.parametrize(
    ("modes", "photons"),
    (
        (3, 5),
        (5, 3),
    ),
)
def test_evolution_reduced(modes: int, photons: int) -> None:
    scattering = haar_random_unitary(modes)
    unitary = photon_unitary(scattering, photons)

    in_state = Fock(photons, *[0] * (modes - 1))
    in_invariant = photon_invariant(in_state, method="reduced")

    out_coefs = unitary @ in_state.state_in_basis()
    out_state = PureState(photon_basis(modes, photons), out_coefs)
    out_invariant = photon_invariant(out_state, method="reduced")

    assert isclose(in_invariant, out_invariant, abs_tol=1e-10)
