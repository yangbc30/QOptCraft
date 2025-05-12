from math import isclose

from numpy.testing import assert_allclose

import pytest

from qoptcraft.state import Fock, PureState
from qoptcraft.operators import haar_random_unitary
from qoptcraft.basis import photon_basis
from qoptcraft.evolution import photon_unitary
from qoptcraft.invariant import photon_invariant, spectral_invariant

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
    out_state = in_state.evolution(unitary)

    in_invariant = photon_invariant(in_state, method="reduced")
    out_invariant = photon_invariant(out_state, method="reduced")

    assert isclose(in_invariant, out_invariant, abs_tol=1e-10)


@pytest.mark.parametrize(
    ("modes", "photons", "subspace", "orthonormal"),
    (
        (2, 2, "preimage", False),
        (2, 3, "preimage", False),
        (3, 2, "preimage", False),
        (2, 2, "image", False),
        (2, 3, "image", False),
        (3, 2, "image", False),
        (2, 2, "complement", True),
        (2, 3, "complement", True),
        (3, 2, "complement", True),
    ),
)
def test_evolution_spectral(modes: int, photons: int, subspace, orthonormal: bool) -> None:
    scattering = haar_random_unitary(modes)
    unitary = photon_unitary(scattering, photons)

    in_state = Fock(photons, *[0] * (modes - 1))
    out_state = in_state.evolution(unitary)

    in_invariant = spectral_invariant(in_state, subspace=subspace, orthonormal=orthonormal)
    out_invariant = spectral_invariant(out_state, subspace=subspace, orthonormal=orthonormal)

    assert_allclose(in_invariant, out_invariant, atol=1e-7, rtol=1e-7)
