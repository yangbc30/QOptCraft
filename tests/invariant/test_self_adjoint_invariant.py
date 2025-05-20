from numpy.testing import assert_allclose

import pytest

from qoptcraft.state import Fock
from qoptcraft.operators import haar_random_unitary
from qoptcraft.evolution import photon_unitary
from qoptcraft.invariant import spectral_invariant, invariant_subspaces_nested_commutator


@pytest.mark.parametrize(
    ("modes", "photons", "order"),
    (
        (2, 2, 2),
        (2, 3, 2),
        (3, 3, 2),
        (2, 3, 3),
        (3, 3, 3),
    ),
)
def test_self_adjoint_subspace_decomposition(modes: int, photons: int, order: int) -> None:
    scattering = haar_random_unitary(modes)
    unitary = photon_unitary(scattering, photons)

    in_state = Fock(photons, *[0] * (modes - 1))
    out_state = in_state.evolution(unitary)

    subspaces = invariant_subspaces_nested_commutator(modes, photons, order)

    for subspace in subspaces:

        in_invariant = spectral_invariant(in_state, subspace=subspace, orthonormal=False)
        out_invariant = spectral_invariant(out_state, subspace=subspace, orthonormal=False)

        assert_allclose(in_invariant, out_invariant, atol=1e-6, rtol=1e-6)
