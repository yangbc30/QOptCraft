from typing import Literal

from numpy.testing import assert_allclose
import pytest

from qoptcraft.state import Fock
from qoptcraft.math import haar_random_unitary
from qoptcraft.evolution import photon_unitary
from qoptcraft.invariant import spectral_invariant, invariant_subspaces


@pytest.mark.parametrize(
    ("modes", "photons", "invariant_operator", "order"),
    (
        (2, 2, "nested_commutator", 2),
        (2, 3, "nested_commutator", 2),
        (2, 2, "nested_commutator", 3),
        (3, 3, "nested_commutator", 2),
        (3, 3, "nested_commutator", 3),
        (2, 3, "nested_commutator", 4),
        (3, 3, "nested_commutator", 4),
        (4, 3, "nested_commutator", 4),
        (2, 2, "higher_order_projection", 2),
        (2, 2, "higher_order_projection", 3),
        (2, 3, "higher_order_projection", 3),
        (3, 3, "higher_order_projection", 2),
        (3, 3, "higher_order_projection", 3),
        (2, 3, "higher_order_projection", 4),
        (3, 3, "higher_order_projection", 4),
        (4, 3, "higher_order_projection", 4),
    ),
)
def test_self_adjoint_subspace_decomposition(
    modes: int,
    photons: int,
    invariant_operator: Literal["higher_order_projection", "nested_commutator"],
    order: int,
) -> None:
    scattering = haar_random_unitary(modes)
    unitary = photon_unitary(scattering, photons)

    in_state = Fock(photons, *[0] * (modes - 1))
    out_state = in_state.evolution(unitary)

    subspaces = invariant_subspaces(modes, photons, invariant_operator=invariant_operator, order=order)

    for subspace in subspaces:

        in_invariant = spectral_invariant(in_state, subspace=subspace, orthonormal=False)
        out_invariant = spectral_invariant(out_state, subspace=subspace, orthonormal=False)

        assert_allclose(in_invariant, out_invariant, atol=1e-6, rtol=1e-6)



@pytest.mark.parametrize(
    ("modes", "photons", "invariant_operator", "order"),
    (
        (2, 2, "nested_commutator", 2),
        (2, 3, "nested_commutator", 2),
        (2, 2, "higher_order_projection", 2),
        (2, 3, "higher_order_projection", 2),
    ),
)
def test_self_adjoint_subspace_decomposition_impossible(
    modes: int,
    photons: int,
    invariant_operator: Literal["higher_order_projection", "nested_commutator"],
    order: int,
) -> None:

    in_state = Fock(photons, 0)
    out_state = Fock(photons - 1, 1)

    subspaces = invariant_subspaces(modes, photons, invariant_operator=invariant_operator, order=order)

    for subspace in subspaces:

        in_invariant = spectral_invariant(in_state, subspace=subspace, orthonormal=False)
        out_invariant = spectral_invariant(out_state, subspace=subspace, orthonormal=False)

        try:
            assert_allclose(in_invariant, out_invariant, atol=1e-6, rtol=1e-6)
        except AssertionError:
            pass
