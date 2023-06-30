import numpy as np
from numpy.testing import assert_allclose
from numpy.random import randint
import pytest

from qoptcraft.optical_elements import clemens_decomposition, reck_decomposition
from qoptcraft.operators import haar_random_unitary


interferometer = haar_random_unitary(4)
left_clemens, diag_clemenes, right_clemens = clemens_decomposition(interferometer)
diag_reck, right_reck = reck_decomposition(interferometer)


@pytest.mark.parametrize(("modes"), (3, 5, 6))
def test_diag_clemens(modes):
    interferometer = haar_random_unitary(modes)
    left, diag, right = clemens_decomposition(interferometer)
    assert_allclose(
        diag - np.diag(np.diag(diag)),
        np.zeros((modes, modes)),
        rtol=1e-6,
        atol=1e-4,
        err_msg="D is not diagonal.",
    )


@pytest.mark.parametrize(("modes"), (3, 5, 6))
def test_diag_reck(modes):
    unitary = haar_random_unitary(modes)
    diag, right = reck_decomposition(unitary)
    assert_allclose(
        diag - np.diag(np.diag(diag)),
        np.zeros((modes, modes)),
        rtol=1e-6,
        atol=1e-4,
        err_msg="D is not diagonal.",
    )


@pytest.mark.parametrize(("modes"), (3, 5, 6))
def test_recomposition_clemens(modes):
    unitary = haar_random_unitary(modes)
    left_list, diag, right_list = clemens_decomposition(unitary)
    unitary_approx = np.eye(modes, dtype=np.complex64)

    for left in left_list:
        unitary_approx = unitary_approx @ left
    unitary_approx = unitary_approx @ diag
    for right in right_list:
        unitary_approx = unitary_approx @ right

    assert_allclose(
        unitary_approx,
        unitary,
        rtol=1e-6,
        atol=1e-4,
        err_msg="Unitary does not equal its decomposition.",
    )


@pytest.mark.parametrize(("modes"), (3, 5, 6))
def test_recomposition_reck(modes):
    unitary = haar_random_unitary(modes)
    diag, right_list = reck_decomposition(unitary)
    unitary_approx = np.eye(modes, dtype=np.complex64)

    unitary_approx = unitary_approx @ diag
    for right in right_list:
        unitary_approx = unitary_approx @ right

    assert_allclose(
        unitary_approx,
        unitary,
        rtol=1e-6,
        atol=1e-4,
        err_msg="Unitary does not equal its decomposition.",
    )
