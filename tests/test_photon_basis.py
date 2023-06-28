from pathlib import Path

import pytest

from qoptcraft.basis import _photon_basis


PHOTONS_1 = 3
PHOTONS_2 = 4

MODES_1 = 2
MODES_2 = 3

BASIS_1 = [(3, 0), (2, 1), (1, 2), (0, 3)]
BASIS_2 = [(4, 0), (3, 1), (2, 2), (1, 3), (0, 4)]
BASIS_3 = [
    (3, 0, 0),
    (0, 3, 0),
    (0, 0, 3),
    (1, 1, 1),
    (0, 2, 1),
    (0, 1, 2),
    (1, 0, 2),
    (2, 0, 1),
    (1, 2, 0),
    (2, 1, 0),
]
BASIS_4 = [
    (4, 0, 0),
    (0, 4, 0),
    (0, 0, 4),
    (3, 1, 0),
    (1, 3, 0),
    (0, 1, 3),
    (0, 3, 1),
    (3, 0, 1),
    (1, 0, 3),
    (1, 1, 2),
    (2, 1, 1),
    (1, 2, 1),
    (2, 2, 0),
    (2, 0, 2),
    (0, 2, 2),
]


@pytest.mark.parametrize(
    ("photons", "modes", "result_basis"),
    (
        (PHOTONS_1, MODES_1, BASIS_1),
        (PHOTONS_2, MODES_1, BASIS_2),
        (PHOTONS_1, MODES_2, BASIS_3),
        (PHOTONS_2, MODES_2, BASIS_4),
    ),
)
def test_photon_basis(photons: int, modes: int, result_basis: tuple[tuple[int, ...]]) -> None:
    result_basis = set(tuple(i) for i in result_basis)
    test_basis = _photon_basis(photons=photons, modes=modes)
    test_basis = set(tuple(i) for i in test_basis)
    assert result_basis == test_basis
