"""The conftest.py file allows us to initialise test functions
that can be repeatedly used across several tests.
"""
from pathlib import Path
import pytest

import qoptcraft as qoc
from qoptcraft.config import SAVE_DATA_PATH

qoc.config.SAVE_DATA_PATH = Path("tests", "save_basis").resolve()


@pytest.fixture
def vector_fixture():
    """This vector can be used in tests as `vector_fixture`
    instead of `Vector(2, 1)`.
    """
    return ...
