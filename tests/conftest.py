"""The conftest.py file allows us to initialise test functions
that can be repeatedly used across several tests.
"""
import shutil
from pathlib import Path
import pytest

import qoptcraft as qoc


SAVE_DATA_PATH = qoc.config.SAVE_DATA_PATH = Path("tests", "save_basis").resolve()
shutil.rmtree(SAVE_DATA_PATH)


@pytest.fixture
def some_fixture():
    return ...
