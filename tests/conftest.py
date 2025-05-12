"""The conftest.py file allows us to initialise test functions
that can be repeatedly used across several tests.
"""

import shutil
from pathlib import Path
import pytest

import qoptcraft as qoc


# Remove cached data folder
SAVE_DATA_PATH = qoc.config.SAVE_DATA_PATH = Path("tests", "save_basis").resolve()
try:
    shutil.rmtree(SAVE_DATA_PATH)
except FileNotFoundError:
    ...


@pytest.fixture
def some_fixture():
    return ...
