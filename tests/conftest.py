"""The conftest.py file allows us to initialise test functions
that can be repeatedly used across several tests.
"""

import pytest


@pytest.fixture
def vector_fixture():
    """This vector can be used in tests as `vector_fixture`
    instead of `Vector(2, 1)`.
    """
    return ...
