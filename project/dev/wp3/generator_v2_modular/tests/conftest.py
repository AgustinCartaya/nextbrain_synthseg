"""pytest support (optional): the `device` fixture and the import paths."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import pytest  # noqa: E402


@pytest.fixture
def device():
    """Device of the tests (CPU; use run_tests.py --device for a GPU)."""
    return 'cpu'
