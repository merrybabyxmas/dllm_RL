"""
pytest configuration and shared fixtures for cc_rl tests.
"""
import os
import sys
import pytest

# Ensure the package src directory is on the path for editable installs
_SRC = os.path.join(os.path.dirname(__file__), "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# Path to the test fixtures directory
FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


@pytest.fixture
def toy_fixture_path():
    """Return the absolute path to the toy_2_plus_3.jsonl fixture."""
    return os.path.join(FIXTURES_DIR, "toy_2_plus_3.jsonl")


@pytest.fixture
def toy_trajectories(toy_fixture_path):
    """Load raw toy trajectories (no advantages assigned yet)."""
    from cc_rl.data.toy_loader import load_toy_2_plus_3
    return load_toy_2_plus_3(toy_fixture_path)
