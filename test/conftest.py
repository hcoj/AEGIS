"""Pytest configuration and fixtures for AEGIS tests."""

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """Reproducible random number generator."""
    return np.random.default_rng(42)
