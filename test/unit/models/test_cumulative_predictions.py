"""Tests for point prediction semantics.

All models should predict the expected value at time t+h (point prediction),
not cumulative sums. This enables correct multi-horizon scoring.
"""

import numpy as np
import pytest

from aegis.models.persistence import LocalLevelModel, RandomWalkModel
from aegis.models.reversion import MeanReversionModel
from aegis.models.trend import LinearTrendModel


class TestPointPredictionSemantics:
    """Tests that models predict point values at horizon h."""

    def test_random_walk_predicts_point(self) -> None:
        """RandomWalkModel should predict last_observation at all horizons."""
        model = RandomWalkModel()

        # Feed some observations
        for t in range(10):
            model.update(float(t), t)

        # Point prediction at any horizon should be last_y (9.0)
        for h in [1, 4, 10, 100]:
            pred = model.predict(horizon=h)
            assert pred.mean == pytest.approx(9.0, rel=0.01)

    def test_local_level_predicts_point(self) -> None:
        """LocalLevelModel should predict smoothed_level at all horizons."""
        model = LocalLevelModel(alpha=0.3)

        # Feed constant values to let level converge
        for t in range(50):
            model.update(2.0, t)

        # Level should have converged to ~2.0
        # Point prediction at any horizon should be the level
        for h in [1, 4, 10]:
            pred = model.predict(horizon=h)
            assert pred.mean == pytest.approx(2.0, rel=0.05)

    def test_linear_trend_predicts_point(self) -> None:
        """LinearTrendModel should predict value at t+h."""
        model = LinearTrendModel()

        # Feed linearly increasing values
        for t in range(100):
            y = float(t)
            model.update(y, t)

        # Model should learn: intercept ≈ 0, slope ≈ 1
        # At h=10, should predict y(t+10) ≈ 99 + 10 = 109
        pred = model.predict(horizon=10)
        assert pred.mean == pytest.approx(109.0, rel=0.1)

    def test_mean_reversion_predicts_point(self) -> None:
        """MeanReversionModel should predict value at t+h."""
        model = MeanReversionModel(mu=0.0, phi=0.5, decay=0.99)

        # Initialize with observation far from mu
        model.update(10.0, 0)
        model.mu = 0.0  # Force mu to 0 for predictable test

        # At horizon h, prediction is mu + phi^h * (last_y - mu)
        pred = model.predict(horizon=4)

        # Expected: 0 + 0.5^4 * 10 = 0.0625 * 10 = 0.625
        expected = 0.0 + (0.5**4) * 10.0
        assert pred.mean == pytest.approx(expected, rel=0.01)


class TestPolynomialTrendAccuracy:
    """Test that polynomial trends are predicted correctly end-to-end."""

    def test_polynomial_trend_long_horizon(self) -> None:
        """AEGIS should correctly predict polynomial trends at long horizons."""
        from aegis.system import AEGIS

        # Generate polynomial trend: y = 0.0001*t^2 + 0.05*t
        n = 500
        t = np.arange(n)
        a, b = 0.0001, 0.05
        signal = a * t**2 + b * t

        aegis = AEGIS()
        aegis.add_stream("test")

        for y in signal:
            aegis.observe("test", y)
            aegis.end_period()

        # Predict at h=64 (reasonable horizon for point prediction)
        pred = aegis.predict("test", horizon=64)

        # True value at t = n-1 + 64 = 563
        future_t = n - 1 + 64
        true_val = a * future_t**2 + b * future_t

        # For polynomial trends, error should be reasonable
        # (may not be perfect due to model selection dynamics)
        error = abs(pred.mean - true_val)
        # Allow 50% error since trend models may not perfectly capture polynomials
        assert error < 0.5 * true_val, f"Error {error:.1f} too large (true={true_val:.1f})"
