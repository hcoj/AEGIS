"""Tests for cumulative prediction semantics.

All models should predict the cumulative change in observations over the horizon,
not the instantaneous value at time t+h. This enables correct handling of
trending dynamics without special-case logic in scale_manager.
"""

import numpy as np
import pytest

from aegis.models.persistence import LocalLevelModel, RandomWalkModel
from aegis.models.reversion import MeanReversionModel
from aegis.models.trend import LinearTrendModel


class TestCumulativePredictionSemantics:
    """Tests that models predict cumulative change over horizon."""

    def test_random_walk_predicts_cumulative(self) -> None:
        """RandomWalkModel should predict horizon × last_observation."""
        model = RandomWalkModel()

        # Feed constant returns
        for t in range(10):
            model.update(0.5, t)  # Constant return of 0.5

        # Cumulative over h=10 should be 10 × 0.5 = 5.0
        pred = model.predict(horizon=10)
        assert pred.mean == pytest.approx(5.0, rel=0.01)

        # Cumulative over h=1 should be 1 × 0.5 = 0.5
        pred = model.predict(horizon=1)
        assert pred.mean == pytest.approx(0.5, rel=0.01)

    def test_local_level_predicts_cumulative(self) -> None:
        """LocalLevelModel should predict horizon × smoothed_level."""
        model = LocalLevelModel(alpha=0.3)

        # Feed constant values to let level converge
        for t in range(50):
            model.update(2.0, t)

        # Level should have converged to ~2.0
        # Cumulative over h=10 should be 10 × 2.0 = 20.0
        pred = model.predict(horizon=10)
        assert pred.mean == pytest.approx(20.0, rel=0.05)

    def test_linear_trend_predicts_cumulative_sum(self) -> None:
        """LinearTrendModel should predict sum of future returns, not return at t+h."""
        model = LinearTrendModel()

        # Feed linearly increasing returns: 0.1, 0.2, 0.3, ...
        # This represents a polynomial trend in levels
        for t in range(100):
            r = 0.1 * (t + 1)  # return = 0.1 * t
            model.update(r, t)

        # Model should learn: intercept ≈ 0.1, slope ≈ 0.1
        # At t=100, the next returns would be:
        #   r_101 = 0.1 * 101 = 10.1
        #   r_102 = 0.1 * 102 = 10.2
        #   ...
        #   r_110 = 0.1 * 110 = 11.0
        # Sum = 0.1 * (101 + 102 + ... + 110) = 0.1 * 1055 = 105.5

        pred = model.predict(horizon=10)

        # Expected cumulative: sum of arithmetic sequence
        # = h * intercept + slope * (h * last_t + h*(h+1)/2)
        # ≈ 10 * 0.1 + 0.1 * (10 * 100 + 10*11/2)
        # = 1 + 0.1 * (1000 + 55) = 1 + 105.5 = 106.5
        # (approximately, depending on exact learning)
        assert pred.mean == pytest.approx(105.5, rel=0.1)

    def test_mean_reversion_predicts_cumulative(self) -> None:
        """MeanReversionModel should predict cumulative path toward mean."""
        # Use fixed parameters to avoid learning dynamics changing them
        model = MeanReversionModel(mu=0.0, phi=0.5, decay=0.99)

        # Initialize with a single observation far from mu
        model.update(10.0, 0)

        # At this point: last_y = 10, mu = 10 (set to first obs)
        # So we need to manually set mu back to test reversion
        model.mu = 0.0  # Force mu to 0 for predictable test

        # Now x = last_y - mu = 10 - 0 = 10
        pred = model.predict(horizon=4)

        # Cumulative = 4*0 + 10 * 0.5 * (1 - 0.5^4) / (1 - 0.5)
        # = 10 * 0.5 * 0.9375 / 0.5 = 10 * 0.9375 = 9.375
        x = 10.0  # last_y - mu
        phi = 0.5
        expected = 4 * 0.0 + x * phi * (1 - phi**4) / (1 - phi)
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

        for i, y in enumerate(signal):
            aegis.observe("test", y)
            aegis.end_period()

        # Predict at h=1024
        pred = aegis.predict("test", horizon=1024)

        # True value at t = n-1 + 1024 = 1523
        future_t = n - 1 + 1024
        true_val = a * future_t**2 + b * future_t

        # Error should be small (< 10% of true value)
        error = abs(pred.mean - true_val)
        assert error < 0.1 * true_val, f"Error {error:.1f} too large (true={true_val:.1f})"
