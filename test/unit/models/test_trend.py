"""Unit tests for trend models (LocalTrend, DampedTrend)."""

import numpy as np
import pytest

from aegis.core.prediction import Prediction
from aegis.models.trend import DampedTrendModel, LocalTrendModel


class TestLocalTrendModel:
    """Tests for LocalTrendModel (Holt's method)."""

    def test_local_trend_captures_linear(self) -> None:
        """Test that model captures linear trend."""
        model = LocalTrendModel()

        for t in range(100):
            model.update(2.0 * t, t)

        pred = model.predict(horizon=10)
        expected = 2.0 * 100 + 2.0 * 10
        assert pred.mean == pytest.approx(expected, rel=0.15)

    def test_local_trend_extrapolates(self) -> None:
        """Test that prediction extrapolates trend."""
        model = LocalTrendModel()

        for t in range(50):
            model.update(float(t), t)

        pred_1 = model.predict(horizon=1)
        pred_10 = model.predict(horizon=10)

        assert pred_10.mean > pred_1.mean

    def test_local_trend_horizon_increases_variance(self) -> None:
        """Test that variance increases with horizon."""
        model = LocalTrendModel()

        for t in range(100):
            model.update(float(t) + np.random.normal(0, 0.5), t)

        pred_1 = model.predict(horizon=1)
        pred_10 = model.predict(horizon=10)

        assert pred_10.variance > pred_1.variance

    def test_local_trend_group(self) -> None:
        """Test model group is 'trend'."""
        model = LocalTrendModel()
        assert model.group == "trend"

    def test_local_trend_n_parameters(self) -> None:
        """Test model has 4 parameters (level, slope, sigma_sq, slope_var)."""
        model = LocalTrendModel()
        assert model.n_parameters == 4

    def test_local_trend_name(self) -> None:
        """Test model name."""
        model = LocalTrendModel()
        assert model.name == "LocalTrendModel"

    def test_local_trend_first_observation(self) -> None:
        """Test first observation initializes level."""
        model = LocalTrendModel()
        model.update(10.0, t=0)
        pred = model.predict(horizon=1)
        assert pred.mean == pytest.approx(10.0, abs=0.1)

    def test_local_trend_log_likelihood(self) -> None:
        """Test log-likelihood computation."""
        model = LocalTrendModel()
        for t in range(50):
            model.update(float(t), t)

        pred = model.predict(horizon=1)
        ll_at_pred = model.log_likelihood(pred.mean)
        ll_far = model.log_likelihood(pred.mean + 100)

        assert ll_at_pred > ll_far

    def test_local_trend_reset(self) -> None:
        """Test reset restores toward priors."""
        model = LocalTrendModel()
        for t in range(100):
            model.update(float(t), t)

        initial_slope = model.slope
        model.reset(partial=1.0)
        assert model.slope != initial_slope

    def test_local_trend_custom_alpha_beta(self) -> None:
        """Test custom smoothing parameters."""
        model = LocalTrendModel(alpha=0.5, beta=0.3)
        assert model.alpha == 0.5
        assert model.beta == 0.3

    def test_local_trend_returns_prediction_type(self) -> None:
        """Test that predict returns Prediction instance."""
        model = LocalTrendModel()
        model.update(1.0, t=0)
        pred = model.predict(horizon=1)
        assert isinstance(pred, Prediction)

    def test_local_trend_variance_grows_quadratically(self) -> None:
        """Variance should grow with h² for slope uncertainty.

        For trend extrapolation, prediction error scales as h * slope_error,
        so variance should scale as h². The ratio var(h=100)/var(h=10) should
        be significantly larger than 10 (linear) and approach 100 (quadratic).

        With a baseline offset from sigma_sq, the ratio won't reach exactly 100,
        but should be substantially larger than linear growth would give.
        """
        model = LocalTrendModel()
        for i in range(100):
            model.update(0.1 * i, t=i)  # Linear trend

        var_h10 = model.predict(horizon=10).variance
        var_h100 = model.predict(horizon=100).variance
        var_h1000 = model.predict(horizon=1000).variance

        # Quadratic: var(h=100)/var(h=10) should be ~100x in limit, but with
        # baseline offset we expect >20x (linear would be ~10x)
        ratio_10_to_100 = var_h100 / var_h10
        assert ratio_10_to_100 > 20, (
            f"Expected faster-than-linear growth, got ratio {ratio_10_to_100}"
        )

        # At longer horizons, quadratic term dominates more
        ratio_100_to_1000 = var_h1000 / var_h100
        assert ratio_100_to_1000 > 50, (
            f"Expected near-quadratic growth at long horizons, got ratio {ratio_100_to_1000}"
        )


class TestDampedTrendModel:
    """Tests for DampedTrendModel."""

    def test_damped_trend_limits_extrapolation(self) -> None:
        """Test that damping limits long-horizon extrapolation."""
        model_damped = DampedTrendModel(phi=0.9)
        model_undamped = LocalTrendModel()

        for t in range(100):
            y = float(t)
            model_damped.update(y, t)
            model_undamped.update(y, t)

        pred_damped = model_damped.predict(horizon=50)
        pred_undamped = model_undamped.predict(horizon=50)

        assert pred_damped.mean < pred_undamped.mean

    def test_damped_trend_phi_near_one_like_undamped(self) -> None:
        """Test phi near 1 behaves like undamped."""
        model = DampedTrendModel(phi=0.999)
        for t in range(100):
            model.update(float(t), t)

        pred = model.predict(horizon=10)
        assert pred.mean > 100

    def test_damped_trend_phi_zero_is_flat(self) -> None:
        """Test phi=0 gives flat prediction (no trend extrapolation)."""
        model = DampedTrendModel(phi=0.0)
        for t in range(100):
            model.update(float(t), t)

        pred_1 = model.predict(horizon=1)
        pred_10 = model.predict(horizon=10)

        assert abs(pred_10.mean - pred_1.mean) < 5

    def test_damped_trend_group(self) -> None:
        """Test model group is 'trend'."""
        model = DampedTrendModel()
        assert model.group == "trend"

    def test_damped_trend_n_parameters(self) -> None:
        """Test model has 5 parameters (level, slope, phi, sigma_sq, slope_var)."""
        model = DampedTrendModel()
        assert model.n_parameters == 5

    def test_damped_trend_name(self) -> None:
        """Test model name."""
        model = DampedTrendModel()
        assert model.name == "DampedTrendModel"

    def test_damped_trend_default_phi(self) -> None:
        """Test default phi value."""
        model = DampedTrendModel()
        assert model.phi == 0.9

    def test_damped_trend_custom_phi(self) -> None:
        """Test custom phi value."""
        model = DampedTrendModel(phi=0.8)
        assert model.phi == 0.8

    def test_damped_trend_log_likelihood(self) -> None:
        """Test log-likelihood computation."""
        model = DampedTrendModel()
        for t in range(50):
            model.update(float(t), t)

        pred = model.predict(horizon=1)
        ll_at_pred = model.log_likelihood(pred.mean)
        ll_far = model.log_likelihood(pred.mean + 100)

        assert ll_at_pred > ll_far

    def test_damped_trend_reset(self) -> None:
        """Test reset restores toward priors."""
        model = DampedTrendModel()
        for t in range(100):
            model.update(float(t), t)

        initial_slope = model.slope
        model.reset(partial=1.0)
        assert model.slope != initial_slope

    def test_damped_trend_convergence(self) -> None:
        """Test damped trend converges to asymptotic level."""
        model = DampedTrendModel(phi=0.8)
        for t in range(100):
            model.update(float(t), t)

        pred_100 = model.predict(horizon=100)
        pred_1000 = model.predict(horizon=1000)

        assert abs(pred_1000.mean - pred_100.mean) < abs(pred_100.mean - model.level)

    def test_damped_trend_returns_prediction_type(self) -> None:
        """Test that predict returns Prediction instance."""
        model = DampedTrendModel()
        model.update(1.0, t=0)
        pred = model.predict(horizon=1)
        assert isinstance(pred, Prediction)

    def test_damped_trend_variance_grows_quadratically(self) -> None:
        """Variance should grow with h² for slope uncertainty.

        Even with damped trend, the uncertainty in slope estimation
        should cause variance to grow quadratically with horizon.
        """
        model = DampedTrendModel(phi=0.9)
        for i in range(100):
            model.update(0.1 * i, t=i)  # Linear trend

        var_h10 = model.predict(horizon=10).variance
        var_h100 = model.predict(horizon=100).variance

        # Quadratic: var(h=100)/var(h=10) should be ~100x, not ~10x
        ratio_10_to_100 = var_h100 / var_h10
        assert ratio_10_to_100 > 50, f"Expected quadratic growth, got ratio {ratio_10_to_100}"
