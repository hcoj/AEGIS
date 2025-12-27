"""Unit tests for trend models (LocalTrend, DampedTrend)."""

import numpy as np
import pytest

from aegis.core.prediction import Prediction
from aegis.models.trend import DampedTrendModel, LinearTrendModel, LocalTrendModel


class TestLocalTrendModel:
    """Tests for LocalTrendModel (Holt's method)."""

    def test_local_trend_captures_linear(self) -> None:
        """Test that model captures linear trend with cumulative predictions."""
        model = LocalTrendModel()

        for t in range(100):
            model.update(2.0 * t, t)

        pred = model.predict(horizon=10)
        # Cumulative: sum of predicted values from t=100 to t=109
        # = 200 + 202 + 204 + ... + 218 = 10 * 209 = 2090
        expected_cumulative = 10 * (2.0 * 100 + 2.0 * 109) / 2  # Arithmetic sum
        assert pred.mean == pytest.approx(expected_cumulative, rel=0.15)

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

        # With cumulative predictions, h=10 should be 10x h=1 for flat predictions
        assert pred_10.mean == pytest.approx(10 * pred_1.mean, rel=0.05)

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
        """Test damped trend per-step average converges to asymptotic value."""
        model = DampedTrendModel(phi=0.8)
        for t in range(100):
            model.update(float(t), t)

        pred_100 = model.predict(horizon=100)
        pred_1000 = model.predict(horizon=1000)

        # Per-step average converges to level + slope * phi / (1 - phi)
        asymptote = model.level + model.slope * model.phi / (1 - model.phi)
        avg_100 = pred_100.mean / 100
        avg_1000 = pred_1000.mean / 1000

        # avg_1000 should be closer to asymptote than avg_100
        assert abs(avg_1000 - asymptote) < abs(avg_100 - asymptote)

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


class TestLinearTrendModel:
    """Tests for LinearTrendModel (pure regression-based)."""

    def test_linear_trend_captures_slope(self) -> None:
        """Test that model captures linear slope from regression with cumulative predictions."""
        model = LinearTrendModel()
        for i in range(100):
            model.update(2.0 * i + 5.0, t=i)  # y = 2x + 5

        pred = model.predict(horizon=10)
        # Cumulative: sum of y at t=100, 101, ..., 109
        # = sum(2*t + 5) for t=100..109 = 2*sum(100..109) + 10*5
        # = 2 * (100+109)*10/2 + 50 = 2*1045 + 50 = 2140
        expected = 2.0 * (100 + 109) * 10 / 2 + 5.0 * 10
        assert pred.mean == pytest.approx(expected, rel=0.05)

    def test_linear_trend_captures_intercept(self) -> None:
        """Test that model captures intercept with cumulative prediction."""
        model = LinearTrendModel()
        for i in range(100):
            model.update(0.5 * i + 10.0, t=i)  # y = 0.5x + 10

        # At t=99, predict horizon=1 (cumulative = value at t=100)
        # y = 0.5*100 + 10 = 60
        pred = model.predict(horizon=1)
        expected = 0.5 * 100 + 10.0
        assert pred.mean == pytest.approx(expected, rel=0.05)

    def test_linear_trend_group_is_trend(self) -> None:
        """Test model group is 'trend'."""
        model = LinearTrendModel()
        assert model.group == "trend"

    def test_linear_trend_name(self) -> None:
        """Test model name."""
        model = LinearTrendModel()
        assert model.name == "LinearTrendModel"

    def test_linear_trend_n_parameters(self) -> None:
        """Test model has 4 parameters (intercept, slope, sigma_sq, slope_var)."""
        model = LinearTrendModel()
        assert model.n_parameters == 4

    def test_linear_trend_variance_quadratic(self) -> None:
        """Test that variance grows quadratically with horizon."""
        np.random.seed(42)
        model = LinearTrendModel()
        for i in range(100):
            model.update(0.5 * i + np.random.normal(0, 0.1), t=i)

        var_h10 = model.predict(horizon=10).variance
        var_h100 = model.predict(horizon=100).variance

        ratio = var_h100 / var_h10
        # Quadratic growth: ratio should be ~100x
        assert ratio > 50, f"Expected quadratic growth, got ratio {ratio}"

    def test_linear_trend_returns_prediction_type(self) -> None:
        """Test that predict returns Prediction instance."""
        model = LinearTrendModel()
        model.update(1.0, t=0)
        pred = model.predict(horizon=1)
        assert isinstance(pred, Prediction)

    def test_linear_trend_log_likelihood(self) -> None:
        """Test log-likelihood computation."""
        model = LinearTrendModel()
        for t in range(50):
            model.update(float(t), t)

        pred = model.predict(horizon=1)
        ll_at_pred = model.log_likelihood(pred.mean)
        ll_far = model.log_likelihood(pred.mean + 100)

        assert ll_at_pred > ll_far

    def test_linear_trend_reset(self) -> None:
        """Test reset restores toward priors."""
        model = LinearTrendModel()
        for t in range(100):
            model.update(float(t), t)

        initial_slope = model.slope
        model.reset(partial=1.0)
        assert model.slope != initial_slope

    def test_linear_trend_first_observation(self) -> None:
        """Test first observation initializes model."""
        model = LinearTrendModel()
        model.update(10.0, t=0)
        pred = model.predict(horizon=1)
        # With one point, predict same value
        assert pred.mean == pytest.approx(10.0, abs=1.0)

    def test_linear_trend_with_noise(self) -> None:
        """Test model handles noisy linear data with cumulative predictions."""
        np.random.seed(123)
        model = LinearTrendModel()
        for i in range(200):
            y = 1.5 * i + 20.0 + np.random.normal(0, 2.0)
            model.update(y, t=i)

        pred = model.predict(horizon=10)
        # Cumulative: sum of y at t=200, 201, ..., 209
        # = sum(1.5*t + 20) for t=200..209 = 1.5*sum(200..209) + 10*20
        # = 1.5 * (200+209)*10/2 + 200 = 1.5*2045 + 200 = 3267.5
        expected = 1.5 * (200 + 209) * 10 / 2 + 20.0 * 10
        assert pred.mean == pytest.approx(expected, rel=0.1)


class TestNumericalStability:
    """Tests for numerical stability with extreme values."""

    def test_local_trend_handles_polynomial_growth(self) -> None:
        """LocalTrendModel should not produce NaN for polynomial growth."""
        model = LocalTrendModel()

        # Simulate polynomial trend (quadratic growth)
        for i in range(500):
            y = 0.01 * i**2
            model.update(y, t=i)
            pred = model.predict(horizon=1)
            assert not np.isnan(pred.mean), f"NaN mean at t={i}"
            assert not np.isnan(pred.variance), f"NaN variance at t={i}"
            assert not np.isinf(pred.variance), f"Inf variance at t={i}"

    def test_local_trend_handles_large_values(self) -> None:
        """LocalTrendModel should handle large values gracefully."""
        model = LocalTrendModel()

        # Large values that might cause overflow
        for i in range(100):
            y = 1e6 + i * 1e4
            model.update(y, t=i)
            pred = model.predict(horizon=1)
            assert np.isfinite(pred.mean), f"Non-finite mean at t={i}"
            assert np.isfinite(pred.variance), f"Non-finite variance at t={i}"
            ll = model.log_likelihood(y)
            assert np.isfinite(ll), f"Non-finite log-likelihood at t={i}"

    def test_damped_trend_handles_polynomial_growth(self) -> None:
        """DampedTrendModel should not produce NaN for polynomial growth."""
        model = DampedTrendModel()

        for i in range(500):
            y = 0.01 * i**2
            model.update(y, t=i)
            pred = model.predict(horizon=1)
            assert not np.isnan(pred.mean), f"NaN mean at t={i}"
            assert not np.isnan(pred.variance), f"NaN variance at t={i}"

    def test_linear_trend_handles_polynomial_growth(self) -> None:
        """LinearTrendModel should not produce NaN for polynomial growth."""
        model = LinearTrendModel()

        for i in range(500):
            y = 0.01 * i**2
            model.update(y, t=i)
            pred = model.predict(horizon=1)
            assert not np.isnan(pred.mean), f"NaN mean at t={i}"
            assert not np.isnan(pred.variance), f"NaN variance at t={i}"

    def test_models_handle_extreme_variance(self) -> None:
        """Models should cap variance to prevent overflow."""
        model = LocalTrendModel()

        # Feed data that might cause extreme variance
        for i in range(100):
            # Alternating large jumps
            y = 1e8 if i % 2 == 0 else -1e8
            model.update(y, t=i)
            pred = model.predict(horizon=1)
            assert np.isfinite(pred.variance), f"Non-finite variance at t={i}"
            assert pred.variance < 1e20, f"Variance too large at t={i}: {pred.variance}"
