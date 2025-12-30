"""Unit tests for dynamic models (AR2, MA1)."""

import numpy as np
import pytest

from aegis.core.prediction import Prediction
from aegis.models.dynamic import AR2Model, MA1Model


class TestAR2Model:
    """Tests for AR2Model."""

    def test_ar2_captures_oscillation(self) -> None:
        """Test AR(2) captures oscillatory dynamics."""
        model = AR2Model()

        rng = np.random.default_rng(42)
        y = np.zeros(500)
        true_phi1, true_phi2 = 0.5, -0.3
        for t in range(2, 500):
            y[t] = true_phi1 * y[t - 1] + true_phi2 * y[t - 2] + rng.normal(0, 0.5)

        for t, val in enumerate(y):
            model.update(val, t)

        assert model.phi1 == pytest.approx(true_phi1, abs=0.2)
        assert model.phi2 == pytest.approx(true_phi2, abs=0.2)

    def test_ar2_prediction(self) -> None:
        """Test AR(2) multi-step prediction."""
        model = AR2Model()
        model.phi1 = 0.8
        model.phi2 = -0.3
        model.c = 0.0
        model.y_lag1 = 2.0
        model.y_lag2 = 1.0
        model._n_obs = 10

        pred = model.predict(horizon=1)
        expected = 0.8 * 2.0 - 0.3 * 1.0
        assert pred.mean == pytest.approx(expected, abs=0.1)

    def test_ar2_stationarity_constraint(self) -> None:
        """Test AR(2) enforces stationarity."""
        model = AR2Model()

        rng = np.random.default_rng(123)
        for t in range(200):
            model.update(rng.normal(0, 10), t)

        assert abs(model.phi1) + abs(model.phi2) <= 1.0

    def test_ar2_group(self) -> None:
        """Test model group is 'dynamic'."""
        model = AR2Model()
        assert model.group == "dynamic"

    def test_ar2_n_parameters(self) -> None:
        """Test parameter count (c, phi1, phi2, variance)."""
        model = AR2Model()
        assert model.n_parameters == 4

    def test_ar2_name(self) -> None:
        """Test model name."""
        model = AR2Model()
        assert model.name == "AR2Model"

    def test_ar2_log_likelihood(self) -> None:
        """Test log-likelihood computation."""
        model = AR2Model()
        rng = np.random.default_rng(42)
        for t in range(50):
            model.update(rng.normal(0, 1), t)

        pred = model.predict(horizon=1)
        ll_at_pred = model.log_likelihood(pred.mean)
        ll_far = model.log_likelihood(pred.mean + 100)

        assert ll_at_pred > ll_far

    def test_ar2_reset(self) -> None:
        """Test reset restores toward priors."""
        model = AR2Model()
        rng = np.random.default_rng(42)
        for t in range(100):
            model.update(rng.normal(0, 5), t)

        model.reset(partial=1.0)
        assert model.phi1 == pytest.approx(0.5, abs=0.1)
        assert model.phi2 == pytest.approx(0.3, abs=0.1)

    def test_ar2_handles_early_observations(self) -> None:
        """Test AR(2) handles observations before 2 lags."""
        model = AR2Model()
        model.update(1.0, t=0)
        model.update(2.0, t=1)
        pred = model.predict(horizon=1)
        assert isinstance(pred, Prediction)

    def test_ar2_returns_prediction_type(self) -> None:
        """Test that predict returns Prediction instance."""
        model = AR2Model()
        for t in range(5):
            model.update(float(t), t)
        pred = model.predict(horizon=1)
        assert isinstance(pred, Prediction)

    def test_ar2_constant_term_bounded_for_trending_signal(self) -> None:
        """AR2Model constant term c stays bounded for trending signals."""
        model = AR2Model()
        # 10000 observations like comprehensive evaluation
        for t in range(10000):
            y = 0.0001 * t**2 + 0.05 * t  # Polynomial trend
            model.update(y, t)

        assert np.isfinite(model.c), "Constant term c overflowed to NaN/Inf"
        assert abs(model.c) < 1e6, f"Constant term c unbounded: {model.c}"

    def test_ar2_prediction_finite_for_trending_signal(self) -> None:
        """AR2Model predictions stay finite for trending signals."""
        model = AR2Model()
        # 10000 observations like comprehensive evaluation
        for t in range(10000):
            y = 0.05 * t  # Linear trend
            model.update(y, t)

        pred = model.predict(horizon=1024)
        assert np.isfinite(pred.mean), f"Prediction mean is NaN/Inf: {pred.mean}"
        assert np.isfinite(pred.variance), f"Prediction variance is NaN/Inf: {pred.variance}"


class TestMA1Model:
    """Tests for MA1Model."""

    def test_ma1_captures_shock(self) -> None:
        """Test MA(1) captures shock effects."""
        model = MA1Model(theta=0.5)

        rng = np.random.default_rng(42)
        errors = rng.normal(0, 1, 200)
        y = np.zeros(200)
        for t in range(1, 200):
            y[t] = 0.7 * errors[t - 1] + errors[t]

        for t, val in enumerate(y):
            model.update(val, t)

        assert model.theta == pytest.approx(0.7, abs=0.3)

    def test_ma1_h1_prediction(self) -> None:
        """Test MA(1) horizon=1 prediction uses last error."""
        model = MA1Model()
        model.theta = 0.5
        model.last_error = 2.0

        pred = model.predict(horizon=1)
        assert pred.mean == pytest.approx(1.0, abs=0.01)

    def test_ma1_point_prediction_decays(self) -> None:
        """Test MA(1) point prediction at h>1 is zero (no predictive power)."""
        model = MA1Model()
        model.theta = 0.5
        model.last_error = 2.0

        # MA(1) only has impact at h=1, h>1 predicts 0
        pred_1 = model.predict(horizon=1)
        pred_5 = model.predict(horizon=5)
        assert pred_1.mean == pytest.approx(1.0, abs=0.01)  # theta * last_error
        assert pred_5.mean == pytest.approx(0.0, abs=0.01)  # No predictive power at h>1

    def test_ma1_group(self) -> None:
        """Test model group is 'dynamic'."""
        model = MA1Model()
        assert model.group == "dynamic"

    def test_ma1_n_parameters(self) -> None:
        """Test parameter count (theta, variance)."""
        model = MA1Model()
        assert model.n_parameters == 2

    def test_ma1_name(self) -> None:
        """Test model name."""
        model = MA1Model()
        assert model.name == "MA1Model"

    def test_ma1_custom_theta(self) -> None:
        """Test custom initial theta."""
        model = MA1Model(theta=0.3)
        assert model.theta == pytest.approx(0.3, abs=0.01)

    def test_ma1_theta_bounded(self) -> None:
        """Test theta stays in valid range."""
        model = MA1Model()
        rng = np.random.default_rng(42)
        for t in range(200):
            model.update(rng.normal(0, 10), t)

        assert -1.0 < model.theta < 1.0

    def test_ma1_log_likelihood(self) -> None:
        """Test log-likelihood computation."""
        model = MA1Model()
        rng = np.random.default_rng(42)
        for t in range(50):
            model.update(rng.normal(0, 1), t)

        pred = model.predict(horizon=1)
        ll_at_pred = model.log_likelihood(pred.mean)
        ll_far = model.log_likelihood(pred.mean + 100)

        assert ll_at_pred > ll_far

    def test_ma1_reset(self) -> None:
        """Test reset restores toward priors."""
        model = MA1Model(theta=0.0)
        for t in range(100):
            model.update(float(t), t)

        model.reset(partial=1.0)
        assert model.last_error == 0.0

    def test_ma1_returns_prediction_type(self) -> None:
        """Test that predict returns Prediction instance."""
        model = MA1Model()
        model.update(1.0, t=0)
        pred = model.predict(horizon=1)
        assert isinstance(pred, Prediction)

    def test_ma1_learns_near_zero_theta_for_white_noise(self) -> None:
        """MA1Model should learn theta near 0 for white noise.

        White noise has uncorrelated errors, so optimal MA(1) theta is 0.
        This tests that MA1 doesn't incorrectly learn spurious patterns.
        """
        rng = np.random.default_rng(42)
        model = MA1Model()

        for t in range(1000):
            y = rng.standard_normal()  # White noise
            model.update(y, t)

        # For white noise, optimal MA(1) theta should be near 0
        assert abs(model.theta) < 0.3, f"MA1 learned theta={model.theta} for white noise"

    def test_ma1_log_likelihood_consistent_with_predict(self) -> None:
        """MA1Model log_likelihood should use same variance as predict(1)."""
        model = MA1Model()
        rng = np.random.default_rng(42)

        for t in range(100):
            model.update(rng.normal(0, 1), t)

        y_test = 0.5
        pred = model.predict(horizon=1)

        # Compute expected log-likelihood from predict()
        expected_ll = (
            -0.5 * np.log(2 * np.pi * pred.variance)
            - 0.5 * (y_test - pred.mean) ** 2 / pred.variance
        )
        actual_ll = model.log_likelihood(y_test)

        assert np.isclose(actual_ll, expected_ll)
