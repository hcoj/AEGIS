"""Unit tests for reversion models (MeanReversion, AsymmetricMR, ThresholdAR)."""

import numpy as np
import pytest

from aegis.core.prediction import Prediction
from aegis.models.reversion import (
    AsymmetricMeanReversionModel,
    MeanReversionModel,
    ThresholdARModel,
)


class TestMeanReversionModel:
    """Tests for MeanReversionModel."""

    def test_mean_reversion_predicts_toward_mean(self, ar1_signal) -> None:
        """Test prediction reverts toward learned mean."""
        signal = ar1_signal(n=500, phi=0.9)
        model = MeanReversionModel()

        for t, y in enumerate(signal):
            model.update(y, t)

        model._last_y = 5.0
        pred = model.predict(horizon=1)

        assert pred.mean < 5.0
        assert pred.mean > model.mu

    def test_mean_reversion_phi_constrained(self) -> None:
        """Test phi stays in valid range."""
        model = MeanReversionModel()
        assert 0 <= model.phi <= 0.999

    def test_mean_reversion_horizon_prediction(self) -> None:
        """Test multi-step cumulative prediction shows reversion in per-step average."""
        model = MeanReversionModel()
        model.mu = 0.0
        model._last_y = 1.0

        pred_1 = model.predict(horizon=1)
        pred_10 = model.predict(horizon=10)

        # Per-step average should decrease as we revert toward mu
        avg_per_step_1 = pred_1.mean / 1
        avg_per_step_10 = pred_10.mean / 10

        assert abs(avg_per_step_10) < abs(avg_per_step_1)

    def test_mean_reversion_learns_mean(self, ar1_signal) -> None:
        """Test model learns the process mean."""
        signal = ar1_signal(n=500, phi=0.9)
        signal = signal + 5.0
        model = MeanReversionModel()

        for t, y in enumerate(signal):
            model.update(y, t)

        assert model.mu == pytest.approx(5.0, abs=2.0)

    def test_mean_reversion_group(self) -> None:
        """Test model group is 'reversion'."""
        model = MeanReversionModel()
        assert model.group == "reversion"

    def test_mean_reversion_n_parameters(self) -> None:
        """Test model has 3 parameters (mu, phi, variance)."""
        model = MeanReversionModel()
        assert model.n_parameters == 3

    def test_mean_reversion_name(self) -> None:
        """Test model name."""
        model = MeanReversionModel()
        assert model.name == "MeanReversionModel"

    def test_mean_reversion_log_likelihood(self) -> None:
        """Test log-likelihood computation."""
        model = MeanReversionModel()
        for t in range(50):
            model.update(np.random.normal(0, 1), t)

        pred = model.predict(horizon=1)
        ll_at_pred = model.log_likelihood(pred.mean)
        ll_far = model.log_likelihood(pred.mean + 100)

        assert ll_at_pred > ll_far

    def test_mean_reversion_reset(self) -> None:
        """Test reset restores toward priors."""
        model = MeanReversionModel()
        for t in range(100):
            model.update(float(t), t)

        model.reset(partial=1.0)
        assert model.mu == pytest.approx(0.0, abs=0.1)

    def test_mean_reversion_epistemic_value(self) -> None:
        """Test epistemic value is computed."""
        model = MeanReversionModel()
        model._last_y = 5.0
        model.mu = 0.0
        model.phi_var = 0.01

        ev = model.epistemic_value()
        assert ev >= 0

    def test_mean_reversion_returns_prediction_type(self) -> None:
        """Test that predict returns Prediction instance."""
        model = MeanReversionModel()
        model.update(1.0, t=0)
        pred = model.predict(horizon=1)
        assert isinstance(pred, Prediction)

    def test_mean_reversion_learns_phi_for_ar1(self) -> None:
        """MeanReversionModel learns correct phi for AR(1) signal.

        For AR(1) process y_t = phi * y_{t-1} + epsilon, the model
        should learn phi close to the true value.
        """
        rng = np.random.default_rng(42)
        true_phi = 0.8
        y = 0.0
        model = MeanReversionModel()

        for t in range(500):
            y = true_phi * y + rng.standard_normal() * 0.3
            model.update(y, t)

        # After 500 observations, phi should be close to 0.8
        assert model.phi > 0.6, f"MeanReversion learned phi={model.phi}, expected ~0.8"


class TestAsymmetricMeanReversionModel:
    """Tests for AsymmetricMeanReversionModel."""

    def test_asymmetric_mr_learns_different_speeds(self, asymmetric_ar1_signal) -> None:
        """Test model learns different reversion speeds above/below mean."""
        signal = asymmetric_ar1_signal(n=2000, phi_up=0.5, phi_down=0.95)
        model = AsymmetricMeanReversionModel()

        for t, y in enumerate(signal):
            model.update(y, t)

        assert model.phi_up != model.phi_down
        assert abs(model.phi_up - model.phi_down) > 0.05

    def test_asymmetric_mr_prediction_direction(self) -> None:
        """Test prediction uses correct phi based on position."""
        model = AsymmetricMeanReversionModel()
        model.mu = 0.0
        model.phi_up = 0.5
        model.phi_down = 0.9

        model._last_y = 2.0
        pred_above = model.predict(horizon=1)

        model._last_y = -2.0
        pred_below = model.predict(horizon=1)

        assert abs(pred_above.mean) < abs(pred_below.mean)

    def test_asymmetric_mr_group(self) -> None:
        """Test model group is 'reversion'."""
        model = AsymmetricMeanReversionModel()
        assert model.group == "reversion"

    def test_asymmetric_mr_n_parameters(self) -> None:
        """Test model has 4 parameters."""
        model = AsymmetricMeanReversionModel()
        assert model.n_parameters == 4

    def test_asymmetric_mr_name(self) -> None:
        """Test model name."""
        model = AsymmetricMeanReversionModel()
        assert model.name == "AsymmetricMeanReversionModel"

    def test_asymmetric_mr_log_likelihood(self) -> None:
        """Test log-likelihood computation."""
        model = AsymmetricMeanReversionModel()
        for t in range(50):
            model.update(np.random.normal(0, 1), t)

        pred = model.predict(horizon=1)
        ll_at_pred = model.log_likelihood(pred.mean)
        ll_far = model.log_likelihood(pred.mean + 100)

        assert ll_at_pred > ll_far

    def test_asymmetric_mr_reset(self) -> None:
        """Test reset restores toward priors."""
        model = AsymmetricMeanReversionModel()
        for t in range(100):
            model.update(float(t) if t % 2 == 0 else -float(t), t)

        model.reset(partial=1.0)
        assert model.phi_up == pytest.approx(model.prior_phi_up, abs=0.01)

    def test_asymmetric_mr_returns_prediction_type(self) -> None:
        """Test that predict returns Prediction instance."""
        model = AsymmetricMeanReversionModel()
        model.update(1.0, t=0)
        pred = model.predict(horizon=1)
        assert isinstance(pred, Prediction)


class TestThresholdARModel:
    """Tests for ThresholdARModel."""

    def test_threshold_ar_regime_dependent(self) -> None:
        """Test different dynamics above/below threshold."""
        model = ThresholdARModel(tau=0.0)
        model.phi_low = 0.5
        model.phi_high = 0.95

        model._last_y = -2.0
        pred_low = model.predict(horizon=1)

        model._last_y = 2.0
        pred_high = model.predict(horizon=1)

        assert abs(pred_high.mean) > abs(pred_low.mean)

    def test_threshold_ar_learns_thresholds(self, threshold_ar_signal) -> None:
        """Test model learns regime-specific dynamics."""
        signal = threshold_ar_signal(n=1000, tau=0.0, phi_low=0.5, phi_high=0.9)
        model = ThresholdARModel(tau=0.0)

        for t, y in enumerate(signal):
            model.update(y, t)

        assert model.phi_low < model.phi_high

    def test_threshold_ar_group(self) -> None:
        """Test model group is 'reversion'."""
        model = ThresholdARModel()
        assert model.group == "reversion"

    def test_threshold_ar_n_parameters(self) -> None:
        """Test model has 4 parameters."""
        model = ThresholdARModel()
        assert model.n_parameters == 4

    def test_threshold_ar_name(self) -> None:
        """Test model name."""
        model = ThresholdARModel()
        assert model.name == "ThresholdARModel"

    def test_threshold_ar_custom_tau(self) -> None:
        """Test custom threshold value."""
        model = ThresholdARModel(tau=5.0)
        assert model.tau == 5.0

    def test_threshold_ar_epistemic_peaks_near_threshold(self) -> None:
        """Test epistemic value is higher near threshold."""
        model = ThresholdARModel(tau=0.0)

        model._last_y = -5.0
        ev_below = model.epistemic_value()

        model._last_y = 0.1
        ev_near = model.epistemic_value()

        model._last_y = 5.0
        ev_above = model.epistemic_value()

        assert ev_near >= ev_below
        assert ev_near >= ev_above

    def test_threshold_ar_log_likelihood(self) -> None:
        """Test log-likelihood computation."""
        model = ThresholdARModel()
        for t in range(50):
            model.update(np.random.normal(0, 1), t)

        pred = model.predict(horizon=1)
        ll_at_pred = model.log_likelihood(pred.mean)
        ll_far = model.log_likelihood(pred.mean + 100)

        assert ll_at_pred > ll_far

    def test_threshold_ar_reset(self) -> None:
        """Test reset restores toward priors."""
        model = ThresholdARModel()
        for t in range(100):
            model.update(np.random.normal(0, 5), t)

        model.reset(partial=1.0)
        assert model.phi_low == pytest.approx(model.prior_phi_low, abs=0.01)

    def test_threshold_ar_returns_prediction_type(self) -> None:
        """Test that predict returns Prediction instance."""
        model = ThresholdARModel()
        model.update(1.0, t=0)
        pred = model.predict(horizon=1)
        assert isinstance(pred, Prediction)


class TestLevelAwareMeanReversionModel:
    """Tests for LevelAwareMeanReversionModel."""

    def test_level_aware_tracks_level(self) -> None:
        """Test that level is tracked as cumulative sum of inputs."""
        from aegis.models.reversion import LevelAwareMeanReversionModel

        model = LevelAwareMeanReversionModel()

        model.update(1.0, 0)
        assert model.level == 1.0

        model.update(2.0, 1)
        assert model.level == 3.0

        model.update(-1.0, 2)
        assert model.level == 2.0

    def test_level_aware_predicts_reversion(self) -> None:
        """Test prediction reverts toward mean level."""
        from aegis.models.reversion import LevelAwareMeanReversionModel

        model = LevelAwareMeanReversionModel(mu=0.0, phi=0.8)
        model.level = 10.0

        pred = model.predict(horizon=1)
        # Deviation is 10.0, expected level is 0 + 0.8*10 = 8
        # Return is 8 - 10 = -2
        assert pred.mean < 0, "Should predict negative return to revert toward mu"

    def test_level_aware_learns_from_ou_signal(self) -> None:
        """Test model correctly identifies mean reversion in OU-like signal.

        When given returns from a mean-reverting level process, the model
        should learn that the level reverts, not the returns.
        """
        from aegis.models.reversion import LevelAwareMeanReversionModel

        rng = np.random.default_rng(42)
        model = LevelAwareMeanReversionModel(phi=0.5)

        # Simulate OU process and feed returns
        level = 0.0
        mu = 0.0
        true_phi = 0.9
        sigma = 0.1

        for t in range(500):
            next_level = mu + true_phi * (level - mu) + rng.normal(0, sigma)
            return_val = next_level - level
            model.update(return_val, t)
            level = next_level

        # Model should have learned high phi (level persists, reversion is slow)
        assert model.phi > 0.5, f"Should learn high phi for OU, got {model.phi}"

    def test_level_aware_group(self) -> None:
        """Test model group is 'reversion'."""
        from aegis.models.reversion import LevelAwareMeanReversionModel

        model = LevelAwareMeanReversionModel()
        assert model.group == "reversion"

    def test_level_aware_name(self) -> None:
        """Test model name."""
        from aegis.models.reversion import LevelAwareMeanReversionModel

        model = LevelAwareMeanReversionModel()
        assert model.name == "LevelAwareMeanReversionModel"

    def test_level_aware_n_parameters(self) -> None:
        """Test model has 3 parameters."""
        from aegis.models.reversion import LevelAwareMeanReversionModel

        model = LevelAwareMeanReversionModel()
        assert model.n_parameters == 3

    def test_level_aware_returns_prediction_type(self) -> None:
        """Test that predict returns Prediction instance."""
        from aegis.models.reversion import LevelAwareMeanReversionModel

        model = LevelAwareMeanReversionModel()
        model.update(1.0, t=0)
        pred = model.predict(horizon=1)
        assert isinstance(pred, Prediction)

    def test_level_aware_reset(self) -> None:
        """Test reset restores toward priors and resets level."""
        from aegis.models.reversion import LevelAwareMeanReversionModel

        model = LevelAwareMeanReversionModel(mu=0.0)

        for t in range(100):
            model.update(1.0, t)

        assert model.level > 50

        model.reset(partial=1.0)
        assert model.level == pytest.approx(model.mu, abs=0.01)
