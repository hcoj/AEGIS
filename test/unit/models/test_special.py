"""Unit tests for special models (JumpDiffusion, ChangePoint)."""

import numpy as np
import pytest

from aegis.core.prediction import Prediction
from aegis.models.special import ChangePointModel, JumpDiffusionModel


class TestJumpDiffusionModel:
    """Tests for JumpDiffusionModel."""

    def test_jump_diffusion_detects_jumps(self, jump_diffusion_signal) -> None:
        """Test model detects jump events."""
        signal = jump_diffusion_signal(n=500, jump_prob=0.05, jump_size=5.0)
        model = JumpDiffusionModel()

        for t, y in enumerate(signal):
            model.update(y, t)

        lambda_est = model.lambda_mean()
        assert 0.01 < lambda_est < 0.15

    def test_jump_diffusion_prediction_includes_jump_risk(self) -> None:
        """Test prediction variance includes jump risk."""
        model = JumpDiffusionModel()
        model.last_y = 0.0
        model.sigma_sq_diff = 1.0
        model.sigma_sq_jump = 10.0
        model.lambda_a = 5.0
        model.lambda_b = 95.0

        pred = model.predict(horizon=1)
        assert pred.variance > model.sigma_sq_diff

    def test_jump_diffusion_lambda_updates(self) -> None:
        """Test lambda (jump probability) updates correctly."""
        model = JumpDiffusionModel()
        initial_lambda = model.lambda_mean()

        model.update(0.0, 0)
        for t in range(1, 20):
            model.update(0.0 + (10.0 if t == 10 else 0.0), t)

        assert model.lambda_mean() != initial_lambda

    def test_jump_diffusion_group(self) -> None:
        """Test model group is 'special'."""
        model = JumpDiffusionModel()
        assert model.group == "special"

    def test_jump_diffusion_n_parameters(self) -> None:
        """Test parameter count."""
        model = JumpDiffusionModel()
        assert model.n_parameters == 4

    def test_jump_diffusion_name(self) -> None:
        """Test model name."""
        model = JumpDiffusionModel()
        assert model.name == "JumpDiffusionModel"

    def test_jump_diffusion_epistemic_value(self) -> None:
        """Test epistemic value computation."""
        model = JumpDiffusionModel()
        ev1 = model.epistemic_value()

        for t in range(100):
            model.update(float(t % 2), t)

        ev2 = model.epistemic_value()
        assert ev2 < ev1

    def test_jump_diffusion_log_likelihood(self) -> None:
        """Test log-likelihood computation."""
        model = JumpDiffusionModel()
        for t in range(50):
            model.update(np.random.normal(0, 1), t)

        pred = model.predict(horizon=1)
        ll_at_pred = model.log_likelihood(pred.mean)
        ll_far = model.log_likelihood(pred.mean + 100)

        assert ll_at_pred > ll_far

    def test_jump_diffusion_reset(self) -> None:
        """Test reset clears jump history."""
        model = JumpDiffusionModel()
        for t in range(50):
            model.update(10.0 * np.sin(t), t)

        model.reset(partial=1.0)
        assert len(model.recent_jumps) == 0

    def test_jump_diffusion_returns_prediction_type(self) -> None:
        """Test that predict returns Prediction instance."""
        model = JumpDiffusionModel()
        model.update(1.0, t=0)
        pred = model.predict(horizon=1)
        assert isinstance(pred, Prediction)


class TestChangePointModel:
    """Tests for ChangePointModel."""

    def test_change_point_detects_regime_shift(self, regime_switching_signal) -> None:
        """Test model detects regime changes."""
        signal = regime_switching_signal(n=500, mean1=0.0, mean2=10.0)
        model = ChangePointModel()

        for t, y in enumerate(signal):
            model.update(y, t)

        assert model.hazard_a + model.hazard_b > 100

    def test_change_point_learns_regime_mean(self) -> None:
        """Test model learns current regime mean."""
        model = ChangePointModel()

        for t in range(100):
            model.update(5.0 + np.random.normal(0, 0.1), t)

        assert model.mu == pytest.approx(5.0, abs=0.5)

    def test_change_point_run_length_tracks(self) -> None:
        """Test run length increases in stable regime."""
        model = ChangePointModel()

        for t in range(50):
            model.update(0.0 + np.random.normal(0, 0.1), t)

        assert model.run_length > 0

    def test_change_point_group(self) -> None:
        """Test model group is 'special'."""
        model = ChangePointModel()
        assert model.group == "special"

    def test_change_point_n_parameters(self) -> None:
        """Test parameter count."""
        model = ChangePointModel()
        assert model.n_parameters == 3

    def test_change_point_name(self) -> None:
        """Test model name."""
        model = ChangePointModel()
        assert model.name == "ChangePointModel"

    def test_change_point_custom_hazard(self) -> None:
        """Test custom hazard rate."""
        model = ChangePointModel(hazard_rate=0.05)
        assert model.hazard_rate == 0.05

    def test_change_point_epistemic_value(self) -> None:
        """Test epistemic value computation."""
        model = ChangePointModel()
        ev = model.epistemic_value()
        assert ev >= 0

    def test_change_point_log_likelihood(self) -> None:
        """Test log-likelihood computation."""
        model = ChangePointModel()
        for t in range(50):
            model.update(np.random.normal(0, 1), t)

        ll_at_mean = model.log_likelihood(model.mu)
        ll_far = model.log_likelihood(model.mu + 100)

        assert ll_at_mean > ll_far

    def test_change_point_reset(self) -> None:
        """Test reset restores hazard prior."""
        model = ChangePointModel()
        for t in range(100):
            model.update(float(t), t)

        model.reset(partial=1.0)
        assert model.hazard_a == pytest.approx(1.0, abs=0.5)

    def test_change_point_returns_prediction_type(self) -> None:
        """Test that predict returns Prediction instance."""
        model = ChangePointModel()
        model.update(1.0, t=0)
        pred = model.predict(horizon=1)
        assert isinstance(pred, Prediction)

    def test_change_point_prediction_variance_increases(self) -> None:
        """Test prediction variance increases with horizon due to change risk."""
        model = ChangePointModel()
        for t in range(50):
            model.update(0.0, t)

        pred_1 = model.predict(horizon=1)
        pred_10 = model.predict(horizon=10)

        assert pred_10.variance > pred_1.variance
