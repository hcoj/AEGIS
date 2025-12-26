"""Unit tests for variance models (VolatilityTracker, LevelDependentVol)."""

import numpy as np
import pytest

from aegis.core.prediction import Prediction
from aegis.models.variance import LevelDependentVolModel, VolatilityTrackerModel


class TestVolatilityTrackerModel:
    """Tests for VolatilityTrackerModel."""

    def test_volatility_tracker_tracks_variance(self, white_noise_signal) -> None:
        """Test model tracks innovation variance."""
        signal = white_noise_signal(n=500, sigma=2.0)
        model = VolatilityTrackerModel()

        for t, y in enumerate(signal):
            model.update(y, t)

        expected_innovation_std = 2.0 * np.sqrt(2)
        assert np.sqrt(model.sigma_sq) == pytest.approx(expected_innovation_std, abs=0.8)

    def test_volatility_tracker_responds_to_change(self) -> None:
        """Test volatility updates when regime changes."""
        model = VolatilityTrackerModel()

        rng = np.random.default_rng(42)
        for t in range(100):
            model.update(rng.normal(0, 1.0), t)

        vol_before = model.sigma_sq

        for t in range(100, 200):
            model.update(rng.normal(0, 5.0), t)

        assert model.sigma_sq > vol_before

    def test_volatility_tracker_long_run_var(self) -> None:
        """Test long-run variance estimation."""
        model = VolatilityTrackerModel()

        rng = np.random.default_rng(42)
        for t in range(1000):
            model.update(rng.normal(0, 2.0), t)

        assert model.long_run_var > 0

    def test_volatility_tracker_ratio(self) -> None:
        """Test volatility ratio computation."""
        model = VolatilityTrackerModel()
        model.sigma_sq = 4.0
        model.long_run_var = 1.0

        ratio = model.get_volatility_ratio()
        assert ratio == pytest.approx(2.0, abs=0.01)

    def test_volatility_tracker_group(self) -> None:
        """Test model group is 'variance'."""
        model = VolatilityTrackerModel()
        assert model.group == "variance"

    def test_volatility_tracker_n_parameters(self) -> None:
        """Test parameter count."""
        model = VolatilityTrackerModel()
        assert model.n_parameters == 1

    def test_volatility_tracker_name(self) -> None:
        """Test model name."""
        model = VolatilityTrackerModel()
        assert model.name == "VolatilityTrackerModel"

    def test_volatility_tracker_custom_decay(self) -> None:
        """Test custom decay parameter."""
        model = VolatilityTrackerModel(decay=0.9)
        assert model.decay == 0.9

    def test_volatility_tracker_log_likelihood(self) -> None:
        """Test log-likelihood computation."""
        model = VolatilityTrackerModel()
        rng = np.random.default_rng(42)
        for t in range(50):
            model.update(rng.normal(0, 1), t)

        ll_at_last = model.log_likelihood(model.last_y)
        ll_far = model.log_likelihood(model.last_y + 100)

        assert ll_at_last > ll_far

    def test_volatility_tracker_reset(self) -> None:
        """Test reset toward long-run variance."""
        model = VolatilityTrackerModel()
        rng = np.random.default_rng(42)

        for t in range(100):
            model.update(rng.normal(0, 1), t)

        for t in range(100, 200):
            model.update(rng.normal(0, 10), t)

        high_vol = model.sigma_sq
        model.reset(partial=1.0)
        assert model.sigma_sq < high_vol

    def test_volatility_tracker_returns_prediction_type(self) -> None:
        """Test that predict returns Prediction instance."""
        model = VolatilityTrackerModel()
        model.update(1.0, t=0)
        pred = model.predict(horizon=1)
        assert isinstance(pred, Prediction)


class TestLevelDependentVolModel:
    """Tests for LevelDependentVolModel."""

    def test_level_dependent_vol_scales_with_level(self) -> None:
        """Test variance scales with signal level."""
        model = LevelDependentVolModel(gamma=0.5)
        model.sigma_sq_base = 1.0

        model.last_y = 1.0
        pred_low = model.predict(horizon=1)

        model.last_y = 100.0
        pred_high = model.predict(horizon=1)

        assert pred_high.variance > pred_low.variance

    def test_level_dependent_vol_learns_base_variance(self) -> None:
        """Test model learns base variance."""
        model = LevelDependentVolModel(gamma=0.5)

        rng = np.random.default_rng(42)
        for t in range(200):
            level = 100.0
            noise = rng.normal(0, 5 * np.sqrt(level))
            model.update(level + noise, t)

        assert model.sigma_sq_base > 0

    def test_level_dependent_vol_custom_gamma(self) -> None:
        """Test custom gamma parameter."""
        model = LevelDependentVolModel(gamma=1.0)
        assert model.gamma == 1.0

    def test_level_dependent_vol_group(self) -> None:
        """Test model group is 'variance'."""
        model = LevelDependentVolModel()
        assert model.group == "variance"

    def test_level_dependent_vol_n_parameters(self) -> None:
        """Test parameter count."""
        model = LevelDependentVolModel()
        assert model.n_parameters == 2

    def test_level_dependent_vol_name(self) -> None:
        """Test model name."""
        model = LevelDependentVolModel()
        assert model.name == "LevelDependentVolModel"

    def test_level_dependent_vol_handles_near_zero(self) -> None:
        """Test model handles values near zero safely."""
        model = LevelDependentVolModel()
        model.last_y = 0.001
        pred = model.predict(horizon=1)
        assert pred.variance > 0
        assert np.isfinite(pred.variance)

    def test_level_dependent_vol_log_likelihood(self) -> None:
        """Test log-likelihood computation."""
        model = LevelDependentVolModel()
        rng = np.random.default_rng(42)
        for t in range(50):
            model.update(50 + rng.normal(0, 5), t)

        pred = model.predict(horizon=1)
        ll_at_pred = model.log_likelihood(pred.mean)
        ll_far = model.log_likelihood(pred.mean + 100)

        assert ll_at_pred > ll_far

    def test_level_dependent_vol_reset(self) -> None:
        """Test reset restores base variance."""
        model = LevelDependentVolModel()
        rng = np.random.default_rng(42)
        for t in range(100):
            model.update(100 + rng.normal(0, 50), t)

        learned_base = model.sigma_sq_base
        model.reset(partial=1.0)
        assert model.sigma_sq_base != learned_base

    def test_level_dependent_vol_returns_prediction_type(self) -> None:
        """Test that predict returns Prediction instance."""
        model = LevelDependentVolModel()
        model.update(10.0, t=0)
        pred = model.predict(horizon=1)
        assert isinstance(pred, Prediction)

    def test_level_dependent_vol_variance_horizon(self) -> None:
        """Test variance scales with horizon."""
        model = LevelDependentVolModel()
        model.last_y = 100.0
        model.sigma_sq_base = 1.0

        pred_1 = model.predict(horizon=1)
        pred_10 = model.predict(horizon=10)

        assert pred_10.variance == pytest.approx(10 * pred_1.variance, rel=0.01)
