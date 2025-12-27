"""Unit tests for persistence models (RandomWalk, LocalLevel)."""

import numpy as np
import pytest

from aegis.core.prediction import Prediction
from aegis.models.persistence import LocalLevelModel, RandomWalkModel


class TestRandomWalkModel:
    """Tests for RandomWalkModel."""

    def test_random_walk_predicts_last_value(self) -> None:
        """Test that prediction mean equals last observed value."""
        model = RandomWalkModel()
        model.update(5.0, t=0)
        pred = model.predict(horizon=1)
        assert pred.mean == 5.0

    def test_random_walk_multiple_updates(self) -> None:
        """Test prediction after multiple updates."""
        model = RandomWalkModel()
        for i, y in enumerate([1.0, 2.0, 3.0, 4.0, 5.0]):
            model.update(y, t=i)
        pred = model.predict(horizon=1)
        assert pred.mean == 5.0

    def test_random_walk_variance_scales_with_horizon(self) -> None:
        """Test that variance scales linearly with horizon."""
        model = RandomWalkModel()
        for i, y in enumerate([1.0, 2.0, 1.5, 2.5, 2.0]):
            model.update(y, t=i)

        pred_1 = model.predict(horizon=1)
        pred_5 = model.predict(horizon=5)

        assert pred_5.variance == pytest.approx(5 * pred_1.variance, rel=0.01)

    def test_random_walk_log_likelihood_higher_at_mean(self) -> None:
        """Test that log-likelihood is higher at predicted mean."""
        model = RandomWalkModel()
        model.update(0.0, t=0)
        ll_exact = model.log_likelihood(0.0)
        ll_far = model.log_likelihood(10.0)
        assert ll_exact > ll_far

    def test_random_walk_log_likelihood_formula(self) -> None:
        """Test log-likelihood uses Gaussian formula."""
        model = RandomWalkModel()
        model.update(0.0, t=0)
        pred = model.predict(horizon=1)

        y = 1.0
        expected_ll = -0.5 * np.log(2 * np.pi * pred.variance) - 0.5 * (y**2) / pred.variance
        actual_ll = model.log_likelihood(y)

        assert actual_ll == pytest.approx(expected_ll, abs=0.01)

    def test_random_walk_group(self) -> None:
        """Test model group is 'persistence'."""
        model = RandomWalkModel()
        assert model.group == "persistence"

    def test_random_walk_n_parameters(self) -> None:
        """Test model has 1 parameter (variance)."""
        model = RandomWalkModel()
        assert model.n_parameters == 1

    def test_random_walk_name(self) -> None:
        """Test model name."""
        model = RandomWalkModel()
        assert model.name == "RandomWalkModel"

    def test_random_walk_reset(self) -> None:
        """Test reset restores variance toward prior."""
        model = RandomWalkModel(sigma_sq=1.0)
        rng = np.random.default_rng(42)
        for i in range(100):
            model.update(rng.normal(0, 5.0), t=i)

        learned_variance = model.sigma_sq
        assert learned_variance > 5.0

        model.reset(partial=1.0)
        assert model.sigma_sq == model.prior_sigma_sq

    def test_random_walk_variance_estimation(self) -> None:
        """Test variance is estimated from data."""
        model = RandomWalkModel()
        rng = np.random.default_rng(42)
        for i in range(100):
            model.update(rng.normal(0, 2.0), t=i)

        pred = model.predict(horizon=1)
        assert 1.0 < pred.variance < 10.0

    def test_random_walk_returns_prediction_type(self) -> None:
        """Test that predict returns Prediction instance."""
        model = RandomWalkModel()
        model.update(1.0, t=0)
        pred = model.predict(horizon=1)
        assert isinstance(pred, Prediction)


class TestLocalLevelModel:
    """Tests for LocalLevelModel."""

    def test_local_level_smooths(self) -> None:
        """Test that local level smooths toward new data."""
        model = LocalLevelModel(alpha=0.3)

        for t in range(50):
            model.update(0.0, t)
        for t in range(50, 100):
            model.update(10.0, t)

        pred = model.predict(horizon=1)
        assert 9.0 < pred.mean < 10.0

    def test_local_level_converges_to_constant(self) -> None:
        """Test convergence to constant signal."""
        model = LocalLevelModel(alpha=0.1)

        for t in range(200):
            model.update(5.0, t)

        pred = model.predict(horizon=1)
        assert pred.mean == pytest.approx(5.0, abs=0.01)

    def test_local_level_first_observation(self) -> None:
        """Test first observation initializes level."""
        model = LocalLevelModel(alpha=0.1)
        model.update(7.0, t=0)
        pred = model.predict(horizon=1)
        assert pred.mean == 7.0

    def test_local_level_higher_alpha_faster_adaptation(self) -> None:
        """Test that higher alpha adapts faster."""
        model_slow = LocalLevelModel(alpha=0.1)
        model_fast = LocalLevelModel(alpha=0.5)

        for t in range(20):
            model_slow.update(0.0, t)
            model_fast.update(0.0, t)

        model_slow.update(10.0, t=20)
        model_fast.update(10.0, t=20)

        pred_slow = model_slow.predict(horizon=1)
        pred_fast = model_fast.predict(horizon=1)

        assert pred_fast.mean > pred_slow.mean

    def test_local_level_group(self) -> None:
        """Test model group is 'persistence'."""
        model = LocalLevelModel()
        assert model.group == "persistence"

    def test_local_level_n_parameters(self) -> None:
        """Test model has 2 parameters (level, variance)."""
        model = LocalLevelModel()
        assert model.n_parameters == 2

    def test_local_level_name(self) -> None:
        """Test model name."""
        model = LocalLevelModel()
        assert model.name == "LocalLevelModel"

    def test_local_level_default_alpha(self) -> None:
        """Test default alpha value."""
        model = LocalLevelModel()
        assert model.alpha == 0.1

    def test_local_level_custom_alpha(self) -> None:
        """Test custom alpha value."""
        model = LocalLevelModel(alpha=0.5)
        assert model.alpha == 0.5

    def test_local_level_log_likelihood(self) -> None:
        """Test log-likelihood computation."""
        model = LocalLevelModel()
        model.update(5.0, t=0)

        ll_at_level = model.log_likelihood(5.0)
        ll_far = model.log_likelihood(100.0)

        assert ll_at_level > ll_far

    def test_local_level_reset(self) -> None:
        """Test reset restores level toward prior."""
        model = LocalLevelModel()
        for i in range(100):
            model.update(float(i), t=i)

        initial_level = model.level
        model.reset(partial=1.0)
        assert model.level != initial_level

    def test_local_level_cumulative_scales_with_horizon(self) -> None:
        """Test that cumulative prediction scales linearly with horizon."""
        model = LocalLevelModel()
        for t in range(50):
            model.update(5.0 + np.random.normal(0, 0.5), t)

        pred_1 = model.predict(horizon=1)
        pred_10 = model.predict(horizon=10)

        # Cumulative prediction should scale linearly: h=10 is 10x h=1
        assert pred_10.mean == pytest.approx(10 * pred_1.mean, rel=0.01)

    def test_local_level_variance_increases_with_horizon(self) -> None:
        """Test that variance increases with horizon."""
        model = LocalLevelModel()
        for t in range(50):
            model.update(5.0 + np.random.normal(0, 0.5), t)

        pred_1 = model.predict(horizon=1)
        pred_10 = model.predict(horizon=10)

        assert pred_10.variance > pred_1.variance


class TestNumericalStabilityPersistence:
    """Tests for numerical stability with extreme values in persistence models."""

    def test_random_walk_handles_polynomial_growth(self) -> None:
        """RandomWalkModel should handle polynomial growth without NaN."""
        model = RandomWalkModel()

        for i in range(500):
            y = 0.01 * i**2
            model.update(y, t=i)
            pred = model.predict(horizon=1)
            assert np.isfinite(pred.mean), f"Non-finite mean at t={i}"
            assert np.isfinite(pred.variance), f"Non-finite variance at t={i}"

    def test_local_level_handles_polynomial_growth(self) -> None:
        """LocalLevelModel should handle polynomial growth without NaN."""
        model = LocalLevelModel()

        for i in range(500):
            y = 0.01 * i**2
            model.update(y, t=i)
            pred = model.predict(horizon=1)
            assert np.isfinite(pred.mean), f"Non-finite mean at t={i}"
            assert np.isfinite(pred.variance), f"Non-finite variance at t={i}"

    def test_random_walk_handles_large_jumps(self) -> None:
        """RandomWalkModel should handle large jumps without overflow."""
        model = RandomWalkModel()

        for i in range(100):
            y = 1e8 if i % 2 == 0 else -1e8
            model.update(y, t=i)
            pred = model.predict(horizon=1)
            assert np.isfinite(pred.variance), f"Non-finite variance at t={i}"
            assert pred.variance < 1e20, f"Variance too large at t={i}"
