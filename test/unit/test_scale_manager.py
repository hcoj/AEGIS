"""Unit tests for ScaleManager."""

import numpy as np

from aegis.config import AEGISConfig
from aegis.core.scale_manager import ScaleManager
from aegis.models.persistence import LocalLevelModel, RandomWalkModel


def simple_model_factory():
    """Create simple model bank for testing."""
    return [RandomWalkModel(), LocalLevelModel()]


class TestScaleManager:
    """Tests for ScaleManager."""

    def test_scale_manager_creation(self) -> None:
        """Test ScaleManager initializes with per-scale model banks."""
        config = AEGISConfig(scales=[1, 2, 4])
        manager = ScaleManager(config=config, model_factory=simple_model_factory)

        assert len(manager.scales) == 3
        assert 1 in manager.scale_models
        assert 2 in manager.scale_models
        assert 4 in manager.scale_models
        assert len(manager.scale_models[1]) == 2

    def test_scale_manager_observe_updates_history(self) -> None:
        """Test observe adds to history buffer."""
        config = AEGISConfig(scales=[1, 2])
        manager = ScaleManager(config=config, model_factory=simple_model_factory)

        manager.observe(1.0)
        manager.observe(2.0)
        manager.observe(3.0)

        assert len(manager.history) == 3
        assert manager.history[-1] == 3.0

    def test_scale_manager_observe_updates_models(self) -> None:
        """Test observe updates per-scale models."""
        config = AEGISConfig(scales=[1, 2])
        manager = ScaleManager(config=config, model_factory=simple_model_factory)

        for i in range(10):
            manager.observe(float(i))

        model = manager.scale_models[1][0]
        assert model._n_obs > 0

    def test_scale_manager_predict_returns_prediction(self) -> None:
        """Test predict returns combined prediction."""
        config = AEGISConfig(scales=[1, 2])
        manager = ScaleManager(config=config, model_factory=simple_model_factory)

        for i in range(10):
            manager.observe(float(i))

        pred = manager.predict(horizon=1)
        assert np.isfinite(pred.mean)
        assert pred.variance > 0

    def test_scale_manager_predict_converts_to_level(self) -> None:
        """Test prediction is in levels not returns."""
        config = AEGISConfig(scales=[1])
        manager = ScaleManager(config=config, model_factory=simple_model_factory)

        for i in range(20):
            manager.observe(float(i))

        pred = manager.predict(horizon=1)
        assert pred.mean > 10.0

    def test_scale_manager_scale_weights_update(self) -> None:
        """Test scale weights adapt based on accuracy."""
        config = AEGISConfig(scales=[1, 8])
        manager = ScaleManager(config=config, model_factory=simple_model_factory)

        initial_weights = manager.scale_weights.copy()

        for i in range(100):
            manager.observe(float(i))
            manager.update_scale_weights(float(i))

        assert not np.allclose(manager.scale_weights, initial_weights)

    def test_scale_manager_history_trimmed(self) -> None:
        """Test history is trimmed to avoid memory growth."""
        config = AEGISConfig(scales=[1, 2, 4])
        manager = ScaleManager(config=config, model_factory=simple_model_factory)

        for i in range(1000):
            manager.observe(float(i))

        assert len(manager.history) < 100

    def test_scale_manager_predict_at_scale(self) -> None:
        """Test prediction from specific scale."""
        config = AEGISConfig(scales=[1, 4])
        manager = ScaleManager(config=config, model_factory=simple_model_factory)

        for i in range(20):
            manager.observe(float(i))

        pred_1 = manager.predict_at_scale(1, horizon=1)
        pred_4 = manager.predict_at_scale(4, horizon=1)

        assert np.isfinite(pred_1.mean)
        assert np.isfinite(pred_4.mean)

    def test_scale_manager_trigger_break_adaptation(self) -> None:
        """Test break adaptation resets models."""
        config = AEGISConfig(scales=[1, 2])
        manager = ScaleManager(config=config, model_factory=simple_model_factory)

        for i in range(50):
            manager.observe(float(i))

        manager.trigger_break_adaptation()

        weights_after = manager.scale_combiners[1].get_weights()
        assert np.allclose(weights_after, [0.5, 0.5], atol=0.1)

    def test_scale_manager_get_diagnostics(self) -> None:
        """Test diagnostic information."""
        config = AEGISConfig(scales=[1, 2])
        manager = ScaleManager(config=config, model_factory=simple_model_factory)

        for i in range(10):
            manager.observe(float(i))

        diag = manager.get_diagnostics()
        assert "scale_weights" in diag
        assert "per_scale" in diag
        assert 1 in diag["per_scale"]

    def test_scale_manager_multiple_scales_combined(self) -> None:
        """Test prediction combines information from multiple scales."""
        config = AEGISConfig(scales=[1, 2, 4, 8])
        manager = ScaleManager(config=config, model_factory=simple_model_factory)

        rng = np.random.default_rng(42)
        for i in range(100):
            manager.observe(float(i) + rng.normal(0, 0.5))

        pred = manager.predict(horizon=1)
        assert pred.mean > 90
        assert pred.mean < 110

    def test_scale_manager_empty_history_prediction(self) -> None:
        """Test prediction with no history returns default."""
        config = AEGISConfig(scales=[1, 2])
        manager = ScaleManager(config=config, model_factory=simple_model_factory)

        pred = manager.predict(horizon=1)
        assert pred.mean == 0.0
        assert pred.variance == 1.0
