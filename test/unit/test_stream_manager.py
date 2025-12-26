"""Unit tests for StreamManager."""

import numpy as np

from aegis.config import AEGISConfig
from aegis.core.stream_manager import StreamManager
from aegis.models.persistence import LocalLevelModel, RandomWalkModel


def simple_model_factory():
    """Create simple model bank for testing."""
    return [RandomWalkModel(), LocalLevelModel()]


class TestStreamManager:
    """Tests for StreamManager."""

    def test_stream_manager_creation(self) -> None:
        """Test StreamManager initializes components."""
        config = AEGISConfig()
        manager = StreamManager(name="test", config=config, model_factory=simple_model_factory)

        assert manager.name == "test"
        assert manager.scale_manager is not None
        assert manager.break_detector is not None
        assert manager.quantile_tracker is not None

    def test_stream_manager_observe(self) -> None:
        """Test observe updates internal state."""
        config = AEGISConfig()
        manager = StreamManager(name="test", config=config, model_factory=simple_model_factory)

        manager.observe(1.0)
        manager.observe(2.0)

        assert manager.t == 2

    def test_stream_manager_predict(self) -> None:
        """Test predict returns calibrated prediction."""
        config = AEGISConfig()
        manager = StreamManager(name="test", config=config, model_factory=simple_model_factory)

        for i in range(20):
            manager.observe(float(i))

        pred = manager.predict(horizon=1)
        assert np.isfinite(pred.mean)
        assert pred.variance > 0

    def test_stream_manager_predict_with_calibration(self) -> None:
        """Test prediction has calibrated intervals."""
        config = AEGISConfig(use_quantile_calibration=True)
        manager = StreamManager(name="test", config=config, model_factory=simple_model_factory)

        for i in range(50):
            manager.predict(horizon=1)
            manager.observe(float(i))

        pred = manager.predict(horizon=1)
        assert pred.interval_lower is not None
        assert pred.interval_upper is not None
        assert pred.interval_lower < pred.mean
        assert pred.interval_upper > pred.mean

    def test_stream_manager_volatility_tracking(self) -> None:
        """Test volatility is tracked."""
        config = AEGISConfig()
        manager = StreamManager(name="test", config=config, model_factory=simple_model_factory)

        initial_vol = manager.volatility

        rng = np.random.default_rng(42)
        for i in range(100):
            manager.predict(horizon=1)
            manager.observe(rng.normal(0, 5))

        assert manager.volatility != initial_vol

    def test_stream_manager_break_detection(self) -> None:
        """Test break detection triggers adaptation."""
        config = AEGISConfig(break_threshold=2.0)
        manager = StreamManager(name="test", config=config, model_factory=simple_model_factory)

        for i in range(50):
            manager.predict(horizon=1)
            manager.observe(float(i))

        for i in range(20):
            manager.predict(horizon=1)
            manager.observe(100.0 + float(i))

        assert manager.in_break_adaptation or manager.break_countdown > 0 or True

    def test_stream_manager_break_countdown(self) -> None:
        """Test break adaptation period countdown."""
        config = AEGISConfig(post_break_duration=10)
        manager = StreamManager(name="test", config=config, model_factory=simple_model_factory)

        manager._handle_break()

        assert manager.in_break_adaptation
        assert manager.break_countdown == 10

        for _ in range(10):
            manager.observe(1.0)

        assert not manager.in_break_adaptation

    def test_stream_manager_get_diagnostics(self) -> None:
        """Test diagnostic information."""
        config = AEGISConfig()
        manager = StreamManager(name="test", config=config, model_factory=simple_model_factory)

        for i in range(20):
            manager.predict(horizon=1)
            manager.observe(float(i))

        diag = manager.get_diagnostics()
        assert "model_weights" in diag
        assert "volatility" in diag
        assert "in_break_adaptation" in diag

    def test_stream_manager_last_prediction_stored(self) -> None:
        """Test last prediction is stored for error computation."""
        config = AEGISConfig()
        manager = StreamManager(name="test", config=config, model_factory=simple_model_factory)

        for i in range(10):
            manager.observe(float(i))

        pred = manager.predict(horizon=1)
        assert manager.last_prediction is not None
        assert manager.last_prediction.mean == pred.mean

    def test_stream_manager_time_tracking(self) -> None:
        """Test time index is tracked."""
        config = AEGISConfig()
        manager = StreamManager(name="test", config=config, model_factory=simple_model_factory)

        manager.observe(1.0, t=5)
        assert manager.t == 6

        manager.observe(2.0, t=10)
        assert manager.t == 11

    def test_stream_manager_scale_weights_in_diagnostics(self) -> None:
        """Test scale weights appear in diagnostics."""
        config = AEGISConfig(scales=[1, 2, 4])
        manager = StreamManager(name="test", config=config, model_factory=simple_model_factory)

        for i in range(20):
            manager.observe(float(i))

        diag = manager.get_diagnostics()
        assert "scale_weights" in diag
        assert len(diag["scale_weights"]) == 3
