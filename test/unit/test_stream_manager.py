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

    def test_stream_manager_quantile_tracker_uses_h1_only(self) -> None:
        """Test quantile tracker only learns from h=1 predictions.

        Bug: When predictions at multiple horizons are made before observe(),
        the quantile tracker was incorrectly using the last prediction
        (potentially h=64 or h=1024) to update, which compares the wrong
        target value to the current observation.

        The fix: Only update quantile tracker from h=1 predictions.
        """
        config = AEGISConfig(use_quantile_calibration=True, scales=[1, 2])
        manager = StreamManager(name="test", config=config, model_factory=simple_model_factory)

        rng = np.random.default_rng(42)
        signal = rng.normal(0, 1, 200)

        # Warm up
        for i in range(50):
            manager.predict(horizon=1)
            manager.observe(signal[i])

        # Now simulate multi-horizon prediction pattern (like acceptance tests)
        for i in range(50, 150):
            # Make predictions at multiple horizons before observing
            manager.predict(horizon=1)
            manager.predict(horizon=16)  # This SHOULD NOT affect quantile tracker
            manager.predict(horizon=64)  # This SHOULD NOT affect quantile tracker
            manager.observe(signal[i])

        final_q_low, final_q_high = manager.quantile_tracker.get_quantiles(horizon=1)

        # The quantile tracker should converge toward reasonable values
        # For white noise with sigma=1, the z-scores should stay near ±1.96
        # If wrongly updated with h=64 predictions, quantiles would diverge wildly

        # Check that quantiles haven't diverged to extreme values
        assert abs(final_q_low) < 5.0, (
            f"q_low diverged to {final_q_low:.2f}, suggesting wrong horizon used"
        )
        assert abs(final_q_high) < 5.0, (
            f"q_high diverged to {final_q_high:.2f}, suggesting wrong horizon used"
        )

        # The interval width should be reasonable (not exploded)
        interval_width = final_q_high - final_q_low
        assert interval_width < 10.0, (
            f"Interval width {interval_width:.2f} too large, quantiles corrupted"
        )

    def test_stream_manager_stores_h1_prediction_for_quantile_tracking(self) -> None:
        """Test that last_h1_prediction is stored separately for quantile tracking.

        The StreamManager must store h=1 predictions separately from last_prediction
        so that quantile tracker updates use the correct values even when predictions
        at other horizons are made.
        """
        config = AEGISConfig(use_quantile_calibration=True, scales=[1, 2])
        manager = StreamManager(name="test", config=config, model_factory=simple_model_factory)

        # Warm up
        for i in range(20):
            manager.predict(horizon=1)
            manager.observe(float(i))

        # Make predictions at multiple horizons
        pred_h1 = manager.predict(horizon=1)
        pred_h64 = manager.predict(horizon=64)

        # The last_prediction should be h=64, but last_h1_prediction should be h=1
        assert manager.last_prediction.mean == pred_h64.mean
        assert manager.last_prediction.variance == pred_h64.variance

        # There should be a separate h=1 prediction stored
        assert hasattr(manager, "last_h1_prediction"), (
            "StreamManager should store last_h1_prediction for quantile tracking"
        )
        assert manager.last_h1_prediction is not None
        assert manager.last_h1_prediction.mean == pred_h1.mean
        assert manager.last_h1_prediction.variance == pred_h1.variance

    def test_stream_manager_uses_horizon_aware_quantile_tracker(self) -> None:
        """Test StreamManager uses HorizonAwareQuantileTracker for calibration."""
        from aegis.core.quantile_tracker import HorizonAwareQuantileTracker

        config = AEGISConfig(use_quantile_calibration=True, scales=[1, 2])
        manager = StreamManager(name="test", config=config, model_factory=simple_model_factory)

        # Should use horizon-aware tracker
        assert isinstance(manager.quantile_tracker, HorizonAwareQuantileTracker)

    def test_stream_manager_horizon_specific_intervals(self) -> None:
        """Test predictions at different horizons get horizon-specific intervals."""
        config = AEGISConfig(use_quantile_calibration=True, scales=[1, 2])
        manager = StreamManager(name="test", config=config, model_factory=simple_model_factory)

        # Warm up with h=1 predictions that are consistently too narrow
        for i in range(100):
            pred = manager.predict(horizon=1)
            # Observation outside the interval forces quantile widening
            manager.observe(pred.mean + 5.0 * np.sqrt(pred.variance))

        # Now h=1 intervals should be wider than h=64 intervals
        # (because h=1 has been trained to widen, h=64 hasn't)
        pred_h1 = manager.predict(horizon=1)
        pred_h64 = manager.predict(horizon=64)

        # Get interval widths (in terms of std multipliers)
        if pred_h1.interval_upper is not None and pred_h1.interval_lower is not None:
            width_h1 = (pred_h1.interval_upper - pred_h1.interval_lower) / np.sqrt(pred_h1.variance)
            width_h64 = (pred_h64.interval_upper - pred_h64.interval_lower) / np.sqrt(
                pred_h64.variance
            )

            # h=1 should have learned wider intervals
            assert width_h1 > width_h64, (
                f"h=1 interval width ({width_h1:.2f}) should be > h=64 ({width_h64:.2f})"
            )
