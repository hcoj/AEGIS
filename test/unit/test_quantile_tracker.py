"""Unit tests for QuantileTracker."""

import numpy as np
import pytest

from aegis.core.prediction import Prediction
from aegis.core.quantile_tracker import QuantileTracker


class TestQuantileTracker:
    """Tests for QuantileTracker."""

    def test_tracker_initial_quantiles_gaussian(self) -> None:
        """Test initial quantiles are Gaussian."""
        tracker = QuantileTracker(target_coverage=0.95)

        assert tracker.q_low == pytest.approx(-1.96, abs=0.01)
        assert tracker.q_high == pytest.approx(1.96, abs=0.01)

    def test_tracker_calibrates_coverage(self) -> None:
        """Test tracker calibrates toward target coverage."""
        tracker = QuantileTracker(target_coverage=0.95, learning_rate=0.1)

        rng = np.random.default_rng(42)
        in_interval = 0
        total = 0

        for _ in range(500):
            pred_std = 1.0
            pred_mean = 0.0

            interval = tracker.get_interval(pred_mean, pred_std)
            y = rng.normal(0, 1)

            if interval[0] <= y <= interval[1]:
                in_interval += 1
            total += 1

            tracker.update(y, pred_mean, pred_std)

        coverage = in_interval / total
        assert 0.85 < coverage < 1.0

    def test_tracker_adjusts_for_heavy_tails(self, heavy_tailed_signal) -> None:
        """Test tracker widens intervals for heavy-tailed distributions."""
        tracker = QuantileTracker(target_coverage=0.95, learning_rate=0.05)

        signal = heavy_tailed_signal(n=500, df=3)

        for y in signal:
            tracker.update(y, pred_mean=0.0, pred_std=1.0)

        assert abs(tracker.q_low) > 2.0
        assert abs(tracker.q_high) > 2.0

    def test_tracker_get_interval(self) -> None:
        """Test interval computation."""
        tracker = QuantileTracker()

        interval = tracker.get_interval(pred_mean=5.0, pred_std=2.0)

        assert interval[0] < 5.0
        assert interval[1] > 5.0

    def test_tracker_calibrate_prediction(self) -> None:
        """Test calibrating a Prediction object."""
        tracker = QuantileTracker()

        pred = Prediction(mean=10.0, variance=4.0)
        calibrated = tracker.calibrate_prediction(pred)

        assert calibrated.interval_lower is not None
        assert calibrated.interval_upper is not None
        assert calibrated.interval_lower < calibrated.mean
        assert calibrated.interval_upper > calibrated.mean

    def test_tracker_custom_coverage(self) -> None:
        """Test custom target coverage."""
        tracker = QuantileTracker(target_coverage=0.90)
        assert tracker.target_coverage == 0.90

        interval = tracker.get_interval(pred_mean=0.0, pred_std=1.0)
        width_90 = interval[1] - interval[0]

        tracker_95 = QuantileTracker(target_coverage=0.95)
        interval_95 = tracker_95.get_interval(pred_mean=0.0, pred_std=1.0)
        width_95 = interval_95[1] - interval_95[0]

        assert width_95 > width_90

    def test_tracker_reset(self) -> None:
        """Test reset restores Gaussian quantiles."""
        tracker = QuantileTracker()

        for _ in range(100):
            tracker.update(10.0, pred_mean=0.0, pred_std=1.0)

        tracker.reset()
        assert tracker.q_low == pytest.approx(-1.96, abs=0.01)
        assert tracker.q_high == pytest.approx(1.96, abs=0.01)

    def test_tracker_get_coverage_stats(self) -> None:
        """Test coverage statistics."""
        tracker = QuantileTracker()

        rng = np.random.default_rng(42)
        for _ in range(100):
            y = rng.normal(0, 1)
            tracker.update(y, pred_mean=0.0, pred_std=1.0)

        stats = tracker.get_coverage_stats()
        assert "empirical_coverage" in stats
        assert "target_coverage" in stats


class TestHorizonAwareQuantileTracker:
    """Tests for horizon-aware quantile tracking with continuous interpolation."""

    def test_horizon_aware_tracker_exists(self) -> None:
        """Test HorizonAwareQuantileTracker class exists."""
        from aegis.core.quantile_tracker import HorizonAwareQuantileTracker

        tracker = HorizonAwareQuantileTracker(target_coverage=0.95)
        assert tracker is not None

    def test_horizon_aware_tracker_initial_quantiles(self) -> None:
        """Test tracker starts with Gaussian quantiles at all horizons."""
        from aegis.core.quantile_tracker import HorizonAwareQuantileTracker

        tracker = HorizonAwareQuantileTracker(target_coverage=0.95)

        # All horizons should start the same
        for h in [1, 16, 64, 256, 1024]:
            q_low, q_high = tracker.get_quantiles(horizon=h)
            assert q_low == pytest.approx(-1.96, abs=0.01)
            assert q_high == pytest.approx(1.96, abs=0.01)

    def test_horizon_aware_tracker_updates_nearby_anchors(self) -> None:
        """Test updates affect nearby anchors with interpolated weights."""
        from aegis.core.quantile_tracker import HorizonAwareQuantileTracker

        tracker = HorizonAwareQuantileTracker(target_coverage=0.95, learning_rate=0.1)

        # Update at h=1 with consistently high z-scores
        for _ in range(50):
            tracker.update(y=5.0, pred_mean=0.0, pred_std=1.0, horizon=1)

        # h=1 quantiles should have widened significantly
        q1_low, q1_high = tracker.get_quantiles(horizon=1)
        assert q1_high > 2.5

        # h=1024 (far anchor) should still be near initial
        q1024_low, q1024_high = tracker.get_quantiles(horizon=1024)
        assert q1024_high == pytest.approx(1.96, abs=0.1)

    def test_horizon_aware_tracker_continuous_interpolation(self) -> None:
        """Test quantiles change smoothly across horizons - NO discontinuities."""
        from aegis.core.quantile_tracker import HorizonAwareQuantileTracker

        tracker = HorizonAwareQuantileTracker(target_coverage=0.95, learning_rate=0.1)

        # Train h=1 to have wider quantiles
        for _ in range(100):
            tracker.update(y=5.0, pred_mean=0.0, pred_std=1.0, horizon=1)

        # Check continuity: quantiles should change monotonically toward untrained anchors
        q_values = []
        horizons = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        for h in horizons:
            _, q_high = tracker.get_quantiles(horizon=h)
            q_values.append(q_high)

        # h=1 should have highest quantile (trained), h=1024 lowest (untrained)
        assert q_values[0] > q_values[-1], "h=1 should have wider quantile than h=1024"

        # Should be monotonically decreasing (no sudden jumps UP)
        for i in range(1, len(q_values)):
            assert q_values[i] <= q_values[i - 1] + 0.01, (
                f"Non-monotonic at h={horizons[i]}: "
                f"q[{horizons[i - 1]}]={q_values[i - 1]:.3f}, q[{horizons[i]}]={q_values[i]:.3f}"
            )

    def test_horizon_aware_tracker_interpolation_between_anchors(self) -> None:
        """Test that horizons between anchors get interpolated values."""
        from aegis.core.quantile_tracker import HorizonAwareQuantileTracker

        tracker = HorizonAwareQuantileTracker(target_coverage=0.95, learning_rate=0.1)

        # Set anchor h=1 to have high quantile, h=16 stays default
        for _ in range(100):
            tracker.update(y=5.0, pred_mean=0.0, pred_std=1.0, horizon=1)

        q1_low, q1_high = tracker.get_quantiles(horizon=1)
        q16_low, q16_high = tracker.get_quantiles(horizon=16)

        # h=4 should be interpolated between h=1 and h=16
        q4_low, q4_high = tracker.get_quantiles(horizon=4)

        # q4_high should be between q1_high and q16_high
        assert min(q1_high, q16_high) <= q4_high <= max(q1_high, q16_high)

    def test_horizon_aware_tracker_get_interval(self) -> None:
        """Test interval uses horizon-specific quantiles."""
        from aegis.core.quantile_tracker import HorizonAwareQuantileTracker

        tracker = HorizonAwareQuantileTracker(target_coverage=0.95)

        interval_h1 = tracker.get_interval(pred_mean=0.0, pred_std=1.0, horizon=1)
        interval_h64 = tracker.get_interval(pred_mean=0.0, pred_std=1.0, horizon=64)

        # Initially both should be the same
        assert interval_h1[0] == pytest.approx(interval_h64[0], abs=0.01)

    def test_horizon_aware_tracker_calibrate_prediction(self) -> None:
        """Test calibrating a prediction with horizon."""
        from aegis.core.quantile_tracker import HorizonAwareQuantileTracker

        tracker = HorizonAwareQuantileTracker()

        pred = Prediction(mean=10.0, variance=4.0)
        calibrated = tracker.calibrate_prediction(pred, horizon=16)

        assert calibrated.interval_lower is not None
        assert calibrated.interval_upper is not None
