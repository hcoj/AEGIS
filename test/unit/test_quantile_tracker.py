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
