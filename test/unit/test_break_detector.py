"""Unit tests for CUSUMBreakDetector."""

import numpy as np

from aegis.core.break_detector import CUSUMBreakDetector


class TestCUSUMBreakDetector:
    """Tests for CUSUMBreakDetector."""

    def test_detector_no_break_on_stable(self) -> None:
        """Test no break detected on stable errors."""
        detector = CUSUMBreakDetector(threshold=3.0)

        rng = np.random.default_rng(42)
        for _ in range(100):
            error = rng.normal(0, 1)
            assert not detector.update(error)

    def test_detector_detects_mean_shift(self) -> None:
        """Test break detected on mean shift."""
        detector = CUSUMBreakDetector(threshold=3.0)

        rng = np.random.default_rng(42)
        detected = False

        for t in range(100):
            error = rng.normal(0, 1)
            detector.update(error)

        for t in range(100, 150):
            error = rng.normal(5, 1)
            if detector.update(error):
                detected = True
                break

        assert detected

    def test_detector_detects_negative_shift(self) -> None:
        """Test break detected on negative mean shift."""
        detector = CUSUMBreakDetector(threshold=3.0)

        rng = np.random.default_rng(42)
        detected = False

        for t in range(100):
            detector.update(rng.normal(0, 1))

        for t in range(100, 150):
            if detector.update(rng.normal(-5, 1)):
                detected = True
                break

        assert detected

    def test_detector_reset(self) -> None:
        """Test reset clears CUSUM state."""
        detector = CUSUMBreakDetector()

        for _ in range(50):
            detector.update(5.0)

        detector.reset()
        assert detector.cusum_pos == 0.0
        assert detector.cusum_neg == 0.0

    def test_detector_custom_threshold(self) -> None:
        """Test custom threshold value."""
        detector = CUSUMBreakDetector(threshold=5.0)
        assert detector.threshold == 5.0

    def test_detector_custom_drift(self) -> None:
        """Test custom drift value."""
        detector = CUSUMBreakDetector(drift=2.0)
        assert detector.drift == 2.0

    def test_detector_get_state(self) -> None:
        """Test state retrieval."""
        detector = CUSUMBreakDetector()
        detector.update(1.0)

        state = detector.get_state()
        assert "cusum_pos" in state
        assert "cusum_neg" in state
        assert "sigma" in state

    def test_detector_sigma_updates(self) -> None:
        """Test volatility estimate updates."""
        detector = CUSUMBreakDetector()
        initial_sigma = detector.sigma

        rng = np.random.default_rng(42)
        for _ in range(100):
            detector.update(rng.normal(0, 5))

        assert detector.sigma != initial_sigma

    def test_detector_higher_threshold_fewer_breaks(self) -> None:
        """Test higher threshold means fewer false positives."""
        detector_low = CUSUMBreakDetector(threshold=2.0)
        detector_high = CUSUMBreakDetector(threshold=5.0)

        rng = np.random.default_rng(42)
        breaks_low = 0
        breaks_high = 0

        for _ in range(1000):
            error = rng.normal(0, 2)
            if detector_low.update(error):
                breaks_low += 1
                detector_low.reset()
            if detector_high.update(error):
                breaks_high += 1
                detector_high.reset()

        assert breaks_high <= breaks_low
