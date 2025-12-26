"""Unit tests for Prediction dataclass."""

import pytest

from aegis.core.prediction import Prediction


class TestPrediction:
    """Tests for Prediction dataclass."""

    def test_prediction_creation(self) -> None:
        """Test basic Prediction creation."""
        pred = Prediction(mean=5.0, variance=4.0)
        assert pred.mean == 5.0
        assert pred.variance == 4.0

    def test_prediction_std(self) -> None:
        """Test std property returns square root of variance."""
        pred = Prediction(mean=0.0, variance=4.0)
        assert pred.std == 2.0

    def test_prediction_std_zero_variance(self) -> None:
        """Test std with zero variance."""
        pred = Prediction(mean=5.0, variance=0.0)
        assert pred.std == 0.0

    def test_prediction_interval_95(self) -> None:
        """Test 95% confidence interval for standard normal."""
        pred = Prediction(mean=0.0, variance=1.0)
        lower, upper = pred.interval(0.95)
        assert lower == pytest.approx(-1.96, abs=0.01)
        assert upper == pytest.approx(1.96, abs=0.01)

    def test_prediction_interval_90(self) -> None:
        """Test 90% confidence interval."""
        pred = Prediction(mean=0.0, variance=1.0)
        lower, upper = pred.interval(0.90)
        assert lower == pytest.approx(-1.645, abs=0.01)
        assert upper == pytest.approx(1.645, abs=0.01)

    def test_prediction_interval_with_mean_shift(self) -> None:
        """Test interval is centered on mean."""
        pred = Prediction(mean=10.0, variance=4.0)
        lower, upper = pred.interval(0.95)
        midpoint = (lower + upper) / 2
        assert midpoint == pytest.approx(10.0, abs=0.01)

    def test_prediction_interval_scales_with_std(self) -> None:
        """Test interval width scales with standard deviation."""
        pred1 = Prediction(mean=0.0, variance=1.0)
        pred2 = Prediction(mean=0.0, variance=4.0)

        _, upper1 = pred1.interval(0.95)
        _, upper2 = pred2.interval(0.95)

        assert upper2 == pytest.approx(2 * upper1, abs=0.01)

    def test_prediction_calibrated_interval_override(self) -> None:
        """Test that calibrated interval bounds override Gaussian."""
        pred = Prediction(
            mean=0.0,
            variance=1.0,
            interval_lower=-3.0,
            interval_upper=3.0,
        )
        lower, upper = pred.interval(0.95)
        assert lower == -3.0
        assert upper == 3.0

    def test_prediction_calibrated_interval_partial(self) -> None:
        """Test partial calibration (only one bound set)."""
        pred = Prediction(mean=0.0, variance=1.0, interval_lower=-2.5)
        lower, upper = pred.interval(0.95)
        assert lower == -2.5
        assert upper == pytest.approx(1.96, abs=0.01)

    def test_prediction_interval_symmetric(self) -> None:
        """Test interval is symmetric around mean."""
        pred = Prediction(mean=5.0, variance=9.0)
        lower, upper = pred.interval(0.95)
        assert (upper - 5.0) == pytest.approx(5.0 - lower, abs=0.01)

    def test_prediction_immutable_fields(self) -> None:
        """Test that Prediction fields are set correctly."""
        pred = Prediction(mean=1.0, variance=2.0)
        assert pred.mean == 1.0
        assert pred.variance == 2.0
