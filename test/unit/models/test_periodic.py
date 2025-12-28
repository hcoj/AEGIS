"""Unit tests for periodic models (OscillatorBank, SeasonalDummy)."""

import numpy as np
import pytest

from aegis.core.prediction import Prediction
from aegis.models.periodic import OscillatorBankModel, SeasonalDummyModel


class TestOscillatorBankModel:
    """Tests for OscillatorBankModel."""

    def test_oscillator_captures_sine(self, sine_wave_signal) -> None:
        """Test model captures sinusoidal signal."""
        signal = sine_wave_signal(n=500, period=16, amplitude=1.0)
        model = OscillatorBankModel(periods=[16], lr=0.05)

        for t, y in enumerate(signal):
            model.update(y, t)

        pred = model.predict(horizon=1)
        expected = np.sin(2 * np.pi * 501 / 16)
        assert pred.mean == pytest.approx(expected, abs=0.3)

    def test_oscillator_multiple_frequencies(self, sine_wave_signal) -> None:
        """Test model with multiple frequencies."""
        signal = sine_wave_signal(n=300, period=8) + sine_wave_signal(n=300, period=32) * 0.5
        model = OscillatorBankModel(periods=[8, 16, 32])

        for t, y in enumerate(signal):
            model.update(y, t)

        pred = model.predict(horizon=1)
        assert abs(pred.mean) < 3.0

    def test_oscillator_default_periods(self) -> None:
        """Test default period bank."""
        model = OscillatorBankModel()
        assert len(model.periods) > 0
        assert 16 in model.periods

    def test_oscillator_custom_periods(self) -> None:
        """Test custom periods."""
        model = OscillatorBankModel(periods=[7, 14, 28])
        assert model.periods == [7, 14, 28]

    def test_oscillator_group(self) -> None:
        """Test model group is 'periodic'."""
        model = OscillatorBankModel()
        assert model.group == "periodic"

    def test_oscillator_n_parameters(self) -> None:
        """Test parameter count (2 per freq + variance)."""
        model = OscillatorBankModel(periods=[8, 16])
        assert model.n_parameters == 2 * 2 + 1

    def test_oscillator_name(self) -> None:
        """Test model name includes periods."""
        model = OscillatorBankModel(periods=[10])
        assert model.name == "OscillatorBankModel_p10"

    def test_oscillator_log_likelihood(self, sine_wave_signal) -> None:
        """Test log-likelihood computation."""
        signal = sine_wave_signal(n=100, period=16)
        model = OscillatorBankModel(periods=[16])

        for t, y in enumerate(signal):
            model.update(y, t)

        pred = model.predict(horizon=1)
        ll_at_pred = model.log_likelihood(pred.mean)
        ll_far = model.log_likelihood(pred.mean + 10)

        assert ll_at_pred > ll_far

    def test_oscillator_reset(self, sine_wave_signal) -> None:
        """Test reset clears learned coefficients."""
        signal = sine_wave_signal(n=100, period=16, amplitude=5.0)
        model = OscillatorBankModel(periods=[16])

        for t, y in enumerate(signal):
            model.update(y, t)

        initial_amplitude = np.sqrt(model.a[0] ** 2 + model.b[0] ** 2)
        model.reset(partial=1.0)
        reset_amplitude = np.sqrt(model.a[0] ** 2 + model.b[0] ** 2)

        assert reset_amplitude < initial_amplitude

    def test_oscillator_prediction_horizon(self, sine_wave_signal) -> None:
        """Test prediction at different horizons."""
        signal = sine_wave_signal(n=200, period=16)
        model = OscillatorBankModel(periods=[16])

        for t, y in enumerate(signal):
            model.update(y, t)

        pred_1 = model.predict(horizon=1)
        pred_8 = model.predict(horizon=8)

        assert pred_1.mean != pred_8.mean

    def test_oscillator_returns_prediction_type(self) -> None:
        """Test that predict returns Prediction instance."""
        model = OscillatorBankModel()
        model.update(1.0, t=0)
        pred = model.predict(horizon=1)
        assert isinstance(pred, Prediction)

    def test_oscillator_variance_grows_with_horizon(self, sine_wave_signal) -> None:
        """Test that variance grows with horizon due to phase uncertainty.

        Periodic models should have higher variance at longer horizons
        because phase uncertainty accumulates over time.
        """
        signal = sine_wave_signal(n=200, period=16)
        model = OscillatorBankModel(periods=[16])

        for t, y in enumerate(signal):
            model.update(y, t)

        pred_1 = model.predict(horizon=1)
        pred_64 = model.predict(horizon=64)

        # Variance should grow with horizon
        assert pred_64.variance > pred_1.variance, (
            f"Variance should grow with horizon: h=1 var={pred_1.variance:.4f}, "
            f"h=64 var={pred_64.variance:.4f}"
        )


class TestSeasonalDummyModel:
    """Tests for SeasonalDummyModel."""

    def test_seasonal_dummy_learns_pattern(self, seasonal_signal) -> None:
        """Test model learns seasonal pattern."""
        pattern = [10, 12, 15, 14, 13, 8, 5]
        signal = seasonal_signal(n=350, period=7, pattern=pattern, noise_sigma=0.5)
        model = SeasonalDummyModel(period=7)

        for t, y in enumerate(signal):
            model.update(y, t)

        for i, expected in enumerate(pattern):
            assert model.means[i] == pytest.approx(expected, abs=2.0)

    def test_seasonal_dummy_predicts_correctly(self, seasonal_signal) -> None:
        """Test cumulative prediction matches learned pattern sum."""
        pattern = [1, 2, 3, 4]
        signal = seasonal_signal(n=200, period=4, pattern=pattern, noise_sigma=0.1)
        model = SeasonalDummyModel(period=4)

        for t, y in enumerate(signal):
            model.update(y, t)

        # Cumulative prediction for h steps sums over the next h pattern values
        pred_4 = model.predict(horizon=4)
        expected_cycle_sum = sum(pattern)  # 1+2+3+4 = 10
        assert pred_4.mean == pytest.approx(expected_cycle_sum, abs=2.0)

    def test_seasonal_dummy_group(self) -> None:
        """Test model group is 'periodic'."""
        model = SeasonalDummyModel(period=7)
        assert model.group == "periodic"

    def test_seasonal_dummy_n_parameters(self) -> None:
        """Test parameter count (period means + variance)."""
        model = SeasonalDummyModel(period=7)
        assert model.n_parameters == 7 + 1

    def test_seasonal_dummy_name(self) -> None:
        """Test model name includes period."""
        model = SeasonalDummyModel(period=7)
        assert model.name == "SeasonalDummyModel_p7"

    def test_seasonal_dummy_custom_period(self) -> None:
        """Test custom period value."""
        model = SeasonalDummyModel(period=12)
        assert model.period == 12
        assert len(model.means) == 12

    def test_seasonal_dummy_log_likelihood(self, seasonal_signal) -> None:
        """Test log-likelihood computation."""
        pattern = [1, 2, 3, 4]
        signal = seasonal_signal(n=100, period=4, pattern=pattern)
        model = SeasonalDummyModel(period=4)

        for t, y in enumerate(signal):
            model.update(y, t)

        pred = model.predict(horizon=1)
        ll_at_pred = model.log_likelihood(pred.mean)
        ll_far = model.log_likelihood(pred.mean + 100)

        assert ll_at_pred > ll_far

    def test_seasonal_dummy_reset(self, seasonal_signal) -> None:
        """Test reset restores toward priors."""
        pattern = [10, 20, 30, 40]
        signal = seasonal_signal(n=100, period=4, pattern=pattern)
        model = SeasonalDummyModel(period=4)

        for t, y in enumerate(signal):
            model.update(y, t)

        max_mean = np.max(model.means)
        model.reset(partial=1.0)
        reset_max = np.max(model.means)

        assert reset_max < max_mean

    def test_seasonal_dummy_returns_prediction_type(self) -> None:
        """Test that predict returns Prediction instance."""
        model = SeasonalDummyModel(period=7)
        model.update(1.0, t=0)
        pred = model.predict(horizon=1)
        assert isinstance(pred, Prediction)

    def test_seasonal_dummy_period_wrapping(self) -> None:
        """Test cumulative prediction sums correctly over multiple periods."""
        model = SeasonalDummyModel(period=4)
        model.means = np.array([1.0, 2.0, 3.0, 4.0])
        model.t = 10

        # t=10, so next positions are 11,12,13,14,... which mod 4 = 3,0,1,2,...
        pred_4 = model.predict(horizon=4)
        pred_8 = model.predict(horizon=8)

        # Full period sum = 1+2+3+4 = 10
        cycle_sum = sum(model.means)
        assert pred_4.mean == pytest.approx(cycle_sum, abs=0.01)
        assert pred_8.mean == pytest.approx(2 * cycle_sum, abs=0.01)

    def test_seasonal_dummy_variance_grows_with_horizon(self, seasonal_signal) -> None:
        """Test that variance grows with horizon due to phase uncertainty."""
        pattern = [1, 2, 3, 4]
        signal = seasonal_signal(n=200, period=4, pattern=pattern)
        model = SeasonalDummyModel(period=4)

        for t, y in enumerate(signal):
            model.update(y, t)

        pred_1 = model.predict(horizon=1)
        pred_64 = model.predict(horizon=64)

        # Variance should grow with horizon
        assert pred_64.variance > pred_1.variance


class TestOscillatorPhaseTracking:
    """Tests for improved phase tracking in OscillatorBankModel."""

    def test_oscillator_tracks_phase_stability(self, sine_wave_signal) -> None:
        """Test that phase stability is tracked."""
        signal = sine_wave_signal(n=300, period=16, amplitude=1.0)
        model = OscillatorBankModel(periods=[16], lr=0.05)

        for t, y in enumerate(signal):
            model.update(y, t)

        # Phase stability should be available
        assert hasattr(model, "phase_stability")
        # After learning a clean sine, stability should be high
        assert model.phase_stability[0] > 0.5

    def test_oscillator_stable_phase_reduces_uncertainty(self, sine_wave_signal) -> None:
        """Test that stable phase reduces long-horizon uncertainty.

        When the oscillator has locked onto a stable phase, the
        variance at long horizons should be relatively lower than
        when phase is uncertain.
        """
        signal = sine_wave_signal(n=500, period=16, amplitude=2.0)
        model = OscillatorBankModel(periods=[16], lr=0.05)

        for t, y in enumerate(signal):
            model.update(y, t)

        pred_64 = model.predict(horizon=64)

        # With stable phase, the variance should be reasonable
        # Not dominated by linear phase uncertainty growth
        # Variance ratio h=64 vs h=1 should be < 2x if phase is stable
        pred_1 = model.predict(horizon=1)
        variance_ratio = pred_64.variance / pred_1.variance
        assert variance_ratio < 3.0, (
            f"Stable phase should limit variance growth: ratio={variance_ratio:.2f}"
        )

    def test_oscillator_amplitude_computed(self, sine_wave_signal) -> None:
        """Test that amplitude is correctly computed from coefficients."""
        signal = sine_wave_signal(n=300, period=16, amplitude=2.0)
        model = OscillatorBankModel(periods=[16], lr=0.05)

        for t, y in enumerate(signal):
            model.update(y, t)

        # Amplitude should match input amplitude
        amplitude = np.sqrt(model.a[0] ** 2 + model.b[0] ** 2)
        assert amplitude == pytest.approx(2.0, abs=0.5)
