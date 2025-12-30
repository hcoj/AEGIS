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

    def test_oscillator_variance_constant_across_horizons(self, sine_wave_signal) -> None:
        """Test that variance is constant for well-learned periodic signals.

        For a learned periodic signal, we know the pattern, so uncertainty
        doesn't grow with horizon.
        """
        signal = sine_wave_signal(n=500, period=16)
        model = OscillatorBankModel(periods=[16], lr=0.05)

        for t, y in enumerate(signal):
            model.update(y, t)

        pred_1 = model.predict(horizon=1)
        pred_64 = model.predict(horizon=64)
        pred_1024 = model.predict(horizon=1024)

        # Variance should be constant (same sigma_sq)
        assert np.isclose(pred_1.variance, pred_64.variance, rtol=0.01), (
            f"Variance should be constant: h=1 var={pred_1.variance:.4e}, "
            f"h=64 var={pred_64.variance:.4e}"
        )
        assert np.isclose(pred_64.variance, pred_1024.variance, rtol=0.01)

    def test_oscillator_mean_accurate_at_long_horizons(self, sine_wave_signal) -> None:
        """Mean prediction should be accurate for learned periodic signals."""
        signal = sine_wave_signal(n=500, period=16)
        model = OscillatorBankModel(periods=[16], lr=0.05)

        for t, y in enumerate(signal):
            model.update(y, t)

        # Predict point value at various horizons - should match sine wave pattern
        # At h=16 (one full period), we expect ~sin(2*pi*(t+16)/16) ≈ sin(2*pi*t/16)
        # which should be close to the current point in the cycle
        pred_h1 = model.predict(horizon=1)
        pred_h16 = model.predict(horizon=16)

        # Predictions at h=1 and h=16 should be similar (one period apart)
        assert abs(pred_h1.mean - pred_h16.mean) < 0.5, (
            f"h=1 and h=16 should be similar for period=16: "
            f"h=1={pred_h1.mean:.3f}, h=16={pred_h16.mean:.3f}"
        )

    def test_oscillator_variance_has_reasonable_minimum(self, sine_wave_signal) -> None:
        """OscillatorBank maintains minimum variance even for perfect sinusoids.

        When fitting a clean sine wave, sigma_sq decays toward zero.
        But we need a reasonable variance floor to maintain calibrated intervals.
        """
        # Clean sine wave (no noise by default)
        signal = sine_wave_signal(n=500, period=16, amplitude=1.0)
        model = OscillatorBankModel(periods=[16], lr=0.05)

        for t, y in enumerate(signal):
            model.update(y, t)

        pred = model.predict(horizon=64)
        # Variance should be at least 1e-4, not 1e-10
        assert pred.variance >= 1e-4, (
            f"Variance collapsed to {pred.variance:.2e}, need minimum 1e-4 for calibration"
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
        """Test point prediction matches learned pattern value at correct phase."""
        pattern = [1, 2, 3, 4]
        signal = seasonal_signal(n=200, period=4, pattern=pattern, noise_sigma=0.1)
        model = SeasonalDummyModel(period=4)

        for t, y in enumerate(signal):
            model.update(y, t)

        # Point prediction at horizon h gives the seasonal mean for phase at t+h
        # After 200 observations, t=199 (phase 3 since 199 % 4 = 3)
        # h=1 -> phase (199+1) % 4 = 0 -> pattern[0] ≈ 1
        # h=4 -> phase (199+4) % 4 = 3 -> pattern[3] ≈ 4
        pred_1 = model.predict(horizon=1)
        pred_4 = model.predict(horizon=4)
        assert pred_1.mean == pytest.approx(pattern[0], abs=0.5)
        assert pred_4.mean == pytest.approx(pattern[3], abs=0.5)

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
        """Test point prediction correctly wraps around period."""
        model = SeasonalDummyModel(period=4)
        model.means = np.array([1.0, 2.0, 3.0, 4.0])
        model.t = 10

        # t=10, so phase at t+h is (10+h) % 4
        # h=1: (10+1) % 4 = 3 -> means[3] = 4.0
        # h=4: (10+4) % 4 = 2 -> means[2] = 3.0
        # h=8: (10+8) % 4 = 2 -> means[2] = 3.0
        pred_1 = model.predict(horizon=1)
        pred_4 = model.predict(horizon=4)
        pred_8 = model.predict(horizon=8)

        assert pred_1.mean == pytest.approx(4.0, abs=0.01)
        assert pred_4.mean == pytest.approx(3.0, abs=0.01)
        assert pred_8.mean == pytest.approx(3.0, abs=0.01)  # Same phase as h=4

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


class TestOscillatorAmplitude:
    """Tests for amplitude learning in OscillatorBankModel."""

    def test_oscillator_amplitude_computed(self, sine_wave_signal) -> None:
        """Test that amplitude is correctly computed from coefficients."""
        signal = sine_wave_signal(n=300, period=16, amplitude=2.0)
        model = OscillatorBankModel(periods=[16], lr=0.05)

        for t, y in enumerate(signal):
            model.update(y, t)

        # Amplitude should match input amplitude
        amplitude = np.sqrt(model.a[0] ** 2 + model.b[0] ** 2)
        assert amplitude == pytest.approx(2.0, abs=0.5)

    def test_long_horizon_error_bounded(self) -> None:
        """Long-horizon error growth should be bounded."""
        n, period = 1000, 16
        signal = [np.sin(2 * np.pi * t / period) for t in range(n)]
        model = OscillatorBankModel(periods=[period], lr=0.05)

        for t in range(500):
            model.update(signal[t], t)

        err_h1 = abs(model.predict(1).mean - signal[501])
        err_h1024 = abs(model.predict(1024).mean - sum(signal[501 : 501 + min(1024, n - 501)]))

        growth = (err_h1024 + 0.01) / (err_h1 + 0.01)
        assert growth < 50.0, f"Error growth {growth:.1f}x exceeds 50x"
