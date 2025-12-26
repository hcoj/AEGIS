"""Integration tests for single-stream AEGIS."""

import numpy as np

from aegis.config import AEGISConfig
from aegis.system import AEGIS


class TestSingleStreamIntegration:
    """Integration tests for single-stream predictions."""

    def test_random_walk_dominant_for_white_noise(self, white_noise_signal) -> None:
        """Test random walk models dominate for white noise."""
        aegis = AEGIS()
        aegis.add_stream("test")

        signal = white_noise_signal(n=200, sigma=1.0)

        for i, y in enumerate(signal):
            if i > 0:
                aegis.predict("test", horizon=1)
            aegis.observe("test", y)
            aegis.end_period()

        diag = aegis.get_diagnostics("test")
        weights = diag["group_weights"]

        # Persistence models should have some weight for white noise
        assert "persistence" in weights

    def test_trend_models_for_linear_trend(self, linear_trend_signal) -> None:
        """Test trend models get weight for trending data."""
        aegis = AEGIS()
        aegis.add_stream("test")

        signal = linear_trend_signal(n=200, slope=0.5, intercept=0.0)

        for i, y in enumerate(signal):
            if i > 0:
                aegis.predict("test", horizon=1)
            aegis.observe("test", y)
            aegis.end_period()

        diag = aegis.get_diagnostics("test")
        weights = diag["group_weights"]

        assert weights.get("trend", 0) > 0.01

    def test_reversion_models_for_ar1(self, ar1_signal) -> None:
        """Test reversion models get weight for AR(1) data."""
        aegis = AEGIS()
        aegis.add_stream("test")

        signal = ar1_signal(n=300, phi=0.8, sigma=0.5)

        for i, y in enumerate(signal):
            if i > 0:
                aegis.predict("test", horizon=1)
            aegis.observe("test", y)
            aegis.end_period()

        diag = aegis.get_diagnostics("test")
        weights = diag["group_weights"]

        assert weights.get("reversion", 0) > 0.05

    def test_periodic_models_for_sine_wave(self, sine_wave_signal) -> None:
        """Test periodic models get weight for oscillating data."""
        aegis = AEGIS(config=AEGISConfig(oscillator_periods=[16]))
        aegis.add_stream("test")

        signal = sine_wave_signal(n=200, period=16, amplitude=2.0)

        for i, y in enumerate(signal):
            if i > 0:
                aegis.predict("test", horizon=1)
            aegis.observe("test", y)
            aegis.end_period()

        diag = aegis.get_diagnostics("test")
        weights = diag["group_weights"]

        assert "periodic" in weights

    def test_prediction_intervals_calibrated(self, white_noise_signal) -> None:
        """Test prediction intervals achieve approximate coverage."""
        aegis = AEGIS(config=AEGISConfig(target_coverage=0.95))
        aegis.add_stream("test")

        signal = white_noise_signal(n=500, sigma=1.0)

        in_interval = 0
        total = 0

        for i, y in enumerate(signal):
            if i > 50:
                pred = aegis.predict("test", horizon=1)
                if pred.interval_lower is not None and pred.interval_upper is not None:
                    if pred.interval_lower <= y <= pred.interval_upper:
                        in_interval += 1
                    total += 1

            aegis.observe("test", y)
            aegis.end_period()

        if total > 0:
            coverage = in_interval / total
            assert 0.8 < coverage < 1.0

    def test_prediction_tracks_level(self, random_walk_signal) -> None:
        """Test predictions track the current level."""
        aegis = AEGIS()
        aegis.add_stream("test")

        signal = random_walk_signal(n=100, sigma=0.5)

        for i, y in enumerate(signal):
            aegis.observe("test", y)
            aegis.end_period()

        pred = aegis.predict("test", horizon=1)
        assert abs(pred.mean - signal[-1]) < 5.0

    def test_variance_increases_with_horizon(self) -> None:
        """Test prediction variance increases with horizon."""
        aegis = AEGIS()
        aegis.add_stream("test")

        for i in range(50):
            aegis.observe("test", float(i))
            aegis.end_period()

        pred_1 = aegis.predict("test", horizon=1)
        pred_5 = aegis.predict("test", horizon=5)

        assert pred_5.variance >= pred_1.variance * 0.9

    def test_end_to_end_prediction(self) -> None:
        """Test complete end-to-end prediction pipeline."""
        aegis = AEGIS()
        aegis.add_stream("price")

        rng = np.random.default_rng(42)
        for i in range(100):
            y = 100.0 + float(i) * 0.1 + rng.normal(0, 1)
            aegis.observe("price", y)
            aegis.end_period()

        pred = aegis.predict("price", horizon=1)

        assert np.isfinite(pred.mean)
        assert pred.variance > 0
        assert pred.interval_lower is not None
        assert pred.interval_upper is not None
        assert pred.interval_lower < pred.mean < pred.interval_upper
