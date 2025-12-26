"""Validation tests for AEGIS signal taxonomy.

Tests that appropriate models get weight for each signal type.
"""

import numpy as np

from aegis.config import AEGISConfig
from aegis.system import AEGIS


class TestSignalTaxonomy:
    """Validation tests for signal taxonomy."""

    def test_constant_signal(self, constant_signal) -> None:
        """Test constant signal: persistence models should dominate."""
        aegis = AEGIS()
        aegis.add_stream("test")

        signal = constant_signal(n=100, c=5.0)

        for y in signal:
            aegis.observe("test", y)
            aegis.end_period()

        pred = aegis.predict("test", horizon=1)
        assert abs(pred.mean - 5.0) < 1.0

    def test_white_noise_signal(self, white_noise_signal) -> None:
        """Test white noise: random walk should be competitive."""
        aegis = AEGIS()
        aegis.add_stream("test")

        signal = white_noise_signal(n=200, sigma=1.0)

        for y in signal:
            aegis.observe("test", y)
            aegis.end_period()

        diag = aegis.get_diagnostics("test")
        assert "persistence" in diag["group_weights"]

    def test_random_walk_signal(self, random_walk_signal) -> None:
        """Test random walk: persistence models should dominate."""
        aegis = AEGIS()
        aegis.add_stream("test")

        signal = random_walk_signal(n=200, sigma=0.5)

        for y in signal:
            aegis.observe("test", y)
            aegis.end_period()

        diag = aegis.get_diagnostics("test")
        assert diag["group_weights"].get("persistence", 0) > 0.05

    def test_ar1_signal(self, ar1_signal) -> None:
        """Test AR(1) signal: reversion models should get weight."""
        aegis = AEGIS()
        aegis.add_stream("test")

        signal = ar1_signal(n=300, phi=0.8, sigma=0.5)

        for y in signal:
            aegis.observe("test", y)
            aegis.end_period()

        diag = aegis.get_diagnostics("test")
        assert diag["group_weights"].get("reversion", 0) > 0.01

    def test_linear_trend_signal(self, linear_trend_signal) -> None:
        """Test linear trend: trend models should dominate."""
        aegis = AEGIS()
        aegis.add_stream("test")

        signal = linear_trend_signal(n=200, slope=0.5, intercept=0.0)

        for y in signal:
            aegis.observe("test", y)
            aegis.end_period()

        diag = aegis.get_diagnostics("test")
        assert "trend" in diag["group_weights"]

    def test_sine_wave_signal(self, sine_wave_signal) -> None:
        """Test sine wave: periodic models should get weight."""
        config = AEGISConfig(oscillator_periods=[16])
        aegis = AEGIS(config=config)
        aegis.add_stream("test")

        signal = sine_wave_signal(n=200, period=16, amplitude=2.0)

        for y in signal:
            aegis.observe("test", y)
            aegis.end_period()

        diag = aegis.get_diagnostics("test")
        assert "periodic" in diag["group_weights"]

    def test_seasonal_signal(self, seasonal_signal) -> None:
        """Test seasonal signal: seasonal models should get weight."""
        config = AEGISConfig(seasonal_periods=[7])
        aegis = AEGIS(config=config)
        aegis.add_stream("test")

        signal = seasonal_signal(n=200, period=7)

        for y in signal:
            aegis.observe("test", y)
            aegis.end_period()

        diag = aegis.get_diagnostics("test")
        assert diag["group_weights"].get("periodic", 0) > 0.01

    def test_regime_switching_signal(self, regime_switching_signal) -> None:
        """Test regime switching: system should adapt."""
        aegis = AEGIS()
        aegis.add_stream("test")

        signal = regime_switching_signal(n=300)

        errors = []
        for t, y in enumerate(signal):
            if t > 50:
                pred = aegis.predict("test", horizon=1)
                errors.append(abs(y - pred.mean))
            aegis.observe("test", y)
            aegis.end_period()

        mae = np.mean(errors)
        assert mae < 15.0

    def test_threshold_ar_signal(self, threshold_ar_signal) -> None:
        """Test threshold AR: reversion models should get weight."""
        aegis = AEGIS()
        aegis.add_stream("test")

        signal = threshold_ar_signal(n=300)

        for y in signal:
            aegis.observe("test", y)
            aegis.end_period()

        diag = aegis.get_diagnostics("test")
        assert diag["group_weights"].get("reversion", 0) > 0.005

    def test_asymmetric_ar1_signal(self, asymmetric_ar1_signal) -> None:
        """Test asymmetric AR(1): reversion models should get weight."""
        aegis = AEGIS()
        aegis.add_stream("test")

        signal = asymmetric_ar1_signal(n=300)

        for y in signal:
            aegis.observe("test", y)
            aegis.end_period()

        diag = aegis.get_diagnostics("test")
        assert "reversion" in diag["group_weights"]

    def test_heavy_tailed_signal(self, heavy_tailed_signal) -> None:
        """Test heavy-tailed: prediction intervals should widen."""
        aegis = AEGIS()
        aegis.add_stream("test")

        signal = heavy_tailed_signal(n=300, df=3)

        for y in signal:
            aegis.observe("test", y)
            aegis.end_period()

        pred = aegis.predict("test", horizon=1)
        assert np.isfinite(pred.mean)
        assert pred.variance > 0

    def test_correlated_streams(self, correlated_streams_signal) -> None:
        """Test correlated streams: cross-stream should learn."""
        config = AEGISConfig(cross_stream_lags=3)
        aegis = AEGIS(config=config)
        aegis.add_stream("A")
        aegis.add_stream("B")

        stream_a, stream_b = correlated_streams_signal(n=300, correlation=0.8)

        for t in range(len(stream_a)):
            aegis.observe("A", stream_a[t])
            aegis.observe("B", stream_b[t])
            aegis.end_period()

        diag = aegis.get_diagnostics("A")
        assert "cross_stream" in diag

    def test_lead_lag_streams(self, lead_lag_signal) -> None:
        """Test lead-lag: cross-stream should capture relationship."""
        config = AEGISConfig(cross_stream_lags=3)
        aegis = AEGIS(config=config)
        aegis.add_stream("leader")
        aegis.add_stream("follower")

        leader, follower = lead_lag_signal(n=500, lag=1, sigma=0.5)

        for t in range(len(leader)):
            # Call predict() to enable cross-stream learning via residuals
            if t > 0:
                aegis.predict("leader", horizon=1)
                aegis.predict("follower", horizon=1)
            aegis.observe("leader", leader[t])
            aegis.observe("follower", follower[t])
            aegis.end_period()

        diag = aegis.get_diagnostics("follower")
        coef = diag["cross_stream"]["coefficients"]["follower"]
        assert np.any(np.abs(coef) > 0.01)
