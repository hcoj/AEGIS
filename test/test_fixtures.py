"""Smoke tests for signal generator fixtures."""

import numpy as np


def test_rng_reproducible(rng: np.random.Generator) -> None:
    """Test that RNG produces reproducible results."""
    first = rng.random()
    rng2 = np.random.default_rng(42)
    assert first == rng2.random()


def test_constant_signal(constant_signal) -> None:
    """Test constant signal generator."""
    signal = constant_signal(n=100, c=5.0)
    assert len(signal) == 100
    assert np.all(signal == 5.0)


def test_white_noise_signal(white_noise_signal) -> None:
    """Test white noise signal generator."""
    signal = white_noise_signal(n=1000, sigma=1.0)
    assert len(signal) == 1000
    assert np.abs(np.mean(signal)) < 0.2
    assert np.abs(np.std(signal) - 1.0) < 0.2


def test_random_walk_signal(random_walk_signal) -> None:
    """Test random walk signal generator."""
    signal = random_walk_signal(n=100)
    assert len(signal) == 100
    assert signal[0] != 0 or len(np.unique(signal)) > 1


def test_ar1_signal(ar1_signal) -> None:
    """Test AR(1) signal generator."""
    signal = ar1_signal(n=500, phi=0.9)
    assert len(signal) == 500
    correlation = np.corrcoef(signal[:-1], signal[1:])[0, 1]
    assert correlation > 0.7


def test_linear_trend_signal(linear_trend_signal) -> None:
    """Test linear trend signal generator."""
    signal = linear_trend_signal(n=100, slope=2.0, intercept=5.0)
    assert len(signal) == 100
    assert signal[0] == 5.0
    assert signal[99] == 5.0 + 2.0 * 99


def test_sine_wave_signal(sine_wave_signal) -> None:
    """Test sine wave signal generator."""
    signal = sine_wave_signal(n=64, period=16, amplitude=2.0)
    assert len(signal) == 64
    assert np.max(signal) <= 2.0
    assert np.min(signal) >= -2.0


def test_threshold_ar_signal(threshold_ar_signal) -> None:
    """Test threshold AR signal generator."""
    signal = threshold_ar_signal(n=500)
    assert len(signal) == 500


def test_regime_switching_signal(regime_switching_signal) -> None:
    """Test regime switching signal generator."""
    signal = regime_switching_signal(n=200, break_point=100, mean1=0.0, mean2=10.0)
    assert len(signal) == 200
    assert np.abs(np.mean(signal[:100])) < 2.0
    assert np.abs(np.mean(signal[100:]) - 10.0) < 2.0


def test_jump_diffusion_signal(jump_diffusion_signal) -> None:
    """Test jump diffusion signal generator."""
    signal = jump_diffusion_signal(n=500, jump_prob=0.05, jump_size=10.0)
    assert len(signal) == 500


def test_seasonal_signal(seasonal_signal) -> None:
    """Test seasonal signal generator."""
    pattern = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    signal = seasonal_signal(n=70, period=7, pattern=pattern, noise_sigma=0.1)
    assert len(signal) == 70


def test_asymmetric_ar1_signal(asymmetric_ar1_signal) -> None:
    """Test asymmetric AR(1) signal generator."""
    signal = asymmetric_ar1_signal(n=500)
    assert len(signal) == 500


def test_heavy_tailed_signal(heavy_tailed_signal) -> None:
    """Test heavy-tailed signal generator."""
    signal = heavy_tailed_signal(n=100, df=4.0)
    assert len(signal) == 100


def test_correlated_streams_signal(correlated_streams_signal) -> None:
    """Test correlated streams signal generator."""
    y1, y2 = correlated_streams_signal(n=500, correlation=0.9)
    assert len(y1) == 500
    assert len(y2) == 500
    diff1 = np.diff(y1)
    diff2 = np.diff(y2)
    corr = np.corrcoef(diff1, diff2)[0, 1]
    assert corr > 0.7


def test_lead_lag_signal(lead_lag_signal) -> None:
    """Test lead-lag signal generator."""
    leader, follower = lead_lag_signal(n=100, lag=3)
    assert len(leader) == 100
    assert len(follower) == 100
    assert follower[0] == 0
    assert follower[1] == 0
    assert follower[2] == 0


def test_signal_generators_dict(signal_generators) -> None:
    """Test that signal_generators dictionary works."""
    assert "random_walk" in signal_generators
    assert "ar1_0.9" in signal_generators
    signal = signal_generators["random_walk"](n=100)
    assert len(signal) == 100
