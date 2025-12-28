"""Pytest configuration and fixtures for AEGIS tests."""

from collections.abc import Callable

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """Reproducible random number generator."""
    return np.random.default_rng(42)


@pytest.fixture
def constant_signal() -> Callable[[int, float], np.ndarray]:
    """Generate a constant signal.

    Args:
        n: Number of observations
        c: Constant value (default 5.0)

    Returns:
        Array of constant values
    """

    def generate(n: int, c: float = 5.0) -> np.ndarray:
        return np.full(n, c)

    return generate


@pytest.fixture
def white_noise_signal(rng: np.random.Generator) -> Callable[[int, float], np.ndarray]:
    """Generate white noise (i.i.d. Gaussian).

    Args:
        n: Number of observations
        sigma: Standard deviation (default 1.0)

    Returns:
        Array of i.i.d. Gaussian noise
    """

    def generate(n: int, sigma: float = 1.0) -> np.ndarray:
        return rng.normal(0, sigma, n)

    return generate


@pytest.fixture
def random_walk_signal(rng: np.random.Generator) -> Callable[[int, float], np.ndarray]:
    """Generate a random walk (cumulative sum of noise).

    Args:
        n: Number of observations
        sigma: Innovation standard deviation (default 1.0)

    Returns:
        Cumulative sum of Gaussian innovations
    """

    def generate(n: int, sigma: float = 1.0) -> np.ndarray:
        innovations = rng.normal(0, sigma, n)
        return np.cumsum(innovations)

    return generate


@pytest.fixture
def ar1_signal(rng: np.random.Generator) -> Callable[[int, float, float], np.ndarray]:
    """Generate an AR(1) process.

    Args:
        n: Number of observations
        phi: Autoregressive coefficient (default 0.9)
        sigma: Innovation standard deviation (default 1.0)

    Returns:
        AR(1) time series
    """

    def generate(n: int, phi: float = 0.9, sigma: float = 1.0) -> np.ndarray:
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = phi * y[t - 1] + rng.normal(0, sigma)
        return y

    return generate


@pytest.fixture
def linear_trend_signal() -> Callable[[int, float, float], np.ndarray]:
    """Generate a linear trend signal.

    Args:
        n: Number of observations
        slope: Slope of trend (default 0.1)
        intercept: Y-intercept (default 0.0)

    Returns:
        Linear trend array
    """

    def generate(n: int, slope: float = 0.1, intercept: float = 0.0) -> np.ndarray:
        return intercept + slope * np.arange(n)

    return generate


@pytest.fixture
def sine_wave_signal() -> Callable[[int, int, float], np.ndarray]:
    """Generate a sinusoidal signal.

    Args:
        n: Number of observations
        period: Period of oscillation (default 16)
        amplitude: Amplitude (default 1.0)

    Returns:
        Sinusoidal time series
    """

    def generate(n: int, period: int = 16, amplitude: float = 1.0) -> np.ndarray:
        return amplitude * np.sin(2 * np.pi * np.arange(n) / period)

    return generate


@pytest.fixture
def threshold_ar_signal(
    rng: np.random.Generator,
) -> Callable[[int, float, float, float, float], np.ndarray]:
    """Generate a threshold AR process with regime-dependent dynamics.

    Args:
        n: Number of observations
        tau: Threshold value (default 0.0)
        phi_low: AR coefficient below threshold (default 0.5)
        phi_high: AR coefficient above threshold (default 0.95)
        sigma: Innovation standard deviation (default 0.5)

    Returns:
        Threshold AR time series
    """

    def generate(
        n: int,
        tau: float = 0.0,
        phi_low: float = 0.5,
        phi_high: float = 0.95,
        sigma: float = 0.5,
    ) -> np.ndarray:
        y = np.zeros(n)
        for t in range(1, n):
            if y[t - 1] < tau:
                y[t] = phi_low * y[t - 1] + rng.normal(0, sigma)
            else:
                y[t] = phi_high * y[t - 1] + rng.normal(0, sigma)
        return y

    return generate


@pytest.fixture
def regime_switching_signal(
    rng: np.random.Generator,
) -> Callable[[int, int | None, float, float, float], np.ndarray]:
    """Generate a regime-switching signal with mean shift.

    Args:
        n: Number of observations
        break_point: Time of regime change (default n//2)
        mean1: Mean in first regime (default 0.0)
        mean2: Mean in second regime (default 5.0)
        sigma: Standard deviation in both regimes (default 1.0)

    Returns:
        Regime-switching time series
    """

    def generate(
        n: int,
        break_point: int | None = None,
        mean1: float = 0.0,
        mean2: float = 5.0,
        sigma: float = 1.0,
    ) -> np.ndarray:
        if break_point is None:
            break_point = n // 2

        y = np.zeros(n)
        y[:break_point] = rng.normal(mean1, sigma, break_point)
        y[break_point:] = rng.normal(mean2, sigma, n - break_point)
        return y

    return generate


@pytest.fixture
def jump_diffusion_signal(
    rng: np.random.Generator,
) -> Callable[[int, float, float, float], np.ndarray]:
    """Generate a jump-diffusion process.

    Args:
        n: Number of observations
        sigma_diff: Diffusion standard deviation (default 0.5)
        jump_prob: Probability of jump per period (default 0.02)
        jump_size: Mean absolute jump size (default 5.0)

    Returns:
        Jump-diffusion time series
    """

    def generate(
        n: int,
        sigma_diff: float = 0.5,
        jump_prob: float = 0.02,
        jump_size: float = 5.0,
    ) -> np.ndarray:
        y = np.zeros(n)
        for t in range(1, n):
            diffusion = rng.normal(0, sigma_diff)
            jump = jump_size * (rng.random() < jump_prob) * rng.choice([-1, 1])
            y[t] = y[t - 1] + diffusion + jump
        return y

    return generate


@pytest.fixture
def seasonal_signal(
    rng: np.random.Generator,
) -> Callable[[int, int, list[float] | None, float], np.ndarray]:
    """Generate a seasonal signal with periodic pattern.

    Args:
        n: Number of observations
        period: Seasonal period (default 7)
        pattern: Seasonal pattern values (default weekly pattern)
        noise_sigma: Noise standard deviation (default 0.5)

    Returns:
        Seasonal time series
    """

    def generate(
        n: int,
        period: int = 7,
        pattern: list[float] | None = None,
        noise_sigma: float = 0.5,
    ) -> np.ndarray:
        if pattern is None:
            pattern = [10.0, 12.0, 15.0, 14.0, 13.0, 8.0, 5.0]

        y = np.zeros(n)
        for t in range(n):
            y[t] = pattern[t % period] + rng.normal(0, noise_sigma)
        return y

    return generate


@pytest.fixture
def asymmetric_ar1_signal(
    rng: np.random.Generator,
) -> Callable[[int, float, float, float], np.ndarray]:
    """Generate an asymmetric AR(1) with different reversion above/below zero.

    Args:
        n: Number of observations
        phi_up: AR coefficient when above zero (default 0.7)
        phi_down: AR coefficient when below zero (default 0.95)
        sigma: Innovation standard deviation (default 0.5)

    Returns:
        Asymmetric AR(1) time series
    """

    def generate(
        n: int,
        phi_up: float = 0.7,
        phi_down: float = 0.95,
        sigma: float = 0.5,
    ) -> np.ndarray:
        y = np.zeros(n)
        for t in range(1, n):
            if y[t - 1] > 0:
                y[t] = phi_up * y[t - 1] + rng.normal(0, sigma)
            else:
                y[t] = phi_down * y[t - 1] + rng.normal(0, sigma)
        return y

    return generate


@pytest.fixture
def heavy_tailed_signal(rng: np.random.Generator) -> Callable[[int, float], np.ndarray]:
    """Generate a random walk with heavy-tailed (Student-t) innovations.

    Args:
        n: Number of observations
        df: Degrees of freedom for t-distribution (default 4.0)

    Returns:
        Heavy-tailed random walk
    """

    def generate(n: int, df: float = 4.0) -> np.ndarray:
        from scipy.stats import t

        innovations = t.rvs(df, size=n, random_state=rng)
        return np.cumsum(innovations)

    return generate


@pytest.fixture
def contaminated_signal(
    rng: np.random.Generator,
) -> Callable[[int, float, float, float], np.ndarray]:
    """Generate contaminated random walk with outliers.

    Args:
        n: Number of observations
        sigma: Base innovation standard deviation (default 1.0)
        contamination_prob: Probability of outlier (default 0.02)
        contamination_scale: Outlier magnitude multiplier (default 10.0)

    Returns:
        Random walk with occasional large outliers
    """

    def generate(
        n: int = 500,
        sigma: float = 1.0,
        contamination_prob: float = 0.02,
        contamination_scale: float = 10.0,
    ) -> np.ndarray:
        y = np.zeros(n)
        for t in range(1, n):
            innovation = rng.normal(0, sigma)
            if rng.random() < contamination_prob:
                innovation *= contamination_scale
            y[t] = y[t - 1] + innovation
        return y

    return generate


@pytest.fixture
def correlated_streams_signal(
    rng: np.random.Generator,
) -> Callable[[int, float], tuple[np.ndarray, np.ndarray]]:
    """Generate two correlated random walk streams.

    Args:
        n: Number of observations
        correlation: Correlation coefficient (default 0.8)

    Returns:
        Tuple of two correlated random walk arrays
    """

    def generate(n: int, correlation: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
        z1 = rng.normal(0, 1, n)
        z2 = correlation * z1 + np.sqrt(1 - correlation**2) * rng.normal(0, 1, n)

        y1 = np.cumsum(z1)
        y2 = np.cumsum(z2)
        return y1, y2

    return generate


@pytest.fixture
def lead_lag_signal(
    rng: np.random.Generator,
) -> Callable[[int, int, float], tuple[np.ndarray, np.ndarray]]:
    """Generate leader-follower streams with lag relationship.

    Args:
        n: Number of observations
        lag: Number of periods follower lags leader (default 3)
        sigma: Noise in follower (default 0.5)

    Returns:
        Tuple of (leader, follower) arrays
    """

    def generate(n: int, lag: int = 3, sigma: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
        leader = np.cumsum(rng.normal(0, 1, n))
        follower = np.zeros(n)
        for t in range(lag, n):
            follower[t] = leader[t - lag] + rng.normal(0, sigma)
        return leader, follower

    return generate


@pytest.fixture
def signal_generators(
    constant_signal: Callable,
    white_noise_signal: Callable,
    random_walk_signal: Callable,
    ar1_signal: Callable,
    linear_trend_signal: Callable,
    sine_wave_signal: Callable,
    threshold_ar_signal: Callable,
    regime_switching_signal: Callable,
    seasonal_signal: Callable,
) -> dict[str, Callable]:
    """Dictionary of named signal generators for parametrized tests.

    Returns:
        Dictionary mapping signal names to generator functions
    """
    return {
        "constant": constant_signal,
        "white_noise": white_noise_signal,
        "random_walk": random_walk_signal,
        "ar1_0.9": lambda n: ar1_signal(n, phi=0.9),
        "ar1_0.5": lambda n: ar1_signal(n, phi=0.5),
        "linear_trend": linear_trend_signal,
        "sine_16": lambda n: sine_wave_signal(n, period=16),
        "sine_32": lambda n: sine_wave_signal(n, period=32),
        "threshold_ar": threshold_ar_signal,
        "regime_switch": regime_switching_signal,
        "seasonal_7": lambda n: seasonal_signal(n, period=7),
    }
