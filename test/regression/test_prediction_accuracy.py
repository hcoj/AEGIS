"""Regression tests for prediction accuracy.

These tests establish baseline MAE values and ensure future changes
don't cause significant degradation in prediction accuracy.

IMPORTANT: If a test fails, do NOT simply update the thresholds.
Investigate why the change caused a regression and fix the root cause.
"""

import numpy as np
import pytest

from aegis.config import AEGISConfig
from aegis.system import AEGIS


class SignalGenerator:
    """Generate test signals with fixed seed for reproducibility."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def white_noise(self, n: int = 500) -> np.ndarray:
        """White noise signal."""
        return self.rng.normal(0, 1, n)

    def random_walk(self, n: int = 500) -> np.ndarray:
        """Random walk signal."""
        return np.cumsum(self.rng.normal(0, 1, n))

    def ar1(self, n: int = 500, phi: float = 0.8, sigma: float = 0.5) -> np.ndarray:
        """AR(1) signal."""
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = phi * y[t - 1] + self.rng.normal(0, sigma)
        return y

    def linear_trend(self, n: int = 500, slope: float = 0.1) -> np.ndarray:
        """Linear trend signal."""
        return slope * np.arange(n)

    def trend_plus_noise(self, n: int = 500, slope: float = 0.05, sigma: float = 0.5) -> np.ndarray:
        """Linear trend with noise."""
        return slope * np.arange(n) + self.rng.normal(0, sigma, n)


def compute_mae(signal: np.ndarray, horizon: int, warmup: int = 100) -> float:
    """Compute MAE for a signal at a given horizon."""
    n = len(signal)
    config = AEGISConfig()
    aegis = AEGIS(config=config)
    aegis.add_stream("test")

    errors = []

    for t in range(n):
        if t > warmup and t + horizon < n:
            pred = aegis.predict("test", horizon=horizon)
            actual = signal[t + horizon]
            errors.append(abs(actual - pred.mean))

        aegis.observe("test", signal[t])
        aegis.end_period()

    return np.mean(errors) if errors else float("inf")


class TestPredictionAccuracyBaseline:
    """Test that prediction accuracy stays within acceptable bounds.

    These thresholds are based on v3 performance report values.
    A 20% degradation tolerance is allowed for natural variation.
    """

    # Baseline MAE values from v3 report with 20% tolerance
    # Format: (baseline, max_allowed = baseline * 1.2)
    TOLERANCE = 1.20  # 20% degradation allowed

    @pytest.fixture
    def gen(self) -> SignalGenerator:
        """Signal generator with fixed seed."""
        return SignalGenerator(seed=42)

    def test_white_noise_h1(self, gen: SignalGenerator) -> None:
        """White noise MAE at h=1 should not exceed baseline."""
        baseline = 1.13
        signal = gen.white_noise(n=500)
        mae = compute_mae(signal, horizon=1)
        assert mae < baseline * self.TOLERANCE, (
            f"White noise h=1 MAE regression: {mae:.3f} > {baseline * self.TOLERANCE:.3f}"
        )

    def test_white_noise_h64(self, gen: SignalGenerator) -> None:
        """White noise MAE at h=64 should not exceed baseline."""
        baseline = 1.29
        signal = gen.white_noise(n=500)
        mae = compute_mae(signal, horizon=64)
        assert mae < baseline * self.TOLERANCE, (
            f"White noise h=64 MAE regression: {mae:.3f} > {baseline * self.TOLERANCE:.3f}"
        )

    def test_random_walk_h1(self, gen: SignalGenerator) -> None:
        """Random walk MAE at h=1 should not exceed baseline."""
        baseline = 1.14
        signal = gen.random_walk(n=500)
        mae = compute_mae(signal, horizon=1)
        assert mae < baseline * self.TOLERANCE, (
            f"Random walk h=1 MAE regression: {mae:.3f} > {baseline * self.TOLERANCE:.3f}"
        )

    def test_random_walk_h64(self, gen: SignalGenerator) -> None:
        """Random walk MAE at h=64 should not exceed baseline."""
        baseline = 10.05
        signal = gen.random_walk(n=500)
        mae = compute_mae(signal, horizon=64)
        assert mae < baseline * self.TOLERANCE, (
            f"Random walk h=64 MAE regression: {mae:.3f} > {baseline * self.TOLERANCE:.3f}"
        )

    def test_ar1_h1(self, gen: SignalGenerator) -> None:
        """AR(1) phi=0.8 MAE at h=1 should not exceed baseline."""
        baseline = 0.57
        signal = gen.ar1(n=500, phi=0.8)
        mae = compute_mae(signal, horizon=1)
        assert mae < baseline * self.TOLERANCE, (
            f"AR(1) h=1 MAE regression: {mae:.3f} > {baseline * self.TOLERANCE:.3f}"
        )

    def test_ar1_h64(self, gen: SignalGenerator) -> None:
        """AR(1) phi=0.8 MAE at h=64 should not exceed baseline."""
        baseline = 1.91
        signal = gen.ar1(n=500, phi=0.8)
        mae = compute_mae(signal, horizon=64)
        assert mae < baseline * self.TOLERANCE, (
            f"AR(1) h=64 MAE regression: {mae:.3f} > {baseline * self.TOLERANCE:.3f}"
        )

    def test_linear_trend_h1(self, gen: SignalGenerator) -> None:
        """Linear trend MAE at h=1 should not exceed baseline."""
        baseline = 0.10
        signal = gen.linear_trend(n=500)
        mae = compute_mae(signal, horizon=1)
        assert mae < baseline * self.TOLERANCE, (
            f"Linear trend h=1 MAE regression: {mae:.3f} > {baseline * self.TOLERANCE:.3f}"
        )

    def test_linear_trend_h64(self, gen: SignalGenerator) -> None:
        """Linear trend MAE at h=64 should not exceed baseline."""
        baseline = 0.10
        signal = gen.linear_trend(n=500)
        mae = compute_mae(signal, horizon=64)
        assert mae < baseline * self.TOLERANCE, (
            f"Linear trend h=64 MAE regression: {mae:.3f} > {baseline * self.TOLERANCE:.3f}"
        )

    def test_trend_plus_noise_h1(self, gen: SignalGenerator) -> None:
        """Trend + noise MAE at h=1 should not exceed baseline."""
        baseline = 1.09
        signal = gen.trend_plus_noise(n=500)
        mae = compute_mae(signal, horizon=1)
        assert mae < baseline * self.TOLERANCE, (
            f"Trend+noise h=1 MAE regression: {mae:.3f} > {baseline * self.TOLERANCE:.3f}"
        )

    def test_trend_plus_noise_h64(self, gen: SignalGenerator) -> None:
        """Trend + noise MAE at h=64 should not exceed baseline."""
        baseline = 1.28
        signal = gen.trend_plus_noise(n=500)
        mae = compute_mae(signal, horizon=64)
        assert mae < baseline * self.TOLERANCE, (
            f"Trend+noise h=64 MAE regression: {mae:.3f} > {baseline * self.TOLERANCE:.3f}"
        )


class TestLongHorizonAccuracy:
    """Test long-horizon prediction accuracy.

    These tests specifically guard against regressions at h=1024
    which was the main issue with the scale weight change.

    Long horizon tests have higher tolerance due to:
    1. Higher variance in long-horizon predictions
    2. Sensitivity to signal realization
    """

    TOLERANCE = 1.50  # 50% tolerance for long horizons (high variance)

    @pytest.fixture
    def gen(self) -> SignalGenerator:
        """Signal generator with fixed seed."""
        return SignalGenerator(seed=42)

    def test_random_walk_h1024(self, gen: SignalGenerator) -> None:
        """Random walk MAE at h=1024 should not exceed baseline."""
        baseline = 110.37
        signal = gen.random_walk(n=2000)
        mae = compute_mae(signal, horizon=1024, warmup=200)
        assert mae < baseline * self.TOLERANCE, (
            f"Random walk h=1024 MAE regression: {mae:.2f} > {baseline * self.TOLERANCE:.2f}"
        )

    def test_ar1_h1024(self, gen: SignalGenerator) -> None:
        """AR(1) MAE at h=1024 should not exceed baseline."""
        baseline = 28.30
        signal = gen.ar1(n=2000, phi=0.8)
        mae = compute_mae(signal, horizon=1024, warmup=200)
        assert mae < baseline * self.TOLERANCE, (
            f"AR(1) h=1024 MAE regression: {mae:.2f} > {baseline * self.TOLERANCE:.2f}"
        )


class TestAccuracyRatios:
    """Test that error growth ratios stay reasonable.

    These tests ensure that long-horizon errors don't grow
    excessively compared to short-horizon errors.
    """

    @pytest.fixture
    def gen(self) -> SignalGenerator:
        """Signal generator with fixed seed."""
        return SignalGenerator(seed=42)

    def test_ar1_error_growth_ratio(self, gen: SignalGenerator) -> None:
        """AR(1) error growth from h=1 to h=64 should be bounded."""
        signal = gen.ar1(n=500, phi=0.8)
        mae_h1 = compute_mae(signal, horizon=1)
        mae_h64 = compute_mae(signal, horizon=64)

        ratio = mae_h64 / mae_h1 if mae_h1 > 0 else float("inf")

        # Based on v3: h=1 MAE 0.57, h=64 MAE 1.91, ratio ~3.4x
        # Allow up to 5x growth
        max_ratio = 5.0
        assert ratio < max_ratio, f"AR(1) error growth ratio too high: {ratio:.1f}x > {max_ratio}x"

    def test_random_walk_error_growth_ratio(self, gen: SignalGenerator) -> None:
        """Random walk error growth from h=1 to h=64 should be bounded."""
        signal = gen.random_walk(n=500)
        mae_h1 = compute_mae(signal, horizon=1)
        mae_h64 = compute_mae(signal, horizon=64)

        ratio = mae_h64 / mae_h1 if mae_h1 > 0 else float("inf")

        # Based on v3: h=1 MAE 1.14, h=64 MAE 10.05, ratio ~8.8x
        # Random walk error grows as sqrt(h), so h=64 should be ~8x h=1
        # Allow up to 12x growth
        max_ratio = 12.0
        assert ratio < max_ratio, (
            f"Random walk error growth ratio too high: {ratio:.1f}x > {max_ratio}x"
        )
