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


def compute_coverage(signal: np.ndarray, horizon: int, warmup: int = 100) -> float:
    """Compute 95% interval coverage for a signal at a given horizon."""
    n = len(signal)
    config = AEGISConfig(use_quantile_calibration=True)
    aegis = AEGIS(config=config)
    aegis.add_stream("test")

    in_interval = 0
    total = 0

    for t in range(n):
        if t > warmup and t + horizon < n:
            pred = aegis.predict("test", horizon=horizon)
            actual = signal[t + horizon]

            if pred.interval_lower is not None and pred.interval_upper is not None:
                if pred.interval_lower <= actual <= pred.interval_upper:
                    in_interval += 1
                total += 1

        aegis.observe("test", signal[t])
        aegis.end_period()

    return in_interval / total if total > 0 else 0.0


class TestCoverageBaseline:
    """Test that prediction interval coverage is reasonable.

    Target coverage is 95%. We allow coverage to be between 85% and 100%
    to account for:
    - Calibration learning time
    - Signal-specific variance patterns
    - Small sample sizes

    Coverage below 85% indicates intervals are too narrow.
    Coverage at 100% might indicate intervals are too wide.
    """

    MIN_COVERAGE = 0.85  # Minimum acceptable coverage
    TARGET_COVERAGE = 0.95

    @pytest.fixture
    def gen(self) -> SignalGenerator:
        """Signal generator with fixed seed."""
        return SignalGenerator(seed=42)

    def test_white_noise_coverage_h1(self, gen: SignalGenerator) -> None:
        """White noise h=1 coverage should be near 95%."""
        signal = gen.white_noise(n=800)
        coverage = compute_coverage(signal, horizon=1, warmup=200)
        assert coverage >= self.MIN_COVERAGE, (
            f"White noise h=1 coverage too low: {coverage:.1%} < {self.MIN_COVERAGE:.0%}"
        )

    def test_random_walk_coverage_h1(self, gen: SignalGenerator) -> None:
        """Random walk h=1 coverage should be near 95%."""
        signal = gen.random_walk(n=800)
        coverage = compute_coverage(signal, horizon=1, warmup=200)
        assert coverage >= self.MIN_COVERAGE, (
            f"Random walk h=1 coverage too low: {coverage:.1%} < {self.MIN_COVERAGE:.0%}"
        )

    def test_ar1_coverage_h1(self, gen: SignalGenerator) -> None:
        """AR(1) h=1 coverage should be near 95%."""
        signal = gen.ar1(n=800, phi=0.8)
        coverage = compute_coverage(signal, horizon=1, warmup=200)
        assert coverage >= self.MIN_COVERAGE, (
            f"AR(1) h=1 coverage too low: {coverage:.1%} < {self.MIN_COVERAGE:.0%}"
        )

    def test_trend_plus_noise_coverage_h1(self, gen: SignalGenerator) -> None:
        """Trend + noise h=1 coverage should be near 95%."""
        signal = gen.trend_plus_noise(n=800)
        coverage = compute_coverage(signal, horizon=1, warmup=200)
        assert coverage >= self.MIN_COVERAGE, (
            f"Trend+noise h=1 coverage too low: {coverage:.1%} < {self.MIN_COVERAGE:.0%}"
        )


class TestPeriodicLongHorizon:
    """Test periodic model long-horizon accuracy.

    Phase-locking should prevent error explosion at long horizons
    for sinusoidal signals.
    """

    def test_sinusoidal_h1024_mae_bounded(self) -> None:
        """Sinusoidal h=1024 MAE should be < 5.0 (was 18.51 before fix)."""
        from aegis.models.periodic import OscillatorBankModel

        n, period = 2000, 16
        signal = np.sin(2 * np.pi * np.arange(n) / period)

        model = OscillatorBankModel(periods=[period], lr=0.05)
        errors = []
        for t in range(n):
            if t >= 500 and t + 1024 < n:
                pred = model.predict(1024).mean
                true_cumsum = sum(signal[t + 1 : t + 1025])
                errors.append(abs(pred - true_cumsum))
            model.update(signal[t], t)

        mae = np.mean(errors) if errors else 0
        assert mae < 5.0, f"Sinusoidal h=1024 MAE {mae:.2f} > 5.0"

    def test_sinusoidal_error_growth_bounded(self) -> None:
        """Sinusoidal error growth h=1 to h=1024 should be < 50x (was 147x)."""
        from aegis.models.periodic import OscillatorBankModel

        n, period = 2000, 16
        signal = np.sin(2 * np.pi * np.arange(n) / period)

        model = OscillatorBankModel(periods=[period], lr=0.05)

        for t in range(500):
            model.update(signal[t], t)

        # Compute errors at h=1 and h=1024
        errors_h1 = []
        errors_h1024 = []

        for t in range(500, min(1500, n - 1024)):
            pred_h1 = model.predict(1).mean
            true_h1 = signal[t + 1]
            errors_h1.append(abs(pred_h1 - true_h1))

            pred_h1024 = model.predict(1024).mean
            true_h1024 = sum(signal[t + 1 : t + 1025])
            errors_h1024.append(abs(pred_h1024 - true_h1024))

            model.update(signal[t], t)

        mae_h1 = np.mean(errors_h1)
        mae_h1024 = np.mean(errors_h1024)
        growth = (mae_h1024 + 0.01) / (mae_h1 + 0.01)

        assert growth < 50.0, f"Sinusoidal error growth {growth:.1f}x exceeds 50x"
