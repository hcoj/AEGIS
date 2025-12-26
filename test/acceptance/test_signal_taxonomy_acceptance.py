"""Acceptance tests for AEGIS signal taxonomy.

Comprehensive tests covering all signal types from Appendix D,
measuring performance metrics and generating a report.
"""

import dataclasses
import json
import time
from pathlib import Path

import numpy as np

from aegis.config import AEGISConfig
from aegis.system import AEGIS


@dataclasses.dataclass
class SignalMetrics:
    """Metrics collected for a signal test."""

    signal_name: str
    signal_category: str
    n_observations: int
    mae: float
    rmse: float
    coverage_95: float | None
    mean_interval_width: float | None
    dominant_group: str
    dominant_group_weight: float
    expected_dominant: str
    dominant_matches_expected: bool
    runtime_seconds: float
    notes: str = ""


class SignalGenerators:
    """Signal generators for all taxonomy types."""

    def __init__(self, seed: int = 42) -> None:
        """Initialize with random seed."""
        self.rng = np.random.default_rng(seed)

    # =========================================================================
    # 2. Deterministic Signals
    # =========================================================================

    def constant(self, n: int = 200, c: float = 5.0) -> np.ndarray:
        """Constant signal: y_t = c."""
        return np.full(n, c)

    def linear_trend(self, n: int = 200, slope: float = 0.5, intercept: float = 0.0) -> np.ndarray:
        """Linear trend: y_t = a*t + b."""
        t = np.arange(n)
        return slope * t + intercept

    def sinusoidal(
        self, n: int = 200, period: int = 16, amplitude: float = 2.0, phase: float = 0.0
    ) -> np.ndarray:
        """Sinusoidal: y_t = A*sin(omega*t + phi)."""
        t = np.arange(n)
        omega = 2 * np.pi / period
        return amplitude * np.sin(omega * t + phase)

    def square_wave(self, n: int = 200, period: int = 7, amplitude: float = 1.0) -> np.ndarray:
        """Square wave / sharp seasonal pattern."""
        t = np.arange(n)
        phase = t % period
        return np.where(phase < period // 2, amplitude, -amplitude)

    def polynomial_trend(
        self, n: int = 200, a: float = 0.001, b: float = 0.1, c: float = 0.0
    ) -> np.ndarray:
        """Polynomial trend: y_t = a*t^2 + b*t + c."""
        t = np.arange(n)
        return a * t**2 + b * t + c

    # =========================================================================
    # 3. Simple Stochastic Processes
    # =========================================================================

    def white_noise(self, n: int = 200, sigma: float = 1.0) -> np.ndarray:
        """White noise: y_t ~ N(0, sigma^2) i.i.d."""
        return self.rng.normal(0, sigma, n)

    def random_walk(self, n: int = 200, sigma: float = 1.0, y0: float = 0.0) -> np.ndarray:
        """Random walk: y_t = y_{t-1} + epsilon_t."""
        innovations = self.rng.normal(0, sigma, n)
        y = np.zeros(n)
        y[0] = y0 + innovations[0]
        for t in range(1, n):
            y[t] = y[t - 1] + innovations[t]
        return y

    def ar1_mean_reverting(
        self, n: int = 300, phi: float = 0.8, sigma: float = 0.5, mu: float = 0.0
    ) -> np.ndarray:
        """AR(1) mean-reverting: y_t = phi*y_{t-1} + epsilon_t."""
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = phi * y[t - 1] + self.rng.normal(0, sigma)
        return y + mu

    def ar1_near_unit_root(self, n: int = 500, phi: float = 0.99, sigma: float = 0.5) -> np.ndarray:
        """AR(1) near unit root: phi close to 1."""
        return self.ar1_mean_reverting(n, phi, sigma)

    def ma1(self, n: int = 200, theta: float = 0.6, sigma: float = 1.0) -> np.ndarray:
        """MA(1): y_t = epsilon_t + theta*epsilon_{t-1}."""
        eps = self.rng.normal(0, sigma, n + 1)
        y = np.zeros(n)
        for t in range(n):
            y[t] = eps[t + 1] + theta * eps[t]
        return y

    def arma11(
        self, n: int = 300, phi: float = 0.7, theta: float = 0.3, sigma: float = 1.0
    ) -> np.ndarray:
        """ARMA(1,1): y_t = phi*y_{t-1} + epsilon_t + theta*epsilon_{t-1}."""
        eps = self.rng.normal(0, sigma, n + 1)
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = phi * y[t - 1] + eps[t + 1] + theta * eps[t]
        return y

    def ornstein_uhlenbeck(
        self, n: int = 300, theta: float = 0.1, mu: float = 0.0, sigma: float = 0.5
    ) -> np.ndarray:
        """Ornstein-Uhlenbeck: discretized mean-reverting diffusion."""
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = y[t - 1] + theta * (mu - y[t - 1]) + self.rng.normal(0, sigma)
        return y

    # =========================================================================
    # 4. Composite Signals
    # =========================================================================

    def trend_plus_noise(self, n: int = 200, slope: float = 0.3, sigma: float = 1.0) -> np.ndarray:
        """Trend + noise: y_t = a*t + epsilon_t."""
        t = np.arange(n)
        return slope * t + self.rng.normal(0, sigma, n)

    def sine_plus_noise(
        self, n: int = 200, period: int = 16, amplitude: float = 2.0, sigma: float = 0.5
    ) -> np.ndarray:
        """Sine + noise: y_t = A*sin(omega*t) + epsilon_t."""
        return self.sinusoidal(n, period, amplitude) + self.rng.normal(0, sigma, n)

    def trend_seasonality_noise(
        self,
        n: int = 300,
        slope: float = 0.1,
        period: int = 7,
        seasonal_amp: float = 2.0,
        sigma: float = 0.5,
    ) -> np.ndarray:
        """Trend + seasonality + noise."""
        t = np.arange(n)
        trend = slope * t
        seasonal = seasonal_amp * np.sin(2 * np.pi * t / period)
        noise = self.rng.normal(0, sigma, n)
        return trend + seasonal + noise

    def mean_reversion_plus_oscillation(
        self,
        n: int = 300,
        phi: float = 0.7,
        period: int = 16,
        amplitude: float = 1.0,
        sigma: float = 0.3,
    ) -> np.ndarray:
        """Mean-reversion + oscillation."""
        ar = self.ar1_mean_reverting(n, phi, sigma)
        osc = self.sinusoidal(n, period, amplitude)
        return ar + osc

    # =========================================================================
    # 5. Non-Stationary and Regime-Changing
    # =========================================================================

    def random_walk_with_drift(
        self, n: int = 200, drift: float = 0.1, sigma: float = 1.0
    ) -> np.ndarray:
        """Random walk with drift: y_t = mu + y_{t-1} + epsilon_t."""
        innovations = self.rng.normal(drift, sigma, n)
        y = np.zeros(n)
        y[0] = innovations[0]
        for t in range(1, n):
            y[t] = y[t - 1] + innovations[t]
        return y

    def variance_switching(
        self,
        n: int = 300,
        sigma_low: float = 0.5,
        sigma_high: float = 3.0,
        switch_prob: float = 0.02,
    ) -> np.ndarray:
        """Variance switching: two-state volatility."""
        y = np.zeros(n)
        high_vol = False
        for t in range(n):
            if self.rng.random() < switch_prob:
                high_vol = not high_vol
            sigma = sigma_high if high_vol else sigma_low
            y[t] = self.rng.normal(0, sigma)
        return y

    def mean_switching(
        self,
        n: int = 300,
        mu_a: float = 0.0,
        mu_b: float = 5.0,
        sigma: float = 1.0,
        switch_prob: float = 0.02,
    ) -> np.ndarray:
        """Mean switching: two-state mean."""
        y = np.zeros(n)
        state_b = False
        for t in range(n):
            if self.rng.random() < switch_prob:
                state_b = not state_b
            mu = mu_b if state_b else mu_a
            y[t] = self.rng.normal(mu, sigma)
        return y

    def threshold_ar(
        self,
        n: int = 300,
        phi_low: float = 0.9,
        phi_high: float = -0.5,
        threshold: float = 0.0,
        sigma: float = 0.5,
    ) -> np.ndarray:
        """Threshold AR: different dynamics above/below threshold."""
        y = np.zeros(n)
        for t in range(1, n):
            phi = phi_high if y[t - 1] > threshold else phi_low
            y[t] = phi * y[t - 1] + self.rng.normal(0, sigma)
        return y

    def structural_break(
        self,
        n: int = 300,
        break_time: int = 150,
        mu_before: float = 0.0,
        mu_after: float = 5.0,
        sigma: float = 1.0,
    ) -> np.ndarray:
        """Structural break: parameter change at known time."""
        y = np.zeros(n)
        for t in range(n):
            mu = mu_after if t >= break_time else mu_before
            y[t] = self.rng.normal(mu, sigma)
        return y

    def gradual_drift(
        self, n: int = 500, drift_rate: float = 0.01, sigma: float = 1.0
    ) -> np.ndarray:
        """Gradual drift: slowly changing mean."""
        t = np.arange(n)
        drift = drift_rate * t
        return drift + self.rng.normal(0, sigma, n)

    # =========================================================================
    # 6. Heavy-Tailed Signals
    # =========================================================================

    def student_t_innovations(self, n: int = 300, df: float = 4.0, phi: float = 0.7) -> np.ndarray:
        """Student-t innovations: AR(1) with t-distributed errors."""
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = phi * y[t - 1] + self.rng.standard_t(df)
        return y

    def occasional_jumps(
        self, n: int = 300, sigma: float = 0.5, jump_prob: float = 0.02, jump_size: float = 5.0
    ) -> np.ndarray:
        """Random walk with occasional jumps."""
        y = np.zeros(n)
        for t in range(1, n):
            innovation = self.rng.normal(0, sigma)
            if self.rng.random() < jump_prob:
                innovation += self.rng.choice([-1, 1]) * jump_size
            y[t] = y[t - 1] + innovation
        return y

    def power_law_tails(self, n: int = 300, alpha: float = 2.5) -> np.ndarray:
        """Power-law tailed innovations (Pareto-like)."""
        y = np.zeros(n)
        for t in range(1, n):
            # Generate Pareto-like noise
            u = self.rng.random()
            sign = self.rng.choice([-1, 1])
            innovation = sign * ((1 - u) ** (-1 / alpha) - 1)
            y[t] = y[t - 1] + np.clip(innovation, -10, 10)  # Clip extremes
        return y

    # =========================================================================
    # 7. Multi-Scale Structure
    # =========================================================================

    def fractional_brownian_persistent(self, n: int = 500, h: float = 0.7) -> np.ndarray:
        """Fractional Brownian motion H > 0.5 (persistent)."""
        # Approximate using correlated increments
        d = h - 0.5
        # Start from k=1 to avoid 0^d issue for negative d
        weights = np.array([(k + 1) ** d - max(k, 0.001) ** d for k in range(min(n, 100))])
        weights = np.abs(weights)  # Handle sign issues
        weights = weights / (np.sum(weights) + 1e-10)

        noise = self.rng.normal(0, 1, n + len(weights))
        increments = np.convolve(noise, weights, mode="valid")[:n]
        return np.cumsum(increments)

    def fractional_brownian_antipersistent(self, n: int = 500, h: float = 0.3) -> np.ndarray:
        """Fractional Brownian motion H < 0.5 (antipersistent/mean-reverting)."""
        return self.fractional_brownian_persistent(n, h)

    def multi_timescale_mr(
        self, n: int = 500, phi_fast: float = 0.5, phi_slow: float = 0.98, sigma: float = 0.5
    ) -> np.ndarray:
        """Multi-timescale mean-reversion: fast + slow components."""
        fast = np.zeros(n)
        slow = np.zeros(n)
        for t in range(1, n):
            fast[t] = phi_fast * fast[t - 1] + self.rng.normal(0, sigma)
            slow[t] = phi_slow * slow[t - 1] + self.rng.normal(0, sigma * 0.3)
        return fast + slow

    def trend_momentum_reversion(
        self,
        n: int = 500,
        trend_strength: float = 0.05,
        momentum_phi: float = 0.3,
        reversion_phi: float = 0.98,
        sigma: float = 0.5,
    ) -> np.ndarray:
        """Trend + momentum + reversion at different timescales."""
        t = np.arange(n)
        trend = trend_strength * t

        # Short-term momentum (AR2-like)
        momentum = np.zeros(n)
        for i in range(2, n):
            momentum[i] = momentum_phi * momentum[i - 1] + self.rng.normal(0, sigma)

        # Long-term reversion
        reversion = np.zeros(n)
        for i in range(1, n):
            reversion[i] = reversion_phi * reversion[i - 1] + self.rng.normal(0, sigma * 0.2)

        return trend + momentum + reversion

    def garch_like(
        self, n: int = 500, alpha: float = 0.1, beta: float = 0.85, omega: float = 0.05
    ) -> np.ndarray:
        """GARCH-like volatility clustering."""
        y = np.zeros(n)
        sigma2 = np.zeros(n)
        sigma2[0] = omega / (1 - alpha - beta)

        for t in range(1, n):
            sigma2[t] = omega + alpha * y[t - 1] ** 2 + beta * sigma2[t - 1]
            y[t] = self.rng.normal(0, np.sqrt(sigma2[t]))
        return y

    # =========================================================================
    # 8. Multiple Correlated Series
    # =========================================================================

    def perfectly_correlated(self, n: int = 200) -> tuple[np.ndarray, np.ndarray]:
        """Perfectly correlated streams."""
        x = self.random_walk(n, sigma=1.0)
        return x, x.copy()

    def contemporaneous_relationship(
        self, n: int = 200, beta: float = 0.8, sigma: float = 0.5
    ) -> tuple[np.ndarray, np.ndarray]:
        """Contemporaneous relationship: y2_t = beta*y1_t + noise."""
        y1 = self.random_walk(n, sigma=1.0)
        y2 = beta * y1 + self.rng.normal(0, sigma, n)
        return y1, y2

    def lead_lag(
        self, n: int = 300, lag: int = 1, sigma: float = 0.5
    ) -> tuple[np.ndarray, np.ndarray]:
        """Lead-lag relationship: y2_t = y1_{t-k} + noise."""
        leader = self.random_walk(n + lag, sigma=1.0)
        follower = leader[:-lag] + self.rng.normal(0, sigma, n)
        return leader[lag:], follower

    def cointegrated(
        self, n: int = 300, beta: float = 1.0, spread_phi: float = 0.7, sigma: float = 0.5
    ) -> tuple[np.ndarray, np.ndarray]:
        """Cointegrated pair: both I(1) but spread is I(0)."""
        # Common stochastic trend
        trend = self.random_walk(n, sigma=1.0)

        # Mean-reverting spread
        spread = np.zeros(n)
        for t in range(1, n):
            spread[t] = spread_phi * spread[t - 1] + self.rng.normal(0, sigma)

        y1 = trend + self.rng.normal(0, 0.1, n)
        y2 = beta * trend + spread + self.rng.normal(0, 0.1, n)
        return y1, y2

    # =========================================================================
    # 10. Adversarial and Edge Cases
    # =========================================================================

    def impulse(self, n: int = 100, spike_time: int = 50, spike_value: float = 10.0) -> np.ndarray:
        """Impulse: single spike."""
        y = np.zeros(n)
        y[spike_time] = spike_value
        return y

    def step_function(
        self, n: int = 300, levels: list[float] | None = None, change_times: list[int] | None = None
    ) -> np.ndarray:
        """Step function: piecewise constant."""
        if levels is None:
            levels = [0.0, 5.0, 2.0, 8.0]
        if change_times is None:
            change_times = [0, 75, 150, 225]

        y = np.zeros(n)
        for i, (start, level) in enumerate(zip(change_times, levels)):
            end = change_times[i + 1] if i + 1 < len(change_times) else n
            y[start:end] = level
        return y

    def contaminated(
        self,
        n: int = 300,
        phi: float = 0.8,
        sigma: float = 0.5,
        contamination_prob: float = 0.05,
        contamination_scale: float = 10.0,
    ) -> np.ndarray:
        """Contaminated data: true process + outliers."""
        y = self.ar1_mean_reverting(n, phi, sigma)
        for t in range(n):
            if self.rng.random() < contamination_prob:
                y[t] += self.rng.choice([-1, 1]) * contamination_scale
        return y

    def very_short(self, n: int = 30) -> np.ndarray:
        """Very short series."""
        return self.ar1_mean_reverting(n, phi=0.8, sigma=0.5)

    def very_long(self, n: int = 5000) -> np.ndarray:
        """Very long series."""
        return self.ar1_mean_reverting(n, phi=0.8, sigma=0.5)


def run_test(
    signal: np.ndarray,
    signal_name: str,
    signal_category: str,
    expected_dominant: str,
    config: AEGISConfig | None = None,
    warmup: int = 50,
    notes: str = "",
) -> SignalMetrics:
    """Run a single test and collect metrics."""
    config = config or AEGISConfig()
    aegis = AEGIS(config=config)
    aegis.add_stream("test")

    n = len(signal)
    errors: list[float] = []
    in_interval = 0
    interval_widths: list[float] = []

    start_time = time.time()

    for t, y in enumerate(signal):
        if t > warmup:
            pred = aegis.predict("test", horizon=1)
            errors.append(y - pred.mean)

            if pred.interval_lower is not None and pred.interval_upper is not None:
                width = pred.interval_upper - pred.interval_lower
                interval_widths.append(width)
                if pred.interval_lower <= y <= pred.interval_upper:
                    in_interval += 1

        aegis.observe("test", y)
        aegis.end_period()

    runtime = time.time() - start_time

    # Compute metrics
    errors_arr = np.array(errors)
    mae = float(np.mean(np.abs(errors_arr)))
    rmse = float(np.sqrt(np.mean(errors_arr**2)))

    coverage = in_interval / len(errors) if errors else None
    mean_width = float(np.mean(interval_widths)) if interval_widths else None

    # Get dominant group
    diag = aegis.get_diagnostics("test")
    group_weights = diag["group_weights"]

    if group_weights:
        dominant_group = max(group_weights, key=group_weights.get)
        dominant_weight = group_weights[dominant_group]
    else:
        dominant_group = "unknown"
        dominant_weight = 0.0

    return SignalMetrics(
        signal_name=signal_name,
        signal_category=signal_category,
        n_observations=n,
        mae=mae,
        rmse=rmse,
        coverage_95=coverage,
        mean_interval_width=mean_width,
        dominant_group=dominant_group,
        dominant_group_weight=dominant_weight,
        expected_dominant=expected_dominant,
        dominant_matches_expected=dominant_group == expected_dominant,
        runtime_seconds=runtime,
        notes=notes,
    )


def run_multistream_test(
    streams: dict[str, np.ndarray],
    signal_name: str,
    signal_category: str,
    expected_dominant: str,
    target_stream: str,
    config: AEGISConfig | None = None,
    warmup: int = 50,
    notes: str = "",
) -> SignalMetrics:
    """Run a multi-stream test and collect metrics."""
    config = config or AEGISConfig(cross_stream_lags=3)
    aegis = AEGIS(config=config)

    for name in streams:
        aegis.add_stream(name)

    n = len(next(iter(streams.values())))
    errors: list[float] = []
    in_interval = 0
    interval_widths: list[float] = []

    start_time = time.time()

    for t in range(n):
        if t > warmup:
            for name in streams:
                aegis.predict(name, horizon=1)

            pred = aegis.predict(target_stream, horizon=1)
            y = streams[target_stream][t]
            errors.append(y - pred.mean)

            if pred.interval_lower is not None and pred.interval_upper is not None:
                width = pred.interval_upper - pred.interval_lower
                interval_widths.append(width)
                if pred.interval_lower <= y <= pred.interval_upper:
                    in_interval += 1

        for name, signal in streams.items():
            aegis.observe(name, signal[t])
        aegis.end_period()

    runtime = time.time() - start_time

    errors_arr = np.array(errors)
    mae = float(np.mean(np.abs(errors_arr)))
    rmse = float(np.sqrt(np.mean(errors_arr**2)))

    coverage = in_interval / len(errors) if errors else None
    mean_width = float(np.mean(interval_widths)) if interval_widths else None

    diag = aegis.get_diagnostics(target_stream)
    group_weights = diag["group_weights"]

    if group_weights:
        dominant_group = max(group_weights, key=group_weights.get)
        dominant_weight = group_weights[dominant_group]
    else:
        dominant_group = "unknown"
        dominant_weight = 0.0

    return SignalMetrics(
        signal_name=signal_name,
        signal_category=signal_category,
        n_observations=n,
        mae=mae,
        rmse=rmse,
        coverage_95=coverage,
        mean_interval_width=mean_width,
        dominant_group=dominant_group,
        dominant_group_weight=dominant_weight,
        expected_dominant=expected_dominant,
        dominant_matches_expected=dominant_group == expected_dominant,
        runtime_seconds=runtime,
        notes=notes,
    )


def run_all_tests() -> list[SignalMetrics]:
    """Run all signal taxonomy tests."""
    gen = SignalGenerators(seed=42)
    results: list[SignalMetrics] = []

    # =========================================================================
    # 2. Deterministic Signals
    # =========================================================================
    print("Testing: Deterministic Signals...")

    results.append(
        run_test(
            gen.constant(200, 5.0),
            "Constant Value",
            "Deterministic",
            "persistence",
            notes="LocalLevel should converge to constant",
        )
    )

    results.append(
        run_test(
            gen.linear_trend(200, 0.5, 0.0),
            "Linear Trend",
            "Deterministic",
            "trend",
            notes="LocalTrend or DampedTrend should dominate",
        )
    )

    results.append(
        run_test(
            gen.sinusoidal(200, 16, 2.0),
            "Sinusoidal",
            "Deterministic",
            "periodic",
            config=AEGISConfig(oscillator_periods=[16]),
            notes="OscillatorBank should dominate with matching period",
        )
    )

    results.append(
        run_test(
            gen.square_wave(200, 7, 2.0),
            "Square Wave",
            "Deterministic",
            "periodic",
            config=AEGISConfig(seasonal_periods=[7]),
            notes="SeasonalDummy should capture sharp transitions",
        )
    )

    results.append(
        run_test(
            gen.polynomial_trend(200, 0.001, 0.1, 0.0),
            "Polynomial Trend",
            "Deterministic",
            "trend",
            notes="LocalTrend tracks slope but underestimates curvature",
        )
    )

    # =========================================================================
    # 3. Simple Stochastic Processes
    # =========================================================================
    print("Testing: Simple Stochastic Processes...")

    results.append(
        run_test(
            gen.white_noise(200, 1.0),
            "White Noise",
            "Stochastic",
            "persistence",
            notes="RandomWalk should dominate, predicting 0",
        )
    )

    results.append(
        run_test(
            gen.random_walk(200, 1.0),
            "Random Walk",
            "Stochastic",
            "persistence",
            notes="RandomWalk is optimal for this signal",
        )
    )

    results.append(
        run_test(
            gen.ar1_mean_reverting(300, 0.8, 0.5),
            "AR(1) phi=0.8",
            "Stochastic",
            "reversion",
            notes="MeanReversion should dominate",
        )
    )

    results.append(
        run_test(
            gen.ar1_near_unit_root(500, 0.99, 0.5),
            "AR(1) phi=0.99",
            "Stochastic",
            "persistence",
            notes="Near unit root - hard to distinguish from RW at short scales",
        )
    )

    results.append(
        run_test(
            gen.ma1(200, 0.6, 1.0),
            "MA(1)",
            "Stochastic",
            "dynamic",
            notes="MA1 model should capture structure",
        )
    )

    results.append(
        run_test(
            gen.arma11(300, 0.7, 0.3, 1.0),
            "ARMA(1,1)",
            "Stochastic",
            "dynamic",
            notes="Combination of AR and MA models",
        )
    )

    results.append(
        run_test(
            gen.ornstein_uhlenbeck(300, 0.1, 0.0, 0.5),
            "Ornstein-Uhlenbeck",
            "Stochastic",
            "reversion",
            notes="Discretized OU is AR(1) toward mean",
        )
    )

    # =========================================================================
    # 4. Composite Signals
    # =========================================================================
    print("Testing: Composite Signals...")

    results.append(
        run_test(
            gen.trend_plus_noise(200, 0.3, 1.0),
            "Trend + Noise",
            "Composite",
            "trend",
            notes="LocalTrend should dominate",
        )
    )

    results.append(
        run_test(
            gen.sine_plus_noise(200, 16, 2.0, 0.5),
            "Sine + Noise",
            "Composite",
            "periodic",
            config=AEGISConfig(oscillator_periods=[16]),
            notes="OscillatorBank captures periodic component",
        )
    )

    results.append(
        run_test(
            gen.trend_seasonality_noise(300, 0.1, 7, 2.0, 0.5),
            "Trend + Seasonality + Noise",
            "Composite",
            "trend",
            config=AEGISConfig(seasonal_periods=[7]),
            notes="Mix of LocalTrend and SeasonalDummy",
        )
    )

    results.append(
        run_test(
            gen.mean_reversion_plus_oscillation(300, 0.7, 16, 1.0, 0.3),
            "Mean-Reversion + Oscillation",
            "Composite",
            "periodic",
            config=AEGISConfig(oscillator_periods=[16]),
            notes="Split weight between reversion and periodic",
        )
    )

    # =========================================================================
    # 5. Non-Stationary and Regime-Changing
    # =========================================================================
    print("Testing: Non-Stationary and Regime-Changing...")

    results.append(
        run_test(
            gen.random_walk_with_drift(200, 0.1, 1.0),
            "Random Walk with Drift",
            "Non-Stationary",
            "trend",
            notes="LocalTrend captures drift",
        )
    )

    results.append(
        run_test(
            gen.variance_switching(300, 0.5, 3.0, 0.02),
            "Variance Switching",
            "Non-Stationary",
            "persistence",
            notes="VolatilityTracker should adapt",
        )
    )

    results.append(
        run_test(
            gen.mean_switching(300, 0.0, 5.0, 1.0, 0.02),
            "Mean Switching",
            "Non-Stationary",
            "persistence",
            config=AEGISConfig(break_threshold=2.0),
            notes="Break detection should trigger on large shifts",
        )
    )

    results.append(
        run_test(
            gen.threshold_ar(300, 0.9, -0.5, 0.0, 0.5),
            "Threshold AR",
            "Non-Stationary",
            "reversion",
            notes="ThresholdAR should learn regimes",
        )
    )

    results.append(
        run_test(
            gen.structural_break(300, 150, 0.0, 5.0, 1.0),
            "Structural Break",
            "Non-Stationary",
            "persistence",
            config=AEGISConfig(break_threshold=2.0),
            notes="CUSUM should detect break",
        )
    )

    results.append(
        run_test(
            gen.gradual_drift(500, 0.01, 1.0),
            "Gradual Drift",
            "Non-Stationary",
            "trend",
            notes="Exponential forgetting tracks drift",
        )
    )

    # =========================================================================
    # 6. Heavy-Tailed Signals
    # =========================================================================
    print("Testing: Heavy-Tailed Signals...")

    results.append(
        run_test(
            gen.student_t_innovations(300, 4.0, 0.7),
            "Student-t (df=4)",
            "Heavy-Tailed",
            "reversion",
            notes="QuantileTracker should calibrate intervals",
        )
    )

    results.append(
        run_test(
            gen.student_t_innovations(300, 3.0, 0.7),
            "Student-t (df=3)",
            "Heavy-Tailed",
            "reversion",
            notes="Heavier tails, harder calibration",
        )
    )

    results.append(
        run_test(
            gen.occasional_jumps(300, 0.5, 0.02, 5.0),
            "Occasional Jumps",
            "Heavy-Tailed",
            "persistence",
            notes="JumpDiffusion should provide jump risk variance",
        )
    )

    results.append(
        run_test(
            gen.power_law_tails(300, 2.5),
            "Power-Law Tails (alpha=2.5)",
            "Heavy-Tailed",
            "persistence",
            notes="Finite variance, trackable",
        )
    )

    # =========================================================================
    # 7. Multi-Scale Structure
    # =========================================================================
    print("Testing: Multi-Scale Structure...")

    results.append(
        run_test(
            gen.fractional_brownian_persistent(500, 0.7),
            "fBM Persistent (H=0.7)",
            "Multi-Scale",
            "persistence",
            notes="Long scales should detect persistence",
        )
    )

    results.append(
        run_test(
            gen.fractional_brownian_antipersistent(500, 0.3),
            "fBM Antipersistent (H=0.3)",
            "Multi-Scale",
            "persistence",
            notes="Long scales should detect mean-reversion",
        )
    )

    results.append(
        run_test(
            gen.multi_timescale_mr(500, 0.5, 0.98, 0.5),
            "Multi-Timescale Mean-Reversion",
            "Multi-Scale",
            "reversion",
            notes="Different scales capture different components",
        )
    )

    results.append(
        run_test(
            gen.trend_momentum_reversion(500, 0.05, 0.3, 0.98, 0.5),
            "Trend + Momentum + Reversion",
            "Multi-Scale",
            "trend",
            notes="Multi-scale architecture advantage",
        )
    )

    results.append(
        run_test(
            gen.garch_like(500, 0.1, 0.85, 0.05),
            "GARCH-like Volatility",
            "Multi-Scale",
            "persistence",
            notes="VolatilityTracker captures clustering",
        )
    )

    # =========================================================================
    # 8. Multiple Correlated Series
    # =========================================================================
    print("Testing: Multiple Correlated Series...")

    y1, y2 = gen.perfectly_correlated(200)
    results.append(
        run_multistream_test(
            {"stream1": y1, "stream2": y2},
            "Perfectly Correlated",
            "Multi-Stream",
            "persistence",
            "stream2",
            notes="Cross-stream should identify common factor",
        )
    )

    y1, y2 = gen.contemporaneous_relationship(200, 0.8, 0.5)
    results.append(
        run_multistream_test(
            {"leader": y1, "follower": y2},
            "Contemporaneous Relationship",
            "Multi-Stream",
            "persistence",
            "follower",
            config=AEGISConfig(include_lag_zero=True, cross_stream_lags=3),
            notes="Lag-0 regression captures beta",
        )
    )

    y1, y2 = gen.lead_lag(300, 1, 0.5)
    results.append(
        run_multistream_test(
            {"leader": y1, "follower": y2},
            "Lead-Lag",
            "Multi-Stream",
            "persistence",
            "follower",
            config=AEGISConfig(cross_stream_lags=3),
            notes="Cross-stream regression learns lag",
        )
    )

    y1, y2 = gen.cointegrated(300, 1.0, 0.7, 0.5)
    results.append(
        run_multistream_test(
            {"y1": y1, "y2": y2},
            "Cointegrated Pair",
            "Multi-Stream",
            "persistence",
            "y2",
            config=AEGISConfig(cross_stream_lags=3),
            notes="Cross-stream captures error correction",
        )
    )

    # =========================================================================
    # 10. Adversarial and Edge Cases
    # =========================================================================
    print("Testing: Adversarial and Edge Cases...")

    results.append(
        run_test(
            gen.impulse(100, 50, 10.0),
            "Impulse",
            "Edge Case",
            "persistence",
            warmup=20,
            notes="LocalLevel tracks then decays",
        )
    )

    results.append(
        run_test(
            gen.step_function(300),
            "Step Function",
            "Edge Case",
            "persistence",
            config=AEGISConfig(break_threshold=2.0),
            notes="Break detection may trigger at jumps",
        )
    )

    results.append(
        run_test(
            gen.contaminated(300, 0.8, 0.5, 0.05, 10.0),
            "Contaminated Data",
            "Edge Case",
            "reversion",
            notes="JumpDiffusion absorbs some outliers",
        )
    )

    results.append(
        run_test(
            gen.very_short(30),
            "Very Short Series (n=30)",
            "Edge Case",
            "persistence",
            warmup=10,
            notes="Incomplete convergence expected",
        )
    )

    results.append(
        run_test(
            gen.very_long(2000),
            "Very Long Series (n=2000)",
            "Edge Case",
            "reversion",
            warmup=100,
            notes="Parameters should fully converge",
        )
    )

    return results


def generate_report(results: list[SignalMetrics]) -> str:
    """Generate a markdown report from test results."""
    lines = [
        "# AEGIS Signal Taxonomy Acceptance Test Report",
        "",
        f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Total Tests:** {len(results)}",
        "",
        "---",
        "",
        "## Summary Statistics",
        "",
    ]

    # Overall statistics
    mae_values = [r.mae for r in results]
    rmse_values = [r.rmse for r in results]
    coverage_values = [r.coverage_95 for r in results if r.coverage_95 is not None]
    matches = sum(1 for r in results if r.dominant_matches_expected)
    total_runtime = sum(r.runtime_seconds for r in results)

    lines.extend(
        [
            "| Metric | Value |",
            "|--------|-------|",
            f"| Mean MAE | {np.mean(mae_values):.4f} |",
            f"| Mean RMSE | {np.mean(rmse_values):.4f} |",
            f"| Mean Coverage | {np.mean(coverage_values):.2%} |",
            f"| Dominant Group Match Rate | {matches}/{len(results)} "
            f"({100 * matches / len(results):.1f}%) |",
            f"| Total Runtime | {total_runtime:.2f}s |",
            "",
            "---",
            "",
        ]
    )

    # Results by category
    categories = sorted(set(r.signal_category for r in results))

    for category in categories:
        cat_results = [r for r in results if r.signal_category == category]

        lines.extend(
            [
                f"## {category}",
                "",
                "| Signal | MAE | RMSE | Coverage | Dominant | Expected | Match | Time |",
                "|--------|-----|------|----------|----------|----------|-------|------|",
            ]
        )

        for r in cat_results:
            coverage_str = f"{r.coverage_95:.1%}" if r.coverage_95 else "N/A"
            match_str = "Yes" if r.dominant_matches_expected else "**No**"
            lines.append(
                f"| {r.signal_name} | {r.mae:.4f} | {r.rmse:.4f} | {coverage_str} | "
                f"{r.dominant_group} ({r.dominant_group_weight:.1%}) | {r.expected_dominant} | "
                f"{match_str} | {r.runtime_seconds:.2f}s |"
            )

        # Category summary
        cat_mae = np.mean([r.mae for r in cat_results])
        cat_rmse = np.mean([r.rmse for r in cat_results])
        cat_matches = sum(1 for r in cat_results if r.dominant_matches_expected)
        cat_coverages = [r.coverage_95 for r in cat_results if r.coverage_95 is not None]
        cat_coverage = np.mean(cat_coverages) if cat_coverages else 0

        lines.extend(
            [
                "",
                f"**Category Summary:** MAE={cat_mae:.4f}, RMSE={cat_rmse:.4f}, "
                f"Coverage={cat_coverage:.1%}, Matches={cat_matches}/{len(cat_results)}",
                "",
                "---",
                "",
            ]
        )

    # Detailed notes
    lines.extend(
        [
            "## Detailed Notes",
            "",
            "| Signal | Notes |",
            "|--------|-------|",
        ]
    )

    for r in results:
        if r.notes:
            lines.append(f"| {r.signal_name} | {r.notes} |")

    lines.extend(
        [
            "",
            "---",
            "",
            "## Coverage Analysis",
            "",
            "The target coverage is 95%. Tests with coverage significantly below this "
            "may indicate calibration issues.",
            "",
            "| Signal | Coverage | Status |",
            "|--------|----------|--------|",
        ]
    )

    for r in results:
        if r.coverage_95 is not None:
            if r.coverage_95 >= 0.90:
                status = "Good"
            elif r.coverage_95 >= 0.80:
                status = "Acceptable"
            else:
                status = "**Needs Attention**"
            lines.append(f"| {r.signal_name} | {r.coverage_95:.1%} | {status} |")

    lines.extend(
        [
            "",
            "---",
            "",
            "## Performance Rating Summary",
            "",
            "Based on Appendix D rating scale:",
            "- **Excellent**: MAE < 0.5 and Coverage > 90%",
            "- **Good**: MAE < 1.0 and Coverage > 80%",
            "- **Moderate**: MAE < 2.0 and Coverage > 70%",
            "- **Poor**: Otherwise",
            "",
            "| Rating | Count | Signals |",
            "|--------|-------|---------|",
        ]
    )

    ratings = {"Excellent": [], "Good": [], "Moderate": [], "Poor": []}
    for r in results:
        coverage = r.coverage_95 or 0.0
        if r.mae < 0.5 and coverage > 0.90:
            ratings["Excellent"].append(r.signal_name)
        elif r.mae < 1.0 and coverage > 0.80:
            ratings["Good"].append(r.signal_name)
        elif r.mae < 2.0 and coverage > 0.70:
            ratings["Moderate"].append(r.signal_name)
        else:
            ratings["Poor"].append(r.signal_name)

    for rating, signals in ratings.items():
        signals_str = ", ".join(signals[:5])
        if len(signals) > 5:
            signals_str += f" (+{len(signals) - 5} more)"
        lines.append(f"| {rating} | {len(signals)} | {signals_str} |")

    lines.extend(
        [
            "",
            "---",
            "",
            "*Report generated by AEGIS acceptance test suite*",
        ]
    )

    return "\n".join(lines)


def save_results(results: list[SignalMetrics], output_dir: Path) -> None:
    """Save results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    json_data = [dataclasses.asdict(r) for r in results]
    with open(output_dir / "acceptance_results.json", "w") as f:
        json.dump(json_data, f, indent=2)

    # Save report
    report = generate_report(results)
    with open(output_dir / "acceptance_report.md", "w") as f:
        f.write(report)

    print(f"\nResults saved to {output_dir}/")


class TestSignalTaxonomyAcceptance:
    """Pytest wrapper for acceptance tests."""

    def test_deterministic_signals(self) -> None:
        """Test deterministic signals category."""
        gen = SignalGenerators(seed=42)

        result = run_test(gen.constant(200, 5.0), "Constant", "Deterministic", "persistence")
        assert result.mae < 1.0

        result = run_test(gen.linear_trend(200, 0.5, 0.0), "Linear Trend", "Deterministic", "trend")
        assert result.mae < 2.0

    def test_stochastic_signals(self) -> None:
        """Test stochastic signals category."""
        gen = SignalGenerators(seed=42)

        result = run_test(gen.white_noise(200, 1.0), "White Noise", "Stochastic", "persistence")
        assert result.coverage_95 is not None
        assert result.coverage_95 > 0.7

        result = run_test(gen.ar1_mean_reverting(300, 0.8, 0.5), "AR(1)", "Stochastic", "reversion")
        assert result.mae < 1.0

    def test_regime_changing(self) -> None:
        """Test regime-changing signals."""
        gen = SignalGenerators(seed=42)

        result = run_test(
            gen.structural_break(300, 150, 0.0, 5.0, 1.0),
            "Structural Break",
            "Non-Stationary",
            "persistence",
            config=AEGISConfig(break_threshold=2.0),
        )
        assert result.mae < 5.0

    def test_multi_stream(self) -> None:
        """Test multi-stream signals."""
        gen = SignalGenerators(seed=42)

        y1, y2 = gen.lead_lag(300, 1, 0.5)
        result = run_multistream_test(
            {"leader": y1, "follower": y2},
            "Lead-Lag",
            "Multi-Stream",
            "persistence",
            "follower",
            config=AEGISConfig(cross_stream_lags=3),
        )
        assert result.mae < 5.0  # Allow higher MAE for multi-stream

    def test_edge_cases(self) -> None:
        """Test edge cases."""
        gen = SignalGenerators(seed=42)

        result = run_test(gen.very_short(30), "Very Short", "Edge Case", "persistence", warmup=10)
        # Just verify it runs without error
        assert result.n_observations == 30


if __name__ == "__main__":
    print("=" * 70)
    print("AEGIS Signal Taxonomy Acceptance Tests")
    print("=" * 70)
    print()

    results = run_all_tests()

    print()
    print("=" * 70)
    print("Generating Report...")
    print("=" * 70)

    output_dir = Path(__file__).parent.parent.parent / "reports"
    save_results(results, output_dir)

    report = generate_report(results)
    print()
    print(report)
