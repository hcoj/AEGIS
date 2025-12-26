"""Acceptance tests for AEGIS signal taxonomy.

Comprehensive tests covering all signal types from Appendix D,
measuring performance metrics across multiple forecast horizons.
"""

import dataclasses
import json
import time
from pathlib import Path

import numpy as np

from aegis.config import AEGISConfig
from aegis.system import AEGIS

# Horizons to test - powers of 2 up to 1024
HORIZONS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

# Minimum signal length to support all horizons
MIN_SIGNAL_LENGTH = 3000  # warmup + test points + max horizon


@dataclasses.dataclass
class HorizonMetrics:
    """Metrics for a specific horizon."""

    horizon: int
    mae: float
    rmse: float
    coverage_95: float | None
    mean_interval_width: float | None
    n_predictions: int


@dataclasses.dataclass
class SignalMetrics:
    """Metrics collected for a signal test across all horizons."""

    signal_name: str
    signal_category: str
    n_observations: int
    horizon_metrics: list[HorizonMetrics]
    dominant_group: str
    dominant_group_weight: float
    expected_dominant: str
    dominant_matches_expected: bool
    runtime_seconds: float
    notes: str = ""

    @property
    def mae_h1(self) -> float:
        """MAE at horizon 1."""
        return self.horizon_metrics[0].mae if self.horizon_metrics else 0.0

    @property
    def coverage_h1(self) -> float | None:
        """Coverage at horizon 1."""
        return self.horizon_metrics[0].coverage_95 if self.horizon_metrics else None


class SignalGenerators:
    """Signal generators for all taxonomy types."""

    def __init__(self, seed: int = 42) -> None:
        """Initialize with random seed."""
        self.rng = np.random.default_rng(seed)

    # =========================================================================
    # 2. Deterministic Signals
    # =========================================================================

    def constant(self, n: int = MIN_SIGNAL_LENGTH, c: float = 5.0) -> np.ndarray:
        """Constant signal: y_t = c."""
        return np.full(n, c)

    def linear_trend(
        self, n: int = MIN_SIGNAL_LENGTH, slope: float = 0.1, intercept: float = 0.0
    ) -> np.ndarray:
        """Linear trend: y_t = a*t + b."""
        t = np.arange(n)
        return slope * t + intercept

    def sinusoidal(
        self,
        n: int = MIN_SIGNAL_LENGTH,
        period: int = 64,
        amplitude: float = 2.0,
        phase: float = 0.0,
    ) -> np.ndarray:
        """Sinusoidal: y_t = A*sin(omega*t + phi)."""
        t = np.arange(n)
        omega = 2 * np.pi / period
        return amplitude * np.sin(omega * t + phase)

    def square_wave(
        self, n: int = MIN_SIGNAL_LENGTH, period: int = 64, amplitude: float = 1.0
    ) -> np.ndarray:
        """Square wave / sharp seasonal pattern."""
        t = np.arange(n)
        phase = t % period
        return np.where(phase < period // 2, amplitude, -amplitude)

    def polynomial_trend(
        self, n: int = MIN_SIGNAL_LENGTH, a: float = 0.0001, b: float = 0.05, c: float = 0.0
    ) -> np.ndarray:
        """Polynomial trend: y_t = a*t^2 + b*t + c."""
        t = np.arange(n)
        return a * t**2 + b * t + c

    # =========================================================================
    # 3. Simple Stochastic Processes
    # =========================================================================

    def white_noise(self, n: int = MIN_SIGNAL_LENGTH, sigma: float = 1.0) -> np.ndarray:
        """White noise: y_t ~ N(0, sigma^2) i.i.d."""
        return self.rng.normal(0, sigma, n)

    def random_walk(
        self, n: int = MIN_SIGNAL_LENGTH, sigma: float = 1.0, y0: float = 0.0
    ) -> np.ndarray:
        """Random walk: y_t = y_{t-1} + epsilon_t."""
        innovations = self.rng.normal(0, sigma, n)
        y = np.zeros(n)
        y[0] = y0 + innovations[0]
        for t in range(1, n):
            y[t] = y[t - 1] + innovations[t]
        return y

    def ar1_mean_reverting(
        self, n: int = MIN_SIGNAL_LENGTH, phi: float = 0.8, sigma: float = 0.5, mu: float = 0.0
    ) -> np.ndarray:
        """AR(1) mean-reverting: y_t = phi*y_{t-1} + epsilon_t."""
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = phi * y[t - 1] + self.rng.normal(0, sigma)
        return y + mu

    def ar1_near_unit_root(
        self, n: int = MIN_SIGNAL_LENGTH, phi: float = 0.99, sigma: float = 0.5
    ) -> np.ndarray:
        """AR(1) near unit root: phi close to 1."""
        return self.ar1_mean_reverting(n, phi, sigma)

    def ma1(self, n: int = MIN_SIGNAL_LENGTH, theta: float = 0.6, sigma: float = 1.0) -> np.ndarray:
        """MA(1): y_t = epsilon_t + theta*epsilon_{t-1}."""
        eps = self.rng.normal(0, sigma, n + 1)
        y = np.zeros(n)
        for t in range(n):
            y[t] = eps[t + 1] + theta * eps[t]
        return y

    def arma11(
        self, n: int = MIN_SIGNAL_LENGTH, phi: float = 0.7, theta: float = 0.3, sigma: float = 1.0
    ) -> np.ndarray:
        """ARMA(1,1): y_t = phi*y_{t-1} + epsilon_t + theta*epsilon_{t-1}."""
        eps = self.rng.normal(0, sigma, n + 1)
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = phi * y[t - 1] + eps[t + 1] + theta * eps[t]
        return y

    def ornstein_uhlenbeck(
        self, n: int = MIN_SIGNAL_LENGTH, theta: float = 0.1, mu: float = 0.0, sigma: float = 0.5
    ) -> np.ndarray:
        """Ornstein-Uhlenbeck: discretized mean-reverting diffusion."""
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = y[t - 1] + theta * (mu - y[t - 1]) + self.rng.normal(0, sigma)
        return y

    # =========================================================================
    # 4. Composite Signals
    # =========================================================================

    def trend_plus_noise(
        self, n: int = MIN_SIGNAL_LENGTH, slope: float = 0.05, sigma: float = 1.0
    ) -> np.ndarray:
        """Trend + noise: y_t = a*t + epsilon_t."""
        t = np.arange(n)
        return slope * t + self.rng.normal(0, sigma, n)

    def sine_plus_noise(
        self,
        n: int = MIN_SIGNAL_LENGTH,
        period: int = 64,
        amplitude: float = 2.0,
        sigma: float = 0.5,
    ) -> np.ndarray:
        """Sine + noise: y_t = A*sin(omega*t) + epsilon_t."""
        return self.sinusoidal(n, period, amplitude) + self.rng.normal(0, sigma, n)

    def trend_seasonality_noise(
        self,
        n: int = MIN_SIGNAL_LENGTH,
        slope: float = 0.02,
        period: int = 64,
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
        n: int = MIN_SIGNAL_LENGTH,
        phi: float = 0.7,
        period: int = 64,
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
        self, n: int = MIN_SIGNAL_LENGTH, drift: float = 0.05, sigma: float = 1.0
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
        n: int = MIN_SIGNAL_LENGTH,
        sigma_low: float = 0.5,
        sigma_high: float = 3.0,
        switch_prob: float = 0.01,
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
        n: int = MIN_SIGNAL_LENGTH,
        mu_a: float = 0.0,
        mu_b: float = 5.0,
        sigma: float = 1.0,
        switch_prob: float = 0.01,
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
        n: int = MIN_SIGNAL_LENGTH,
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
        n: int = MIN_SIGNAL_LENGTH,
        break_time: int | None = None,
        mu_before: float = 0.0,
        mu_after: float = 5.0,
        sigma: float = 1.0,
    ) -> np.ndarray:
        """Structural break: parameter change at known time."""
        if break_time is None:
            break_time = n // 2
        y = np.zeros(n)
        for t in range(n):
            mu = mu_after if t >= break_time else mu_before
            y[t] = self.rng.normal(mu, sigma)
        return y

    def gradual_drift(
        self, n: int = MIN_SIGNAL_LENGTH, drift_rate: float = 0.005, sigma: float = 1.0
    ) -> np.ndarray:
        """Gradual drift: slowly changing mean."""
        t = np.arange(n)
        drift = drift_rate * t
        return drift + self.rng.normal(0, sigma, n)

    # =========================================================================
    # 6. Heavy-Tailed Signals
    # =========================================================================

    def student_t_innovations(
        self, n: int = MIN_SIGNAL_LENGTH, df: float = 4.0, phi: float = 0.7
    ) -> np.ndarray:
        """Student-t innovations: AR(1) with t-distributed errors."""
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = phi * y[t - 1] + self.rng.standard_t(df)
        return y

    def occasional_jumps(
        self,
        n: int = MIN_SIGNAL_LENGTH,
        sigma: float = 0.5,
        jump_prob: float = 0.01,
        jump_size: float = 5.0,
    ) -> np.ndarray:
        """Random walk with occasional jumps."""
        y = np.zeros(n)
        for t in range(1, n):
            innovation = self.rng.normal(0, sigma)
            if self.rng.random() < jump_prob:
                innovation += self.rng.choice([-1, 1]) * jump_size
            y[t] = y[t - 1] + innovation
        return y

    def power_law_tails(self, n: int = MIN_SIGNAL_LENGTH, alpha: float = 2.5) -> np.ndarray:
        """Power-law tailed innovations (Pareto-like)."""
        y = np.zeros(n)
        for t in range(1, n):
            u = self.rng.random()
            sign = self.rng.choice([-1, 1])
            innovation = sign * ((1 - u) ** (-1 / alpha) - 1)
            y[t] = y[t - 1] + np.clip(innovation, -10, 10)
        return y

    # =========================================================================
    # 7. Multi-Scale Structure
    # =========================================================================

    def fractional_brownian_persistent(
        self, n: int = MIN_SIGNAL_LENGTH, h: float = 0.7
    ) -> np.ndarray:
        """Fractional Brownian motion H > 0.5 (persistent)."""
        d = h - 0.5
        weights = np.array([(k + 1) ** d - max(k, 0.001) ** d for k in range(min(n, 100))])
        weights = np.abs(weights)
        weights = weights / (np.sum(weights) + 1e-10)

        noise = self.rng.normal(0, 1, n + len(weights))
        increments = np.convolve(noise, weights, mode="valid")[:n]
        return np.cumsum(increments)

    def fractional_brownian_antipersistent(
        self, n: int = MIN_SIGNAL_LENGTH, h: float = 0.3
    ) -> np.ndarray:
        """Fractional Brownian motion H < 0.5 (antipersistent/mean-reverting)."""
        return self.fractional_brownian_persistent(n, h)

    def multi_timescale_mr(
        self,
        n: int = MIN_SIGNAL_LENGTH,
        phi_fast: float = 0.5,
        phi_slow: float = 0.98,
        sigma: float = 0.5,
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
        n: int = MIN_SIGNAL_LENGTH,
        trend_strength: float = 0.01,
        momentum_phi: float = 0.3,
        reversion_phi: float = 0.98,
        sigma: float = 0.5,
    ) -> np.ndarray:
        """Trend + momentum + reversion at different timescales."""
        t = np.arange(n)
        trend = trend_strength * t

        momentum = np.zeros(n)
        for i in range(2, n):
            momentum[i] = momentum_phi * momentum[i - 1] + self.rng.normal(0, sigma)

        reversion = np.zeros(n)
        for i in range(1, n):
            reversion[i] = reversion_phi * reversion[i - 1] + self.rng.normal(0, sigma * 0.2)

        return trend + momentum + reversion

    def garch_like(
        self,
        n: int = MIN_SIGNAL_LENGTH,
        alpha: float = 0.1,
        beta: float = 0.85,
        omega: float = 0.05,
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

    def perfectly_correlated(self, n: int = MIN_SIGNAL_LENGTH) -> tuple[np.ndarray, np.ndarray]:
        """Perfectly correlated streams."""
        x = self.random_walk(n, sigma=1.0)
        return x, x.copy()

    def contemporaneous_relationship(
        self, n: int = MIN_SIGNAL_LENGTH, beta: float = 0.8, sigma: float = 0.5
    ) -> tuple[np.ndarray, np.ndarray]:
        """Contemporaneous relationship: y2_t = beta*y1_t + noise."""
        y1 = self.random_walk(n, sigma=1.0)
        y2 = beta * y1 + self.rng.normal(0, sigma, n)
        return y1, y2

    def lead_lag(
        self, n: int = MIN_SIGNAL_LENGTH, lag: int = 1, sigma: float = 0.5
    ) -> tuple[np.ndarray, np.ndarray]:
        """Lead-lag relationship: y2_t = y1_{t-k} + noise."""
        leader = self.random_walk(n + lag, sigma=1.0)
        follower = leader[:-lag] + self.rng.normal(0, sigma, n)
        return leader[lag:], follower

    def cointegrated(
        self,
        n: int = MIN_SIGNAL_LENGTH,
        beta: float = 1.0,
        spread_phi: float = 0.7,
        sigma: float = 0.5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Cointegrated pair: both I(1) but spread is I(0)."""
        trend = self.random_walk(n, sigma=1.0)
        spread = np.zeros(n)
        for t in range(1, n):
            spread[t] = spread_phi * spread[t - 1] + self.rng.normal(0, sigma)

        y1 = trend + self.rng.normal(0, 0.1, n)
        y2 = beta * trend + spread + self.rng.normal(0, 0.1, n)
        return y1, y2

    # =========================================================================
    # 10. Adversarial and Edge Cases
    # =========================================================================

    def impulse(
        self, n: int = MIN_SIGNAL_LENGTH, spike_time: int | None = None, spike_value: float = 10.0
    ) -> np.ndarray:
        """Impulse: single spike."""
        if spike_time is None:
            spike_time = n // 4
        y = np.zeros(n)
        y[spike_time] = spike_value
        return y

    def step_function(
        self,
        n: int = MIN_SIGNAL_LENGTH,
        levels: list[float] | None = None,
        n_steps: int = 6,
    ) -> np.ndarray:
        """Step function: piecewise constant."""
        if levels is None:
            levels = [0.0, 5.0, 2.0, 8.0, 3.0, 6.0]
        step_size = n // n_steps
        change_times = [i * step_size for i in range(n_steps)]

        y = np.zeros(n)
        for i, (start, level) in enumerate(zip(change_times, levels)):
            end = change_times[i + 1] if i + 1 < len(change_times) else n
            y[start:end] = level
        return y

    def contaminated(
        self,
        n: int = MIN_SIGNAL_LENGTH,
        phi: float = 0.8,
        sigma: float = 0.5,
        contamination_prob: float = 0.02,
        contamination_scale: float = 10.0,
    ) -> np.ndarray:
        """Contaminated data: true process + outliers."""
        y = self.ar1_mean_reverting(n, phi, sigma)
        for t in range(n):
            if self.rng.random() < contamination_prob:
                y[t] += self.rng.choice([-1, 1]) * contamination_scale
        return y


def run_test(
    signal: np.ndarray,
    signal_name: str,
    signal_category: str,
    expected_dominant: str,
    config: AEGISConfig | None = None,
    warmup: int = 200,
    horizons: list[int] | None = None,
    notes: str = "",
) -> SignalMetrics:
    """Run a single test across multiple horizons and collect metrics."""
    if horizons is None:
        horizons = HORIZONS

    config = config or AEGISConfig()
    aegis = AEGIS(config=config)
    aegis.add_stream("test")

    n = len(signal)

    # Storage for errors per horizon
    errors: dict[int, list[float]] = {h: [] for h in horizons}
    in_interval: dict[int, int] = {h: 0 for h in horizons}
    interval_widths: dict[int, list[float]] = {h: [] for h in horizons}

    start_time = time.time()

    # Feed observations and collect predictions
    for t in range(n):
        # Make predictions at multiple horizons before observing
        if t > warmup:
            for h in horizons:
                if t + h < n:  # Only predict if we'll see the actual value
                    pred = aegis.predict("test", horizon=h)
                    actual = signal[t + h]
                    errors[h].append(actual - pred.mean)

                    if pred.interval_lower is not None and pred.interval_upper is not None:
                        width = pred.interval_upper - pred.interval_lower
                        interval_widths[h].append(width)
                        if pred.interval_lower <= actual <= pred.interval_upper:
                            in_interval[h] += 1

        aegis.observe("test", signal[t])
        aegis.end_period()

    runtime = time.time() - start_time

    # Compute metrics per horizon
    horizon_metrics: list[HorizonMetrics] = []
    for h in horizons:
        if errors[h]:
            errors_arr = np.array(errors[h])
            mae = float(np.mean(np.abs(errors_arr)))
            rmse = float(np.sqrt(np.mean(errors_arr**2)))
            coverage = in_interval[h] / len(errors[h]) if errors[h] else None
            mean_width = float(np.mean(interval_widths[h])) if interval_widths[h] else None
            n_preds = len(errors[h])
        else:
            mae, rmse, coverage, mean_width, n_preds = 0.0, 0.0, None, None, 0

        horizon_metrics.append(
            HorizonMetrics(
                horizon=h,
                mae=mae,
                rmse=rmse,
                coverage_95=coverage,
                mean_interval_width=mean_width,
                n_predictions=n_preds,
            )
        )

    # Get dominant group (based on final state)
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
        horizon_metrics=horizon_metrics,
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
    warmup: int = 200,
    horizons: list[int] | None = None,
    notes: str = "",
) -> SignalMetrics:
    """Run a multi-stream test across multiple horizons."""
    if horizons is None:
        horizons = HORIZONS

    config = config or AEGISConfig(cross_stream_lags=3)
    aegis = AEGIS(config=config)

    for name in streams:
        aegis.add_stream(name)

    n = len(next(iter(streams.values())))
    target_signal = streams[target_stream]

    errors: dict[int, list[float]] = {h: [] for h in horizons}
    in_interval: dict[int, int] = {h: 0 for h in horizons}
    interval_widths: dict[int, list[float]] = {h: [] for h in horizons}

    start_time = time.time()

    for t in range(n):
        if t > warmup:
            # Predict for all streams first
            for name in streams:
                aegis.predict(name, horizon=1)

            # Collect metrics for target stream at multiple horizons
            for h in horizons:
                if t + h < n:
                    pred = aegis.predict(target_stream, horizon=h)
                    actual = target_signal[t + h]
                    errors[h].append(actual - pred.mean)

                    if pred.interval_lower is not None and pred.interval_upper is not None:
                        width = pred.interval_upper - pred.interval_lower
                        interval_widths[h].append(width)
                        if pred.interval_lower <= actual <= pred.interval_upper:
                            in_interval[h] += 1

        for name, signal in streams.items():
            aegis.observe(name, signal[t])
        aegis.end_period()

    runtime = time.time() - start_time

    horizon_metrics: list[HorizonMetrics] = []
    for h in horizons:
        if errors[h]:
            errors_arr = np.array(errors[h])
            mae = float(np.mean(np.abs(errors_arr)))
            rmse = float(np.sqrt(np.mean(errors_arr**2)))
            coverage = in_interval[h] / len(errors[h]) if errors[h] else None
            mean_width = float(np.mean(interval_widths[h])) if interval_widths[h] else None
            n_preds = len(errors[h])
        else:
            mae, rmse, coverage, mean_width, n_preds = 0.0, 0.0, None, None, 0

        horizon_metrics.append(
            HorizonMetrics(
                horizon=h,
                mae=mae,
                rmse=rmse,
                coverage_95=coverage,
                mean_interval_width=mean_width,
                n_predictions=n_preds,
            )
        )

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
        horizon_metrics=horizon_metrics,
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
            gen.constant(),
            "Constant Value",
            "Deterministic",
            "persistence",
            notes="LocalLevel should converge to constant",
        )
    )

    results.append(
        run_test(
            gen.linear_trend(),
            "Linear Trend",
            "Deterministic",
            "trend",
            notes="LocalTrend or DampedTrend should dominate",
        )
    )

    results.append(
        run_test(
            gen.sinusoidal(period=64),
            "Sinusoidal",
            "Deterministic",
            "periodic",
            config=AEGISConfig(oscillator_periods=[64]),
            notes="OscillatorBank should dominate with matching period",
        )
    )

    results.append(
        run_test(
            gen.square_wave(period=64),
            "Square Wave",
            "Deterministic",
            "periodic",
            config=AEGISConfig(seasonal_periods=[64]),
            notes="SeasonalDummy should capture sharp transitions",
        )
    )

    results.append(
        run_test(
            gen.polynomial_trend(),
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
            gen.white_noise(),
            "White Noise",
            "Stochastic",
            "persistence",
            notes="RandomWalk should dominate, predicting 0",
        )
    )

    results.append(
        run_test(
            gen.random_walk(),
            "Random Walk",
            "Stochastic",
            "persistence",
            notes="RandomWalk is optimal for this signal",
        )
    )

    results.append(
        run_test(
            gen.ar1_mean_reverting(phi=0.8),
            "AR(1) phi=0.8",
            "Stochastic",
            "reversion",
            notes="MeanReversion should dominate",
        )
    )

    results.append(
        run_test(
            gen.ar1_near_unit_root(phi=0.99),
            "AR(1) phi=0.99",
            "Stochastic",
            "persistence",
            notes="Near unit root - hard to distinguish from RW at short scales",
        )
    )

    results.append(
        run_test(
            gen.ma1(),
            "MA(1)",
            "Stochastic",
            "dynamic",
            notes="MA1 model should capture structure",
        )
    )

    results.append(
        run_test(
            gen.arma11(),
            "ARMA(1,1)",
            "Stochastic",
            "dynamic",
            notes="Combination of AR and MA models",
        )
    )

    results.append(
        run_test(
            gen.ornstein_uhlenbeck(),
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
            gen.trend_plus_noise(),
            "Trend + Noise",
            "Composite",
            "trend",
            notes="LocalTrend should dominate",
        )
    )

    results.append(
        run_test(
            gen.sine_plus_noise(period=64),
            "Sine + Noise",
            "Composite",
            "periodic",
            config=AEGISConfig(oscillator_periods=[64]),
            notes="OscillatorBank captures periodic component",
        )
    )

    results.append(
        run_test(
            gen.trend_seasonality_noise(period=64),
            "Trend + Seasonality + Noise",
            "Composite",
            "trend",
            config=AEGISConfig(seasonal_periods=[64]),
            notes="Mix of LocalTrend and SeasonalDummy",
        )
    )

    results.append(
        run_test(
            gen.mean_reversion_plus_oscillation(period=64),
            "Mean-Reversion + Oscillation",
            "Composite",
            "periodic",
            config=AEGISConfig(oscillator_periods=[64]),
            notes="Split weight between reversion and periodic",
        )
    )

    # =========================================================================
    # 5. Non-Stationary and Regime-Changing
    # =========================================================================
    print("Testing: Non-Stationary and Regime-Changing...")

    results.append(
        run_test(
            gen.random_walk_with_drift(),
            "Random Walk with Drift",
            "Non-Stationary",
            "trend",
            notes="LocalTrend captures drift",
        )
    )

    results.append(
        run_test(
            gen.variance_switching(),
            "Variance Switching",
            "Non-Stationary",
            "persistence",
            notes="VolatilityTracker should adapt",
        )
    )

    results.append(
        run_test(
            gen.mean_switching(),
            "Mean Switching",
            "Non-Stationary",
            "persistence",
            config=AEGISConfig(break_threshold=2.0),
            notes="Break detection should trigger on large shifts",
        )
    )

    results.append(
        run_test(
            gen.threshold_ar(),
            "Threshold AR",
            "Non-Stationary",
            "reversion",
            notes="ThresholdAR should learn regimes",
        )
    )

    results.append(
        run_test(
            gen.structural_break(),
            "Structural Break",
            "Non-Stationary",
            "persistence",
            config=AEGISConfig(break_threshold=2.0),
            notes="CUSUM should detect break",
        )
    )

    results.append(
        run_test(
            gen.gradual_drift(),
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
            gen.student_t_innovations(df=4.0),
            "Student-t (df=4)",
            "Heavy-Tailed",
            "reversion",
            notes="QuantileTracker should calibrate intervals",
        )
    )

    results.append(
        run_test(
            gen.student_t_innovations(df=3.0),
            "Student-t (df=3)",
            "Heavy-Tailed",
            "reversion",
            notes="Heavier tails, harder calibration",
        )
    )

    results.append(
        run_test(
            gen.occasional_jumps(),
            "Occasional Jumps",
            "Heavy-Tailed",
            "persistence",
            notes="JumpDiffusion should provide jump risk variance",
        )
    )

    results.append(
        run_test(
            gen.power_law_tails(),
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
            gen.fractional_brownian_persistent(h=0.7),
            "fBM Persistent (H=0.7)",
            "Multi-Scale",
            "persistence",
            notes="Long scales should detect persistence",
        )
    )

    results.append(
        run_test(
            gen.fractional_brownian_antipersistent(h=0.3),
            "fBM Antipersistent (H=0.3)",
            "Multi-Scale",
            "persistence",
            notes="Long scales should detect mean-reversion",
        )
    )

    results.append(
        run_test(
            gen.multi_timescale_mr(),
            "Multi-Timescale Mean-Reversion",
            "Multi-Scale",
            "reversion",
            notes="Different scales capture different components",
        )
    )

    results.append(
        run_test(
            gen.trend_momentum_reversion(),
            "Trend + Momentum + Reversion",
            "Multi-Scale",
            "trend",
            notes="Multi-scale architecture advantage",
        )
    )

    results.append(
        run_test(
            gen.garch_like(),
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

    y1, y2 = gen.perfectly_correlated()
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

    y1, y2 = gen.contemporaneous_relationship()
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

    y1, y2 = gen.lead_lag()
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

    y1, y2 = gen.cointegrated()
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
            gen.impulse(),
            "Impulse",
            "Edge Case",
            "persistence",
            notes="LocalLevel tracks then decays",
        )
    )

    results.append(
        run_test(
            gen.step_function(),
            "Step Function",
            "Edge Case",
            "persistence",
            config=AEGISConfig(break_threshold=2.0),
            notes="Break detection may trigger at jumps",
        )
    )

    results.append(
        run_test(
            gen.contaminated(),
            "Contaminated Data",
            "Edge Case",
            "reversion",
            notes="JumpDiffusion absorbs some outliers",
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
        f"**Horizons Tested:** {', '.join(map(str, HORIZONS))}",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
    ]

    # Compute summary stats per horizon
    horizon_stats: dict[int, dict] = {}
    for h in HORIZONS:
        maes = []
        rmses = []
        coverages = []
        for r in results:
            for hm in r.horizon_metrics:
                if hm.horizon == h and hm.n_predictions > 0:
                    maes.append(hm.mae)
                    rmses.append(hm.rmse)
                    if hm.coverage_95 is not None:
                        coverages.append(hm.coverage_95)
        if maes:
            horizon_stats[h] = {
                "mae": np.mean(maes),
                "rmse": np.mean(rmses),
                "coverage": np.mean(coverages) if coverages else None,
            }

    total_runtime = sum(r.runtime_seconds for r in results)

    lines.extend(
        [
            "### Performance by Horizon",
            "",
            "| Horizon | Mean MAE | Mean RMSE | Mean Coverage | MAE Ratio (vs h=1) |",
            "|---------|----------|-----------|---------------|-------------------|",
        ]
    )

    mae_h1 = horizon_stats.get(1, {}).get("mae", 1.0)
    for h in HORIZONS:
        if h in horizon_stats:
            stats = horizon_stats[h]
            coverage_str = f"{stats['coverage']:.1%}" if stats["coverage"] else "N/A"
            ratio = stats["mae"] / mae_h1 if mae_h1 > 0 else 0
            lines.append(
                f"| {h} | {stats['mae']:.4f} | {stats['rmse']:.4f} | "
                f"{coverage_str} | {ratio:.2f}x |"
            )

    lines.extend(
        [
            "",
            f"**Total Runtime:** {total_runtime:.2f}s",
            "",
            "---",
            "",
            "## Horizon Scaling Analysis",
            "",
            "How prediction error grows with forecast horizon:",
            "",
        ]
    )

    # Create horizon scaling table for each signal type
    lines.extend(
        [
            "| Signal | h=1 | h=4 | h=16 | h=64 | h=256 | h=1024 |",
            "|--------|-----|-----|------|------|-------|--------|",
        ]
    )

    for r in results:
        horizon_maes = {hm.horizon: hm.mae for hm in r.horizon_metrics}
        cols = [r.signal_name[:25]]
        for h in [1, 4, 16, 64, 256, 1024]:
            if h in horizon_maes:
                cols.append(f"{horizon_maes[h]:.2f}")
            else:
                cols.append("N/A")
        lines.append("| " + " | ".join(cols) + " |")

    lines.extend(
        [
            "",
            "---",
            "",
            "## Coverage by Horizon",
            "",
            "95% prediction interval coverage across horizons:",
            "",
            "| Signal | h=1 | h=4 | h=16 | h=64 | h=256 | h=1024 |",
            "|--------|-----|-----|------|------|-------|--------|",
        ]
    )

    for r in results:
        horizon_cov = {hm.horizon: hm.coverage_95 for hm in r.horizon_metrics}
        cols = [r.signal_name[:25]]
        for h in [1, 4, 16, 64, 256, 1024]:
            cov = horizon_cov.get(h)
            if cov is not None:
                cols.append(f"{cov:.0%}")
            else:
                cols.append("N/A")
        lines.append("| " + " | ".join(cols) + " |")

    lines.extend(
        [
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
                "| Signal | MAE (h=1) | MAE (h=64) | MAE (h=1024) | Coverage (h=1) | Dominant |",
                "|--------|-----------|------------|--------------|----------------|----------|",
            ]
        )

        for r in cat_results:
            h_metrics = {hm.horizon: hm for hm in r.horizon_metrics}
            mae_1 = f"{h_metrics[1].mae:.4f}" if 1 in h_metrics else "N/A"
            mae_64 = f"{h_metrics[64].mae:.4f}" if 64 in h_metrics else "N/A"
            mae_1024 = f"{h_metrics[1024].mae:.4f}" if 1024 in h_metrics else "N/A"
            cov_1 = (
                f"{h_metrics[1].coverage_95:.1%}"
                if 1 in h_metrics and h_metrics[1].coverage_95
                else "N/A"
            )
            lines.append(
                f"| {r.signal_name} | {mae_1} | {mae_64} | {mae_1024} | "
                f"{cov_1} | {r.dominant_group} ({r.dominant_group_weight:.0%}) |"
            )

        lines.extend(["", "---", ""])

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
            "## Key Findings",
            "",
            "### Error Growth Patterns",
            "",
        ]
    )

    # Analyze error growth patterns
    growth_patterns = []
    for r in results:
        if len(r.horizon_metrics) >= 2:
            h1_mae = r.horizon_metrics[0].mae if r.horizon_metrics[0].mae > 0 else 0.001
            h1024_mae = r.horizon_metrics[-1].mae if r.horizon_metrics[-1].n_predictions > 0 else h1_mae
            ratio = h1024_mae / h1_mae
            growth_patterns.append((r.signal_name, ratio, r.signal_category))

    growth_patterns.sort(key=lambda x: x[1])

    lines.append("**Slowest Error Growth (best long-horizon performance):**")
    for name, ratio, cat in growth_patterns[:5]:
        lines.append(f"- {name} ({cat}): {ratio:.1f}x increase from h=1 to h=1024")

    lines.append("")
    lines.append("**Fastest Error Growth (challenging for long-horizon):**")
    for name, ratio, cat in growth_patterns[-5:]:
        lines.append(f"- {name} ({cat}): {ratio:.1f}x increase from h=1 to h=1024")

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

    # Convert to JSON-serializable format
    json_data = []
    for r in results:
        d = dataclasses.asdict(r)
        # HorizonMetrics are already dicts after asdict
        json_data.append(d)

    with open(output_dir / "acceptance_results.json", "w") as f:
        json.dump(json_data, f, indent=2)

    report = generate_report(results)
    with open(output_dir / "acceptance_report.md", "w") as f:
        f.write(report)

    print(f"\nResults saved to {output_dir}/")


class TestSignalTaxonomyAcceptance:
    """Pytest wrapper for acceptance tests."""

    def test_deterministic_signals(self) -> None:
        """Test deterministic signals category."""
        gen = SignalGenerators(seed=42)

        result = run_test(
            gen.constant(n=500),
            "Constant",
            "Deterministic",
            "persistence",
            horizons=[1, 16, 64],
        )
        assert result.mae_h1 < 1.0

        result = run_test(
            gen.linear_trend(n=500),
            "Linear Trend",
            "Deterministic",
            "trend",
            horizons=[1, 16, 64],
        )
        assert result.mae_h1 < 2.0

    def test_stochastic_signals(self) -> None:
        """Test stochastic signals category."""
        gen = SignalGenerators(seed=42)

        result = run_test(
            gen.white_noise(n=500),
            "White Noise",
            "Stochastic",
            "persistence",
            horizons=[1, 16, 64],
        )
        assert result.coverage_h1 is not None
        assert result.coverage_h1 > 0.7

        result = run_test(
            gen.ar1_mean_reverting(n=500),
            "AR(1)",
            "Stochastic",
            "reversion",
            horizons=[1, 16, 64],
        )
        assert result.mae_h1 < 1.0

    def test_regime_changing(self) -> None:
        """Test regime-changing signals."""
        gen = SignalGenerators(seed=42)

        result = run_test(
            gen.structural_break(n=500),
            "Structural Break",
            "Non-Stationary",
            "persistence",
            config=AEGISConfig(break_threshold=2.0),
            horizons=[1, 16, 64],
        )
        assert result.mae_h1 < 5.0

    def test_multi_stream(self) -> None:
        """Test multi-stream signals."""
        gen = SignalGenerators(seed=42)

        y1, y2 = gen.lead_lag(n=500)
        result = run_multistream_test(
            {"leader": y1, "follower": y2},
            "Lead-Lag",
            "Multi-Stream",
            "persistence",
            "follower",
            config=AEGISConfig(cross_stream_lags=3),
            horizons=[1, 16, 64],
        )
        assert result.mae_h1 < 5.0

    def test_long_horizon_forecasting(self) -> None:
        """Test that long-horizon forecasting works."""
        gen = SignalGenerators(seed=42)

        result = run_test(
            gen.ar1_mean_reverting(n=2000, phi=0.8),
            "AR(1) Long Horizon",
            "Stochastic",
            "reversion",
            horizons=[1, 64, 256, 1024],
        )

        # Verify we have predictions at all horizons
        horizons_tested = [hm.horizon for hm in result.horizon_metrics if hm.n_predictions > 0]
        assert 1 in horizons_tested
        assert 1024 in horizons_tested

        # Error should generally increase with horizon for AR(1)
        h1_mae = result.horizon_metrics[0].mae
        h1024_mae = result.horizon_metrics[-1].mae
        # For AR(1) with phi=0.8, long-horizon converges to mean, so error shouldn't explode
        assert h1024_mae < 10 * h1_mae  # Reasonable growth


if __name__ == "__main__":
    print("=" * 70)
    print("AEGIS Signal Taxonomy Acceptance Tests")
    print(f"Testing horizons: {HORIZONS}")
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
