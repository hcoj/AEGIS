"""Periodic models for AEGIS.

Periodic models capture cyclical patterns:
- OscillatorBankModel: Bank of sinusoidal oscillators at different frequencies
- SeasonalDummyModel: Separate mean for each position in the cycle
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from aegis.core.prediction import Prediction
from aegis.models.base import MAX_SIGMA_SQ, TemporalModel

if TYPE_CHECKING:
    from aegis.config import AEGISConfig


class OscillatorBankModel(TemporalModel):
    """Bank of oscillators at different frequencies.

    Captures periodic structure by learning amplitude and phase at each frequency.
    Uses gradient descent to update coefficients: a_k cos(wt) + b_k sin(wt).

    State:
        periods: List of periods for the oscillator bank
        a: Cosine coefficients for each frequency
        b: Sine coefficients for each frequency
        sigma_sq: Estimated prediction error variance
    """

    def __init__(
        self,
        periods: list[int] | None = None,
        lr: float = 0.01,
        decay: float = 0.95,
        phase_lock_threshold: float = 0.8,
        config: AEGISConfig | None = None,
    ) -> None:
        """Initialize OscillatorBankModel.

        Args:
            periods: List of oscillator periods (default: [4, 8, 16, 32, 64, 128, 256])
            lr: Learning rate for coefficient updates
            decay: EWMA decay for variance estimation
            phase_lock_threshold: Deprecated, kept for backwards compatibility
            config: Optional AEGIS configuration for robust estimation
        """
        self.periods: list[int] = periods or [4, 8, 16, 32, 64, 128, 256]
        self.n_freqs: int = len(self.periods)
        self.lr: float = lr
        self.decay: float = decay
        self.phase_lock_threshold: float = phase_lock_threshold  # Kept for backwards compat

        self.a: np.ndarray = np.zeros(self.n_freqs)
        self.b: np.ndarray = np.zeros(self.n_freqs)

        # Phase stability tracking (kept for backwards compatibility, not used in predict)
        self.phase_stability: np.ndarray = np.zeros(self.n_freqs)
        self._last_phase: np.ndarray = np.zeros(self.n_freqs)

        self.sigma_sq: float = 1.0
        self.prior_sigma_sq: float = 1.0
        self.t: int = 0
        self._n_obs: int = 0
        self._use_robust: bool = config.use_robust_estimation if config else False
        self._outlier_threshold: float = config.outlier_threshold if config else 5.0

    def _compute_prediction(self, t: int) -> float:
        """Compute prediction at time t.

        Args:
            t: Time index

        Returns:
            Predicted value
        """
        pred = 0.0
        for k, period in enumerate(self.periods):
            omega = 2 * np.pi / period
            pred += self.a[k] * np.cos(omega * t) + self.b[k] * np.sin(omega * t)
        return pred

    def _is_phase_locked(self, freq_idx: int) -> bool:
        """Check if frequency has stable phase (deprecated).

        Args:
            freq_idx: Index into self.periods

        Returns:
            True if phase stability exceeds threshold and amplitude is significant.
            Kept for backwards compatibility but no longer used in predict().
        """
        amplitude = np.sqrt(self.a[freq_idx] ** 2 + self.b[freq_idx] ** 2)
        return self.phase_stability[freq_idx] >= self.phase_lock_threshold and amplitude > 1e-4

    def update(self, y: float, t: int) -> None:
        """Update model with new observation.

        Args:
            y: Observed value
            t: Time index
        """
        self.t = t

        pred = self._compute_prediction(t)
        error = y - pred

        if self._use_robust:
            from aegis.models.robust import robust_weight

            weight = robust_weight(error, np.sqrt(self.sigma_sq), self._outlier_threshold)
            self.sigma_sq = self.decay * self.sigma_sq + (1 - self.decay) * weight * error**2
        else:
            self.sigma_sq = self.decay * self.sigma_sq + (1 - self.decay) * error**2
        # Maintain minimum variance floor for calibrated intervals
        MIN_SIGMA_SQ = 1e-4
        self.sigma_sq = max(self.sigma_sq, MIN_SIGMA_SQ)
        self.sigma_sq = min(self.sigma_sq, MAX_SIGMA_SQ)

        for k, period in enumerate(self.periods):
            omega = 2 * np.pi / period
            cos_term = np.cos(omega * t)
            sin_term = np.sin(omega * t)

            self.a[k] += self.lr * error * cos_term
            self.b[k] += self.lr * error * sin_term

            # Track phase stability for diagnostics (not used in predict)
            amplitude = np.sqrt(self.a[k] ** 2 + self.b[k] ** 2)
            if amplitude > 1e-6:
                current_phase = np.arctan2(self.b[k], self.a[k])
                phase_diff = abs(current_phase - self._last_phase[k])
                phase_diff = min(phase_diff, 2 * np.pi - phase_diff)

                phase_consistency = 1.0 - phase_diff / np.pi
                stability_decay = 0.95
                self.phase_stability[k] = (
                    stability_decay * self.phase_stability[k]
                    + (1 - stability_decay) * phase_consistency
                )
                self._last_phase[k] = current_phase

        self._n_obs += 1

    def _cumulative_prediction(self, horizon: int) -> float:
        """Compute cumulative prediction over horizon.

        When phase is locked, uses period-aligned extrapolation to avoid drift.
        When not locked, uses closed-form Dirichlet kernel sum.

        Args:
            horizon: Number of steps

        Returns:
            Cumulative predicted value
        """
        cumsum = 0.0
        for k, period in enumerate(self.periods):
            omega = 2 * np.pi / period

            if self._is_phase_locked(k):
                cumsum += self._locked_cumsum(k, horizon)
            else:
                cumsum += self._unlocked_cumsum(k, horizon, omega)
        return cumsum

    def _locked_cumsum(self, freq_idx: int, horizon: int) -> float:
        """Period-aligned cumulative sum using locked phase.

        Uses direct coefficient computation for accuracy:
        pred(t) = a*cos(ωt) + b*sin(ωt)

        Args:
            freq_idx: Index of the frequency
            horizon: Number of steps

        Returns:
            Cumulative sum using period-aligned extrapolation
        """
        period = self.periods[freq_idx]
        omega = 2 * np.pi / period
        a = self.a[freq_idx]
        b = self.b[freq_idx]

        full_periods = horizon // period
        remainder = horizon % period

        period_sum = sum(
            a * np.cos(omega * (self.t + step)) + b * np.sin(omega * (self.t + step))
            for step in range(1, period + 1)
        )

        partial_sum = sum(
            a * np.cos(omega * (self.t + step)) + b * np.sin(omega * (self.t + step))
            for step in range(1, remainder + 1)
        )

        return full_periods * period_sum + partial_sum

    def _unlocked_cumsum(self, freq_idx: int, horizon: int, omega: float) -> float:
        """Closed-form cumulative sum using Dirichlet kernel.

        Uses:
        Σ_{k=1}^{h} cos(ω(t+k)) = sin(hω/2) / sin(ω/2) * cos(ω(t + (h+1)/2))
        Σ_{k=1}^{h} sin(ω(t+k)) = sin(hω/2) / sin(ω/2) * sin(ω(t + (h+1)/2))

        Args:
            freq_idx: Index of the frequency
            horizon: Number of steps
            omega: Angular frequency

        Returns:
            Cumulative sum using closed-form formula
        """
        half_omega = omega / 2

        if abs(np.sin(half_omega)) < 1e-10:
            return horizon * (
                self.a[freq_idx] * np.cos(omega * (self.t + 1))
                + self.b[freq_idx] * np.sin(omega * (self.t + 1))
            )

        scale = np.sin(horizon * half_omega) / np.sin(half_omega)
        center_t = self.t + (horizon + 1) / 2
        return self.a[freq_idx] * scale * np.cos(omega * center_t) + self.b[
            freq_idx
        ] * scale * np.sin(omega * center_t)

    def predict(self, horizon: int) -> Prediction:
        """Predict cumulative change over horizon.

        Args:
            horizon: Steps ahead

        Returns:
            Cumulative prediction with oscillator extrapolation.
            Variance is constant (learned noise level) - for well-learned
            periodic signals, we know the pattern so uncertainty doesn't grow.
        """
        # O(1) closed-form cumulative sum using Dirichlet kernel
        cumsum = 0.0
        for k, period in enumerate(self.periods):
            omega = 2 * np.pi / period
            cumsum += self._unlocked_cumsum(k, horizon, omega)

        # Constant variance - we know the pattern, uncertainty doesn't grow
        return Prediction(mean=cumsum, variance=max(self.sigma_sq, 1e-10))

    def log_likelihood(self, y: float) -> float:
        """Compute log-likelihood of observation.

        Args:
            y: Observed value

        Returns:
            Log probability density
        """
        pred = self._compute_prediction(self.t + 1)
        return -0.5 * np.log(2 * np.pi * self.sigma_sq) - 0.5 * (y - pred) ** 2 / self.sigma_sq

    def reset(self, partial: float = 1.0) -> None:
        """Reset parameters toward priors.

        Args:
            partial: Interpolation weight (1.0 = full reset)
        """
        self.a *= 1 - partial
        self.b *= 1 - partial
        self.sigma_sq = partial * self.prior_sigma_sq + (1 - partial) * self.sigma_sq
        self.phase_stability *= 1 - partial

    @property
    def n_parameters(self) -> int:
        """Number of learnable parameters."""
        return 2 * self.n_freqs + 1

    @property
    def group(self) -> str:
        """Model group."""
        return "periodic"

    @property
    def name(self) -> str:
        """Model name including periods."""
        if len(self.periods) == 1:
            return f"OscillatorBankModel_p{self.periods[0]}"
        return f"OscillatorBankModel_p{'_'.join(str(p) for p in self.periods)}"


class SeasonalDummyModel(TemporalModel):
    """Seasonal dummy model.

    Maintains separate mean for each position in the cycle.
    Better than sinusoids for sharp seasonal patterns.

    State:
        period: Length of the seasonal cycle
        means: Array of means for each position
        counts: Effective sample count for each position (with forgetting)
        sigma_sq: Estimated prediction error variance
    """

    def __init__(
        self,
        period: int,
        forget: float = 0.99,
        decay: float = 0.95,
        config: AEGISConfig | None = None,
    ) -> None:
        """Initialize SeasonalDummyModel.

        Args:
            period: Length of seasonal cycle
            forget: Forgetting factor for seasonal mean updates
            decay: EWMA decay for variance estimation
            config: Optional AEGIS configuration for robust estimation
        """
        self.period: int = period
        self.forget: float = forget
        self.decay: float = decay

        self.means: np.ndarray = np.zeros(period)
        self.counts: np.ndarray = np.zeros(period)

        self.sigma_sq: float = 1.0
        self.prior_sigma_sq: float = 1.0
        self.t: int = 0
        self._n_obs: int = 0
        self._use_robust: bool = config.use_robust_estimation if config else False
        self._outlier_threshold: float = config.outlier_threshold if config else 5.0

    def update(self, y: float, t: int) -> None:
        """Update model with new observation.

        Args:
            y: Observed value
            t: Time index
        """
        self.t = t
        s = t % self.period

        error = y - self.means[s]
        if self._use_robust:
            from aegis.models.robust import robust_weight

            weight = robust_weight(error, np.sqrt(self.sigma_sq), self._outlier_threshold)
            self.sigma_sq = self.decay * self.sigma_sq + (1 - self.decay) * weight * error**2
        else:
            self.sigma_sq = self.decay * self.sigma_sq + (1 - self.decay) * error**2
        self.sigma_sq = min(self.sigma_sq, MAX_SIGMA_SQ)

        self.counts[s] = self.forget * self.counts[s] + 1
        alpha = 1.0 / self.counts[s]
        self.means[s] = (1 - alpha) * self.means[s] + alpha * y

        self._n_obs += 1

    def predict(self, horizon: int) -> Prediction:
        """Predict cumulative change over horizon.

        Args:
            horizon: Steps ahead

        Returns:
            Cumulative prediction with seasonal means.
            Variance grows with horizon due to phase uncertainty.
        """
        # Sum seasonal means over the horizon
        # Each complete cycle contributes sum(means)
        # Plus partial cycle at the end
        full_cycles = horizon // self.period
        remainder = horizon % self.period

        # Sum of one complete cycle
        cycle_sum = np.sum(self.means)

        # Cumulative = full_cycles * cycle_sum + partial cycle
        cumsum = full_cycles * cycle_sum

        # Add partial cycle (positions t+1 to t+remainder)
        for k in range(1, remainder + 1):
            s = (self.t + k) % self.period
            cumsum += self.means[s]

        # Phase uncertainty grows linearly with horizon
        phase_uncertainty = 0.01 * horizon
        variance = self.sigma_sq * (1 + phase_uncertainty)
        return Prediction(mean=cumsum, variance=max(variance, 1e-10))

    def log_likelihood(self, y: float) -> float:
        """Compute log-likelihood of observation.

        Args:
            y: Observed value

        Returns:
            Log probability density
        """
        s = (self.t + 1) % self.period
        pred = self.means[s]
        return -0.5 * np.log(2 * np.pi * self.sigma_sq) - 0.5 * (y - pred) ** 2 / self.sigma_sq

    def reset(self, partial: float = 1.0) -> None:
        """Reset parameters toward priors.

        Args:
            partial: Interpolation weight (1.0 = full reset)
        """
        self.means *= 1 - partial
        self.counts *= 1 - partial
        self.sigma_sq = partial * self.prior_sigma_sq + (1 - partial) * self.sigma_sq

    @property
    def n_parameters(self) -> int:
        """Number of learnable parameters."""
        return self.period + 1

    @property
    def group(self) -> str:
        """Model group."""
        return "periodic"

    @property
    def name(self) -> str:
        """Model name including period."""
        return f"SeasonalDummyModel_p{self.period}"
