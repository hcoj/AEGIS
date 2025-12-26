"""Periodic models for AEGIS.

Periodic models capture cyclical patterns:
- OscillatorBankModel: Bank of sinusoidal oscillators at different frequencies
- SeasonalDummyModel: Separate mean for each position in the cycle
"""

import numpy as np

from aegis.core.prediction import Prediction
from aegis.models.base import TemporalModel


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
    ) -> None:
        """Initialize OscillatorBankModel.

        Args:
            periods: List of oscillator periods (default: [4, 8, 16, 32, 64, 128, 256])
            lr: Learning rate for coefficient updates
            decay: EWMA decay for variance estimation
        """
        self.periods: list[int] = periods or [4, 8, 16, 32, 64, 128, 256]
        self.n_freqs: int = len(self.periods)
        self.lr: float = lr
        self.decay: float = decay

        self.a: np.ndarray = np.zeros(self.n_freqs)
        self.b: np.ndarray = np.zeros(self.n_freqs)

        self.sigma_sq: float = 1.0
        self.prior_sigma_sq: float = 1.0
        self.t: int = 0
        self._n_obs: int = 0

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

    def update(self, y: float, t: int) -> None:
        """Update model with new observation.

        Args:
            y: Observed value
            t: Time index
        """
        self.t = t

        pred = self._compute_prediction(t)
        error = y - pred

        self.sigma_sq = self.decay * self.sigma_sq + (1 - self.decay) * error**2

        for k, period in enumerate(self.periods):
            omega = 2 * np.pi / period
            cos_term = np.cos(omega * t)
            sin_term = np.sin(omega * t)

            self.a[k] += self.lr * error * cos_term
            self.b[k] += self.lr * error * sin_term

        self._n_obs += 1

    def predict(self, horizon: int) -> Prediction:
        """Predict future value.

        Args:
            horizon: Steps ahead

        Returns:
            Prediction with oscillator extrapolation
        """
        future_t = self.t + horizon
        mean = self._compute_prediction(future_t)
        return Prediction(mean=mean, variance=self.sigma_sq)

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

    @property
    def n_parameters(self) -> int:
        """Number of learnable parameters."""
        return 2 * self.n_freqs + 1

    @property
    def group(self) -> str:
        """Model group."""
        return "periodic"


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
    ) -> None:
        """Initialize SeasonalDummyModel.

        Args:
            period: Length of seasonal cycle
            forget: Forgetting factor for seasonal mean updates
            decay: EWMA decay for variance estimation
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

    def update(self, y: float, t: int) -> None:
        """Update model with new observation.

        Args:
            y: Observed value
            t: Time index
        """
        self.t = t
        s = t % self.period

        error = y - self.means[s]
        self.sigma_sq = self.decay * self.sigma_sq + (1 - self.decay) * error**2

        self.counts[s] = self.forget * self.counts[s] + 1
        alpha = 1.0 / self.counts[s]
        self.means[s] = (1 - alpha) * self.means[s] + alpha * y

        self._n_obs += 1

    def predict(self, horizon: int) -> Prediction:
        """Predict future value.

        Args:
            horizon: Steps ahead

        Returns:
            Prediction with seasonal mean
        """
        s = (self.t + horizon) % self.period
        return Prediction(mean=self.means[s], variance=self.sigma_sq)

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
