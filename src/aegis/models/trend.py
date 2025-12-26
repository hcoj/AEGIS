"""Trend models for AEGIS.

Trend models capture directional drift:
- LocalTrendModel: Holt's double exponential smoothing
- DampedTrendModel: Trend that decays toward zero over horizon
"""

import numpy as np

from aegis.core.prediction import Prediction
from aegis.models.base import TemporalModel


class LocalTrendModel(TemporalModel):
    """Local trend model (Holt's double exponential smoothing).

    Maintains exponentially smoothed level and slope.
    Prediction extrapolates: level + slope * horizon

    State:
        level: Current smoothed level
        slope: Current smoothed trend (slope)
        sigma_sq: Estimated prediction error variance
    """

    def __init__(
        self,
        alpha: float = 0.1,
        beta: float = 0.1,
        sigma_sq: float = 1.0,
        decay: float = 0.94,
    ) -> None:
        """Initialize LocalTrendModel.

        Args:
            alpha: Level smoothing parameter
            beta: Slope smoothing parameter
            sigma_sq: Initial variance estimate
            decay: EWMA decay for variance estimation
        """
        self.alpha: float = alpha
        self.beta: float = beta
        self.level: float = 0.0
        self.slope: float = 0.0
        self.sigma_sq: float = sigma_sq
        self.prior_sigma_sq: float = sigma_sq
        self.prior_level: float = 0.0
        self.prior_slope: float = 0.0
        self.decay: float = decay
        self._initialized: bool = False
        self._n_obs: int = 0

    def update(self, y: float, t: int) -> None:
        """Update model with new observation.

        Args:
            y: Observed value
            t: Time index
        """
        if not self._initialized:
            self.level = y
            self.prior_level = y
            self._initialized = True
        else:
            error = y - (self.level + self.slope)
            self.sigma_sq = self.decay * self.sigma_sq + (1 - self.decay) * error**2

            new_level = self.alpha * y + (1 - self.alpha) * (self.level + self.slope)
            self.slope = self.beta * (new_level - self.level) + (1 - self.beta) * self.slope
            self.level = new_level

        self._n_obs += 1

    def predict(self, horizon: int) -> Prediction:
        """Predict future value.

        Args:
            horizon: Steps ahead

        Returns:
            Prediction with mean = level + slope * horizon
        """
        mean = self.level + self.slope * horizon
        variance = self.sigma_sq * (1 + (horizon - 1) * (self.alpha**2 + self.beta**2))
        return Prediction(mean=mean, variance=variance)

    def log_likelihood(self, y: float) -> float:
        """Compute log-likelihood of observation.

        Args:
            y: Observed value

        Returns:
            Log probability density
        """
        pred = self.predict(horizon=1)
        return -0.5 * np.log(2 * np.pi * pred.variance) - 0.5 * (y - pred.mean) ** 2 / pred.variance

    def reset(self, partial: float = 1.0) -> None:
        """Reset parameters toward priors.

        Args:
            partial: Interpolation weight (1.0 = full reset)
        """
        self.level = partial * self.prior_level + (1 - partial) * self.level
        self.slope = partial * self.prior_slope + (1 - partial) * self.slope
        self.sigma_sq = partial * self.prior_sigma_sq + (1 - partial) * self.sigma_sq

    @property
    def n_parameters(self) -> int:
        """Number of learnable parameters."""
        return 3

    @property
    def group(self) -> str:
        """Model group."""
        return "trend"


class DampedTrendModel(TemporalModel):
    """Damped trend model.

    Like LocalTrendModel but slope decays toward zero over horizon.
    Prediction: level + slope * (phi + phi^2 + ... + phi^horizon)

    State:
        level: Current smoothed level
        slope: Current smoothed trend (decays with phi)
        phi: Damping parameter (0 = no trend, 1 = full trend)
        sigma_sq: Estimated prediction error variance
    """

    def __init__(
        self,
        alpha: float = 0.1,
        beta: float = 0.1,
        phi: float = 0.9,
        sigma_sq: float = 1.0,
        decay: float = 0.94,
    ) -> None:
        """Initialize DampedTrendModel.

        Args:
            alpha: Level smoothing parameter
            beta: Slope smoothing parameter
            phi: Damping parameter in (0, 1)
            sigma_sq: Initial variance estimate
            decay: EWMA decay for variance estimation
        """
        self.alpha: float = alpha
        self.beta: float = beta
        self.phi: float = phi
        self.level: float = 0.0
        self.slope: float = 0.0
        self.sigma_sq: float = sigma_sq
        self.prior_sigma_sq: float = sigma_sq
        self.prior_level: float = 0.0
        self.prior_slope: float = 0.0
        self.decay: float = decay
        self._initialized: bool = False
        self._n_obs: int = 0

    def update(self, y: float, t: int) -> None:
        """Update model with new observation.

        Args:
            y: Observed value
            t: Time index
        """
        if not self._initialized:
            self.level = y
            self.prior_level = y
            self._initialized = True
        else:
            error = y - (self.level + self.phi * self.slope)
            self.sigma_sq = self.decay * self.sigma_sq + (1 - self.decay) * error**2

            new_level = self.alpha * y + (1 - self.alpha) * (self.level + self.phi * self.slope)
            self.slope = (
                self.beta * (new_level - self.level) + (1 - self.beta) * self.phi * self.slope
            )
            self.level = new_level

        self._n_obs += 1

    def _damped_sum(self, horizon: int) -> float:
        """Compute sum phi + phi^2 + ... + phi^horizon.

        Args:
            horizon: Number of terms

        Returns:
            Geometric sum
        """
        if self.phi == 1.0:
            return float(horizon)
        if self.phi == 0.0:
            return 0.0
        return self.phi * (1 - self.phi**horizon) / (1 - self.phi)

    def predict(self, horizon: int) -> Prediction:
        """Predict future value.

        Args:
            horizon: Steps ahead

        Returns:
            Prediction with damped trend extrapolation
        """
        mean = self.level + self.slope * self._damped_sum(horizon)
        variance = self.sigma_sq * (1 + (horizon - 1) * (self.alpha**2 + self.beta**2))
        return Prediction(mean=mean, variance=variance)

    def log_likelihood(self, y: float) -> float:
        """Compute log-likelihood of observation.

        Args:
            y: Observed value

        Returns:
            Log probability density
        """
        pred = self.predict(horizon=1)
        return -0.5 * np.log(2 * np.pi * pred.variance) - 0.5 * (y - pred.mean) ** 2 / pred.variance

    def reset(self, partial: float = 1.0) -> None:
        """Reset parameters toward priors.

        Args:
            partial: Interpolation weight (1.0 = full reset)
        """
        self.level = partial * self.prior_level + (1 - partial) * self.level
        self.slope = partial * self.prior_slope + (1 - partial) * self.slope
        self.sigma_sq = partial * self.prior_sigma_sq + (1 - partial) * self.sigma_sq

    @property
    def n_parameters(self) -> int:
        """Number of learnable parameters."""
        return 4

    @property
    def group(self) -> str:
        """Model group."""
        return "trend"
