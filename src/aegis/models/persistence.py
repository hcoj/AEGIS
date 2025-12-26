"""Persistence models for AEGIS.

Persistence models assume the future looks like the recent past:
- RandomWalkModel: Predict last value, variance scales with horizon
- LocalLevelModel: Exponentially smoothed level
"""

import numpy as np

from aegis.core.prediction import Prediction
from aegis.models.base import TemporalModel


class RandomWalkModel(TemporalModel):
    """Random walk model: y_{t+h} = y_t + noise.

    Predicts the last observed value. Variance scales linearly with horizon.
    Variance is estimated using EWMA of squared innovations.

    State:
        last_y: Most recent observation
        sigma_sq: Estimated innovation variance
        prior_sigma_sq: Prior variance for reset
    """

    def __init__(self, sigma_sq: float = 1.0, decay: float = 0.94) -> None:
        """Initialize RandomWalkModel.

        Args:
            sigma_sq: Initial variance estimate
            decay: EWMA decay for variance estimation
        """
        self.last_y: float = 0.0
        self.sigma_sq: float = sigma_sq
        self.prior_sigma_sq: float = sigma_sq
        self.decay: float = decay
        self._n_obs: int = 0

    def update(self, y: float, t: int) -> None:
        """Update model with new observation.

        Args:
            y: Observed value
            t: Time index
        """
        if self._n_obs > 0:
            innovation = y - self.last_y
            self.sigma_sq = self.decay * self.sigma_sq + (1 - self.decay) * innovation**2

        self.last_y = y
        self._n_obs += 1

    def predict(self, horizon: int) -> Prediction:
        """Predict future value.

        Args:
            horizon: Steps ahead

        Returns:
            Prediction with mean = last_y, variance scales with horizon
        """
        return Prediction(
            mean=self.last_y,
            variance=self.sigma_sq * horizon,
        )

    def log_likelihood(self, y: float) -> float:
        """Compute log-likelihood of observation.

        Args:
            y: Observed value

        Returns:
            Log probability density under N(last_y, sigma_sq)
        """
        pred = self.predict(horizon=1)
        return -0.5 * np.log(2 * np.pi * pred.variance) - 0.5 * (y - pred.mean) ** 2 / pred.variance

    def reset(self, partial: float = 1.0) -> None:
        """Reset parameters toward priors.

        Args:
            partial: Interpolation weight (1.0 = full reset)
        """
        self.sigma_sq = partial * self.prior_sigma_sq + (1 - partial) * self.sigma_sq

    @property
    def n_parameters(self) -> int:
        """Number of learnable parameters."""
        return 1

    @property
    def group(self) -> str:
        """Model group."""
        return "persistence"


class LocalLevelModel(TemporalModel):
    """Local level (exponential smoothing) model.

    Maintains an exponentially weighted moving average of observations.
    Prediction is the current smoothed level.

    State:
        level: Current smoothed level
        sigma_sq: Estimated prediction error variance
        alpha: Smoothing parameter (higher = faster adaptation)
    """

    def __init__(self, alpha: float = 0.1, sigma_sq: float = 1.0, decay: float = 0.94) -> None:
        """Initialize LocalLevelModel.

        Args:
            alpha: Smoothing parameter in (0, 1)
            sigma_sq: Initial variance estimate
            decay: EWMA decay for variance estimation
        """
        self.alpha: float = alpha
        self.level: float = 0.0
        self.sigma_sq: float = sigma_sq
        self.prior_sigma_sq: float = sigma_sq
        self.prior_level: float = 0.0
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
            error = y - self.level
            self.sigma_sq = self.decay * self.sigma_sq + (1 - self.decay) * error**2
            self.level = self.alpha * y + (1 - self.alpha) * self.level

        self._n_obs += 1

    def predict(self, horizon: int) -> Prediction:
        """Predict future value.

        Args:
            horizon: Steps ahead

        Returns:
            Prediction with mean = level, variance increases with horizon
        """
        return Prediction(
            mean=self.level,
            variance=self.sigma_sq * (1 + (horizon - 1) * self.alpha**2),
        )

    def log_likelihood(self, y: float) -> float:
        """Compute log-likelihood of observation.

        Args:
            y: Observed value

        Returns:
            Log probability density under N(level, sigma_sq)
        """
        pred = self.predict(horizon=1)
        return -0.5 * np.log(2 * np.pi * pred.variance) - 0.5 * (y - pred.mean) ** 2 / pred.variance

    def reset(self, partial: float = 1.0) -> None:
        """Reset parameters toward priors.

        Args:
            partial: Interpolation weight (1.0 = full reset)
        """
        self.level = partial * self.prior_level + (1 - partial) * self.level
        self.sigma_sq = partial * self.prior_sigma_sq + (1 - partial) * self.sigma_sq

    @property
    def n_parameters(self) -> int:
        """Number of learnable parameters."""
        return 2

    @property
    def group(self) -> str:
        """Model group."""
        return "persistence"
