"""Variance models for AEGIS.

Variance models track and predict volatility:
- VolatilityTrackerModel: EWMA volatility with long-run mean
- LevelDependentVolModel: Variance scales with signal level
"""

import numpy as np

from aegis.core.prediction import Prediction
from aegis.models.base import MAX_SIGMA_SQ, TemporalModel


class VolatilityTrackerModel(TemporalModel):
    """EWMA volatility tracker.

    Tracks short-term and long-run variance.
    Predicts persistence with current volatility scaling.

    State:
        sigma_sq: Current (short-term) variance estimate
        long_run_var: Long-run variance estimate
        last_y: Most recent observation
    """

    def __init__(
        self,
        decay: float = 0.94,
        long_run_decay: float = 0.999,
    ) -> None:
        """Initialize VolatilityTrackerModel.

        Args:
            decay: EWMA decay for short-term variance
            long_run_decay: EWMA decay for long-run variance
        """
        self.decay: float = decay
        self.long_run_decay: float = long_run_decay

        self.sigma_sq: float = 1.0
        self.long_run_var: float = 1.0
        self.last_y: float = 0.0
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
            self.sigma_sq = min(self.sigma_sq, MAX_SIGMA_SQ)
            self.long_run_var = (
                self.long_run_decay * self.long_run_var + (1 - self.long_run_decay) * innovation**2
            )
            self.long_run_var = min(self.long_run_var, MAX_SIGMA_SQ)

        self.last_y = y
        self._n_obs += 1

    def predict(self, horizon: int) -> Prediction:
        """Predict cumulative change over horizon.

        Args:
            horizon: Steps ahead

        Returns:
            Cumulative prediction with current volatility scaling
        """
        return Prediction(mean=horizon * self.last_y, variance=max(self.sigma_sq * horizon, 1e-10))

    def log_likelihood(self, y: float) -> float:
        """Compute log-likelihood of observation.

        Args:
            y: Observed value

        Returns:
            Log probability density
        """
        return (
            -0.5 * np.log(2 * np.pi * self.sigma_sq) - 0.5 * (y - self.last_y) ** 2 / self.sigma_sq
        )

    def reset(self, partial: float = 1.0) -> None:
        """Reset toward long-run variance.

        Args:
            partial: Interpolation weight (1.0 = full reset)
        """
        self.sigma_sq = partial * self.long_run_var + (1 - partial) * self.sigma_sq

    def get_volatility_ratio(self) -> float:
        """Return current volatility relative to long-run.

        Returns:
            Ratio of current to long-run volatility
        """
        return np.sqrt(self.sigma_sq / max(self.long_run_var, 1e-10))

    @property
    def n_parameters(self) -> int:
        """Number of learnable parameters."""
        return 1

    @property
    def group(self) -> str:
        """Model group."""
        return "variance"


class LevelDependentVolModel(TemporalModel):
    """Level-dependent volatility model.

    Variance scales with signal level according to power relationship.
    Useful for count data and prices where percentage volatility is stable.

    State:
        gamma: Power relationship (0.5 = sqrt scaling)
        sigma_sq_base: Base variance estimate (at level 1)
        last_y: Most recent observation
    """

    def __init__(
        self,
        gamma: float = 0.5,
        decay: float = 0.95,
    ) -> None:
        """Initialize LevelDependentVolModel.

        Args:
            gamma: Power for level-variance relationship
            decay: EWMA decay for base variance estimation
        """
        self.gamma: float = gamma
        self.decay: float = decay

        self.sigma_sq_base: float = 1.0
        self.prior_sigma_sq_base: float = 1.0
        self.last_y: float = 1.0
        self._n_obs: int = 0

    def update(self, y: float, t: int) -> None:
        """Update model with new observation.

        Args:
            y: Observed value
            t: Time index
        """
        if self._n_obs > 0 and abs(self.last_y) > 0.01:
            innovation = y - self.last_y
            level_factor = abs(self.last_y) ** self.gamma
            normalised_sq = (innovation / max(level_factor, 1e-6)) ** 2
            self.sigma_sq_base = self.decay * self.sigma_sq_base + (1 - self.decay) * normalised_sq
            self.sigma_sq_base = min(self.sigma_sq_base, MAX_SIGMA_SQ)

        self.last_y = y
        self._n_obs += 1

    def predict(self, horizon: int) -> Prediction:
        """Predict cumulative change over horizon.

        Args:
            horizon: Steps ahead

        Returns:
            Cumulative prediction with level-scaled variance
        """
        level_factor = max(abs(self.last_y), 0.01) ** self.gamma
        variance = self.sigma_sq_base * level_factor**2 * horizon
        return Prediction(mean=horizon * self.last_y, variance=max(variance, 1e-10))

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
        """Reset base variance toward prior.

        Args:
            partial: Interpolation weight (1.0 = full reset)
        """
        self.sigma_sq_base = partial * self.prior_sigma_sq_base + (1 - partial) * self.sigma_sq_base

    @property
    def n_parameters(self) -> int:
        """Number of learnable parameters."""
        return 2

    @property
    def group(self) -> str:
        """Model group."""
        return "variance"
