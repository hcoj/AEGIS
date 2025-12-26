"""Quantile tracking for calibrated prediction intervals.

Implements online quantile estimation for conformal-style calibration.
"""

import numpy as np
from scipy import stats

from aegis.core.prediction import Prediction


class QuantileTracker:
    """Online quantile tracker for calibrated intervals.

    Tracks empirical coverage and adjusts quantiles to achieve
    target coverage using gradient-based online learning.

    Attributes:
        target_coverage: Desired interval coverage (e.g., 0.95)
        learning_rate: Step size for quantile updates
        q_low: Lower quantile (in standardized units)
        q_high: Upper quantile (in standardized units)
    """

    def __init__(
        self,
        target_coverage: float = 0.95,
        learning_rate: float = 0.01,
    ) -> None:
        """Initialize QuantileTracker.

        Args:
            target_coverage: Desired coverage probability
            learning_rate: Learning rate for quantile updates
        """
        self.target_coverage: float = target_coverage
        self.learning_rate: float = learning_rate

        alpha = 1 - target_coverage
        self.q_low: float = stats.norm.ppf(alpha / 2)
        self.q_high: float = stats.norm.ppf(1 - alpha / 2)

        self._n_obs: int = 0
        self._n_in_interval: int = 0

    def get_interval(self, pred_mean: float, pred_std: float) -> tuple[float, float]:
        """Compute prediction interval.

        Args:
            pred_mean: Predicted mean
            pred_std: Predicted standard deviation

        Returns:
            Tuple of (lower, upper) interval bounds
        """
        lower = pred_mean + self.q_low * pred_std
        upper = pred_mean + self.q_high * pred_std
        return (lower, upper)

    def update(self, y: float, pred_mean: float, pred_std: float) -> None:
        """Update quantile estimates based on observation.

        Args:
            y: Observed value
            pred_mean: Predicted mean
            pred_std: Predicted standard deviation
        """
        self._n_obs += 1

        interval = self.get_interval(pred_mean, pred_std)
        in_interval = interval[0] <= y <= interval[1]

        if in_interval:
            self._n_in_interval += 1

        z = (y - pred_mean) / max(pred_std, 1e-10)

        alpha = 1 - self.target_coverage

        if z < self.q_low:
            self.q_low -= self.learning_rate * (1 - alpha / 2)
        else:
            self.q_low += self.learning_rate * (alpha / 2)

        if z > self.q_high:
            self.q_high += self.learning_rate * (1 - alpha / 2)
        else:
            self.q_high -= self.learning_rate * (alpha / 2)

    def calibrate_prediction(self, pred: Prediction) -> Prediction:
        """Add calibrated interval to prediction.

        Args:
            pred: Prediction object with mean and variance

        Returns:
            New Prediction with interval_lower and interval_upper set
        """
        std = np.sqrt(pred.variance)
        lower, upper = self.get_interval(pred.mean, std)

        return Prediction(
            mean=pred.mean,
            variance=pred.variance,
            interval_lower=lower,
            interval_upper=upper,
        )

    def reset(self) -> None:
        """Reset quantiles to Gaussian values."""
        alpha = 1 - self.target_coverage
        self.q_low = stats.norm.ppf(alpha / 2)
        self.q_high = stats.norm.ppf(1 - alpha / 2)
        self._n_obs = 0
        self._n_in_interval = 0

    def get_coverage_stats(self) -> dict:
        """Get coverage statistics.

        Returns:
            Dictionary with empirical and target coverage
        """
        if self._n_obs == 0:
            empirical = self.target_coverage
        else:
            empirical = self._n_in_interval / self._n_obs

        return {
            "empirical_coverage": empirical,
            "target_coverage": self.target_coverage,
            "n_observations": self._n_obs,
            "q_low": self.q_low,
            "q_high": self.q_high,
        }
