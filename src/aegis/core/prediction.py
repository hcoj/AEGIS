"""Prediction dataclass for AEGIS forecasts."""

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass
class Prediction:
    """Point prediction with uncertainty.

    Attributes:
        mean: Predicted value
        variance: Variance of prediction
        interval_lower: Optional calibrated lower bound (overrides Gaussian)
        interval_upper: Optional calibrated upper bound (overrides Gaussian)
    """

    mean: float
    variance: float
    interval_lower: float | None = None
    interval_upper: float | None = None

    @property
    def std(self) -> float:
        """Standard deviation of prediction.

        Returns:
            Square root of variance
        """
        return np.sqrt(self.variance)

    def interval(self, level: float = 0.95) -> tuple[float, float]:
        """Compute confidence interval.

        Uses calibrated bounds if available, otherwise Gaussian interval.

        Args:
            level: Coverage level (default 0.95)

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        z = norm.ppf(0.5 + level / 2)

        if self.interval_lower is not None:
            lower = self.interval_lower
        else:
            lower = self.mean - z * self.std

        if self.interval_upper is not None:
            upper = self.interval_upper
        else:
            upper = self.mean + z * self.std

        return lower, upper
