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


class HorizonAwareQuantileTracker:
    """Horizon-aware quantile tracker with continuous interpolation.

    Maintains quantile estimates at anchor horizons and interpolates smoothly
    between them using log-space interpolation. This avoids discontinuities
    that would occur with discrete buckets.

    Anchor horizons: 1, 16, 64, 256, 1024 (roughly evenly spaced in log2)

    Attributes:
        target_coverage: Desired interval coverage (e.g., 0.95)
        learning_rate: Step size for quantile updates
    """

    ANCHOR_HORIZONS: list[int] = [1, 16, 64, 256, 1024]

    def __init__(
        self,
        target_coverage: float = 0.95,
        learning_rate: float = 0.01,
    ) -> None:
        """Initialize HorizonAwareQuantileTracker.

        Args:
            target_coverage: Desired coverage probability
            learning_rate: Learning rate for quantile updates
        """
        self.target_coverage: float = target_coverage
        self.learning_rate: float = learning_rate

        alpha = 1 - target_coverage
        initial_q_low = stats.norm.ppf(alpha / 2)
        initial_q_high = stats.norm.ppf(1 - alpha / 2)

        self._anchor_quantiles: dict[int, tuple[float, float]] = {}
        for anchor in self.ANCHOR_HORIZONS:
            self._anchor_quantiles[anchor] = (initial_q_low, initial_q_high)

        self._anchor_stats: dict[int, dict] = {}
        for anchor in self.ANCHOR_HORIZONS:
            self._anchor_stats[anchor] = {"n_obs": 0, "n_in_interval": 0}

    def _get_interpolation_weights(self, horizon: int) -> list[tuple[int, float]]:
        """Get anchor horizons and weights for interpolation.

        Uses log-space interpolation for smooth transitions across
        orders of magnitude.

        Args:
            horizon: Prediction horizon

        Returns:
            List of (anchor, weight) tuples that sum to 1.0
        """
        log_h = np.log(max(horizon, 1))
        log_anchors = [np.log(a) for a in self.ANCHOR_HORIZONS]

        if log_h <= log_anchors[0]:
            return [(self.ANCHOR_HORIZONS[0], 1.0)]
        if log_h >= log_anchors[-1]:
            return [(self.ANCHOR_HORIZONS[-1], 1.0)]

        for i in range(len(log_anchors) - 1):
            if log_anchors[i] <= log_h <= log_anchors[i + 1]:
                t = (log_h - log_anchors[i]) / (log_anchors[i + 1] - log_anchors[i])
                return [
                    (self.ANCHOR_HORIZONS[i], 1.0 - t),
                    (self.ANCHOR_HORIZONS[i + 1], t),
                ]

        return [(self.ANCHOR_HORIZONS[0], 1.0)]

    def get_quantiles(self, horizon: int) -> tuple[float, float]:
        """Get interpolated quantiles for a specific horizon.

        Args:
            horizon: Prediction horizon

        Returns:
            Tuple of (q_low, q_high) interpolated from anchors
        """
        weights = self._get_interpolation_weights(horizon)

        q_low = sum(w * self._anchor_quantiles[a][0] for a, w in weights)
        q_high = sum(w * self._anchor_quantiles[a][1] for a, w in weights)

        return (q_low, q_high)

    def get_interval(
        self, pred_mean: float, pred_std: float, horizon: int = 1
    ) -> tuple[float, float]:
        """Compute prediction interval for a specific horizon.

        Args:
            pred_mean: Predicted mean
            pred_std: Predicted standard deviation
            horizon: Prediction horizon

        Returns:
            Tuple of (lower, upper) interval bounds
        """
        q_low, q_high = self.get_quantiles(horizon)
        lower = pred_mean + q_low * pred_std
        upper = pred_mean + q_high * pred_std
        return (lower, upper)

    def update(self, y: float, pred_mean: float, pred_std: float, horizon: int = 1) -> None:
        """Update quantile estimates at nearby anchors with interpolated weights.

        Args:
            y: Observed value
            pred_mean: Predicted mean
            pred_std: Predicted standard deviation
            horizon: Prediction horizon
        """
        weights = self._get_interpolation_weights(horizon)
        z = (y - pred_mean) / max(pred_std, 1e-10)
        alpha = 1 - self.target_coverage

        interval = self.get_interval(pred_mean, pred_std, horizon)
        in_interval = interval[0] <= y <= interval[1]

        for anchor, weight in weights:
            self._anchor_stats[anchor]["n_obs"] += weight
            if in_interval:
                self._anchor_stats[anchor]["n_in_interval"] += weight

            q_low, q_high = self._anchor_quantiles[anchor]
            scaled_lr = self.learning_rate * weight

            if z < q_low:
                q_low -= scaled_lr * (1 - alpha / 2)
            else:
                q_low += scaled_lr * (alpha / 2)

            if z > q_high:
                q_high += scaled_lr * (1 - alpha / 2)
            else:
                q_high -= scaled_lr * (alpha / 2)

            self._anchor_quantiles[anchor] = (q_low, q_high)

    def calibrate_prediction(self, pred: Prediction, horizon: int = 1) -> Prediction:
        """Add calibrated interval to prediction.

        Args:
            pred: Prediction object with mean and variance
            horizon: Prediction horizon

        Returns:
            New Prediction with interval_lower and interval_upper set
        """
        std = np.sqrt(pred.variance)
        lower, upper = self.get_interval(pred.mean, std, horizon)

        return Prediction(
            mean=pred.mean,
            variance=pred.variance,
            interval_lower=lower,
            interval_upper=upper,
        )

    def reset(self) -> None:
        """Reset all anchor quantiles to Gaussian values."""
        alpha = 1 - self.target_coverage
        initial_q_low = stats.norm.ppf(alpha / 2)
        initial_q_high = stats.norm.ppf(1 - alpha / 2)

        for anchor in self.ANCHOR_HORIZONS:
            self._anchor_quantiles[anchor] = (initial_q_low, initial_q_high)
            self._anchor_stats[anchor] = {"n_obs": 0, "n_in_interval": 0}

    def get_coverage_stats(self, horizon: int | None = None) -> dict:
        """Get coverage statistics.

        Args:
            horizon: Optional horizon to get stats for nearest anchor.
                     If None, returns aggregate stats across all anchors.

        Returns:
            Dictionary with empirical and target coverage
        """
        if horizon is not None:
            weights = self._get_interpolation_weights(horizon)
            anchor = weights[0][0]
            stats_data = self._anchor_stats[anchor]
            if stats_data["n_obs"] < 1:
                empirical = self.target_coverage
            else:
                empirical = stats_data["n_in_interval"] / stats_data["n_obs"]

            q_low, q_high = self.get_quantiles(horizon)
            return {
                "empirical_coverage": empirical,
                "target_coverage": self.target_coverage,
                "n_observations": stats_data["n_obs"],
                "q_low": q_low,
                "q_high": q_high,
                "nearest_anchor": anchor,
            }

        total_obs = sum(s["n_obs"] for s in self._anchor_stats.values())
        total_in = sum(s["n_in_interval"] for s in self._anchor_stats.values())

        if total_obs < 1:
            empirical = self.target_coverage
        else:
            empirical = total_in / total_obs

        return {
            "empirical_coverage": empirical,
            "target_coverage": self.target_coverage,
            "n_observations": total_obs,
            "per_anchor": {a: self.get_coverage_stats(a) for a in self.ANCHOR_HORIZONS},
        }
