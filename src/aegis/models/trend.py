"""Trend models for AEGIS.

Trend models capture directional drift:
- LinearTrendModel: Pure regression-based linear trend
- LocalTrendModel: Holt's double exponential smoothing
- DampedTrendModel: Trend that decays toward zero over horizon
"""

import numpy as np

from aegis.core.prediction import Prediction
from aegis.models.base import TemporalModel


class LinearTrendModel(TemporalModel):
    """Linear trend model using online regression.

    Maintains sufficient statistics for linear regression (intercept + slope).
    Prediction extrapolates: intercept + slope * t

    State:
        intercept: Regression intercept
        slope: Regression slope
        sigma_sq: Estimated residual variance
        slope_var: Estimated slope uncertainty from regression SE
    """

    def __init__(self, sigma_sq: float = 1.0, decay: float = 0.99) -> None:
        """Initialize LinearTrendModel.

        Args:
            sigma_sq: Initial variance estimate
            decay: EWMA decay for variance estimation
        """
        self.intercept: float = 0.0
        self.slope: float = 0.0
        self.sigma_sq: float = sigma_sq
        self.prior_sigma_sq: float = sigma_sq
        self.prior_intercept: float = 0.0
        self.prior_slope: float = 0.0
        self.decay: float = decay
        self._initialized: bool = False
        self._n_obs: int = 0
        self.slope_var: float = 0.01
        self.prior_slope_var: float = 0.01

        # Sufficient statistics for online regression
        self._sum_t: float = 0.0
        self._sum_y: float = 0.0
        self._sum_ty: float = 0.0
        self._sum_tt: float = 0.0
        self._last_t: int = 0

    def update(self, y: float, t: int) -> None:
        """Update model with new observation.

        Args:
            y: Observed value
            t: Time index
        """
        self._n_obs += 1
        self._sum_t += t
        self._sum_y += y
        self._sum_ty += t * y
        self._sum_tt += t * t
        self._last_t = t

        if self._n_obs == 1:
            self.intercept = y
            self.prior_intercept = y
            self._initialized = True
        elif self._n_obs >= 2:
            # Compute regression coefficients
            n = self._n_obs
            denom = n * self._sum_tt - self._sum_t**2

            if abs(denom) > 1e-10:
                old_slope = self.slope
                self.slope = (n * self._sum_ty - self._sum_t * self._sum_y) / denom
                self.intercept = (self._sum_y - self.slope * self._sum_t) / n

                # Update slope variance from slope changes
                slope_change = self.slope - old_slope
                self.slope_var = self.decay * self.slope_var + (1 - self.decay) * slope_change**2

                # Update residual variance
                pred = self.intercept + self.slope * t
                error = y - pred
                self.sigma_sq = self.decay * self.sigma_sq + (1 - self.decay) * error**2

                # Compute standard error of slope from regression statistics
                if self._n_obs > 2:
                    # SE(slope)^2 = sigma_sq / sum((t - t_mean)^2)
                    t_mean = self._sum_t / n
                    ss_t = self._sum_tt - n * t_mean**2
                    if ss_t > 1e-10:
                        regression_slope_var = self.sigma_sq / ss_t
                        # Blend empirical slope variance with regression estimate
                        self.slope_var = max(self.slope_var, regression_slope_var)

    def predict(self, horizon: int) -> Prediction:
        """Predict cumulative change over horizon.

        Args:
            horizon: Steps ahead

        Returns:
            Prediction with mean = sum of (intercept + slope * (t+k)) for k=1..h
            This equals h * (intercept + slope * (t + (h+1)/2)).
            Variance grows quadratically with horizon due to slope uncertainty.
        """
        # Cumulative sum of arithmetic sequence: sum_{k=1}^{h} (intercept + slope * (t + k))
        # = h * intercept + slope * (h*t + h*(h+1)/2)
        # = h * (intercept + slope * (t + (h+1)/2))
        mean = horizon * (self.intercept + self.slope * (self._last_t + (horizon + 1) / 2))
        slope_uncertainty = max(self.slope_var, 1e-6)
        variance = self.sigma_sq + (horizon**2) * slope_uncertainty
        return Prediction(mean=mean, variance=max(variance, 1e-10))

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
        self.intercept = partial * self.prior_intercept + (1 - partial) * self.intercept
        self.slope = partial * self.prior_slope + (1 - partial) * self.slope
        self.sigma_sq = partial * self.prior_sigma_sq + (1 - partial) * self.sigma_sq
        self.slope_var = partial * self.prior_slope_var + (1 - partial) * self.slope_var

    @property
    def n_parameters(self) -> int:
        """Number of learnable parameters."""
        return 4

    @property
    def group(self) -> str:
        """Model group."""
        return "trend"


class LocalTrendModel(TemporalModel):
    """Local trend model (Holt's double exponential smoothing).

    Maintains exponentially smoothed level and slope.
    Prediction extrapolates: level + slope * horizon

    State:
        level: Current smoothed level
        slope: Current smoothed trend (slope)
        sigma_sq: Estimated prediction error variance (level component)
        slope_var: Estimated slope uncertainty variance
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
        self.slope_var: float = 0.01
        self.prior_slope_var: float = 0.01
        self._prev_slope: float = 0.0

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
            new_slope = self.beta * (new_level - self.level) + (1 - self.beta) * self.slope

            slope_change = new_slope - self._prev_slope
            self.slope_var = self.decay * self.slope_var + (1 - self.decay) * slope_change**2

            self._prev_slope = self.slope
            self.slope = new_slope
            self.level = new_level

        self._n_obs += 1

    def predict(self, horizon: int) -> Prediction:
        """Predict cumulative change over horizon.

        Args:
            horizon: Steps ahead

        Returns:
            Prediction with mean = sum of (level + slope * k) for k=1..h
            This equals h * level + slope * h*(h+1)/2.
            Variance grows quadratically with horizon due to slope uncertainty.
        """
        # Cumulative: sum_{k=1}^{h} (level + slope * k) = h * level + slope * h*(h+1)/2
        mean = horizon * self.level + self.slope * horizon * (horizon + 1) / 2
        slope_uncertainty = max(self.slope_var, self.sigma_sq * self.beta**2)
        variance = self.sigma_sq + (horizon**2) * slope_uncertainty
        return Prediction(mean=mean, variance=max(variance, 1e-10))

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
        self.slope_var = partial * self.prior_slope_var + (1 - partial) * self.slope_var

    @property
    def n_parameters(self) -> int:
        """Number of learnable parameters."""
        return 4

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
        sigma_sq: Estimated prediction error variance (level component)
        slope_var: Estimated slope uncertainty variance
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
        self.slope_var: float = 0.01
        self.prior_slope_var: float = 0.01
        self._prev_slope: float = 0.0

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
            new_slope = (
                self.beta * (new_level - self.level) + (1 - self.beta) * self.phi * self.slope
            )

            slope_change = new_slope - self._prev_slope
            self.slope_var = self.decay * self.slope_var + (1 - self.decay) * slope_change**2

            self._prev_slope = self.slope
            self.slope = new_slope
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

    def _cumulative_damped_sum(self, horizon: int) -> float:
        """Compute cumulative sum of damped sums: Σ[k=1..h] damped_sum(k).

        Args:
            horizon: Number of terms

        Returns:
            Sum of damped sums
        """
        if self.phi == 0.0:
            return 0.0
        if self.phi == 1.0:
            return horizon * (horizon + 1) / 2

        # Σ[k=1..h] damped_sum(k) where damped_sum(k) = phi*(1-phi^k)/(1-phi)
        # = phi/(1-phi) * Σ[k=1..h] (1 - phi^k)
        # = phi/(1-phi) * (h - phi*(1-phi^h)/(1-phi))
        phi = self.phi
        geometric_sum = phi * (1 - phi**horizon) / (1 - phi)
        return phi / (1 - phi) * horizon - phi / (1 - phi) * geometric_sum

    def predict(self, horizon: int) -> Prediction:
        """Predict cumulative change over horizon.

        Args:
            horizon: Steps ahead

        Returns:
            Prediction with cumulative damped trend.
            Variance grows quadratically with horizon due to slope uncertainty.
        """
        # Cumulative: sum_{k=1}^{h} (level + slope * damped_sum(k))
        mean = horizon * self.level + self.slope * self._cumulative_damped_sum(horizon)
        slope_uncertainty = max(self.slope_var, self.sigma_sq * self.beta**2)
        variance = self.sigma_sq + (horizon**2) * slope_uncertainty
        return Prediction(mean=mean, variance=max(variance, 1e-10))

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
        self.slope_var = partial * self.prior_slope_var + (1 - partial) * self.slope_var

    @property
    def n_parameters(self) -> int:
        """Number of learnable parameters."""
        return 5

    @property
    def group(self) -> str:
        """Model group."""
        return "trend"
