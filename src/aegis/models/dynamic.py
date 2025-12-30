"""Dynamic models for AEGIS.

Dynamic models capture autocorrelation structure:
- AR2Model: Two-lag autoregression with RLS learning
- MA1Model: Moving average with shock effects
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from aegis.core.prediction import Prediction
from aegis.models.base import MAX_SIGMA_SQ, TemporalModel

if TYPE_CHECKING:
    from aegis.config import AEGISConfig


class AR2Model(TemporalModel):
    """AR(2) model with Recursive Least Squares learning.

    Captures richer autocorrelation structure. Can model oscillatory
    behaviour via complex roots.

    Model: y_t = c + phi1 * y_{t-1} + phi2 * y_{t-2} + epsilon

    State:
        c: Constant term
        phi1: First lag coefficient
        phi2: Second lag coefficient
        sigma_sq: Innovation variance
    """

    def __init__(
        self,
        c: float = 0.0,
        phi1: float = 0.5,
        phi2: float = 0.3,
        forget: float = 0.99,
        decay: float = 0.95,
        config: AEGISConfig | None = None,
    ) -> None:
        """Initialize AR2Model.

        Args:
            c: Initial constant term
            phi1: Initial first lag coefficient
            phi2: Initial second lag coefficient
            forget: RLS forgetting factor
            decay: EWMA decay for variance estimation
            config: Optional AEGIS configuration for robust estimation
        """
        self.c: float = c
        self.phi1: float = phi1
        self.phi2: float = phi2
        self.forget: float = forget
        self.decay: float = decay

        self.prior_c: float = c
        self.prior_phi1: float = phi1
        self.prior_phi2: float = phi2

        self.sigma_sq: float = 1.0
        self.prior_sigma_sq: float = 1.0

        self.y_lag1: float = 0.0
        self.y_lag2: float = 0.0
        self._n_obs: int = 0
        self._use_robust: bool = config.use_robust_estimation if config else False
        self._outlier_threshold: float = config.outlier_threshold if config else 5.0

        self.P: np.ndarray = np.eye(3) * 10.0
        self.theta: np.ndarray = np.array([c, phi1, phi2])

    def update(self, y: float, t: int) -> None:
        """Update model with new observation.

        Args:
            y: Observed value
            t: Time index
        """
        if self._n_obs >= 2:
            x = np.array([1.0, self.y_lag1, self.y_lag2])

            pred = np.dot(x, self.theta)
            error = y - pred

            Px = self.P @ x
            denom = self.forget + np.dot(x, Px)
            gain = Px / denom

            # Clip error to prevent overflow in parameter updates
            MAX_ERROR = 1e6
            clipped_error = np.clip(error, -MAX_ERROR, MAX_ERROR)
            self.theta = self.theta + gain * clipped_error
            self.P = (self.P - np.outer(gain, Px)) / self.forget

            # Bound all parameters to prevent overflow
            MAX_CONSTANT = 1e6
            MAX_PHI = 5.0
            self.theta[0] = np.clip(self.theta[0], -MAX_CONSTANT, MAX_CONSTANT)
            self.theta[1] = np.clip(self.theta[1], -MAX_PHI, MAX_PHI)
            self.theta[2] = np.clip(self.theta[2], -MAX_PHI, MAX_PHI)

            self.c, self.phi1, self.phi2 = self.theta

            if abs(self.phi1) + abs(self.phi2) > 0.99:
                scale = 0.99 / (abs(self.phi1) + abs(self.phi2) + 1e-6)
                self.phi1 *= scale
                self.phi2 *= scale
                self.theta[1:] = [self.phi1, self.phi2]

            # Use clipped error for variance update to prevent overflow
            error_sq = min(clipped_error**2, MAX_SIGMA_SQ)
            if self._use_robust:
                from aegis.models.robust import robust_weight

                weight = robust_weight(
                    clipped_error, np.sqrt(self.sigma_sq), self._outlier_threshold
                )
                self.sigma_sq = self.decay * self.sigma_sq + (1 - self.decay) * weight * error_sq
            else:
                self.sigma_sq = self.decay * self.sigma_sq + (1 - self.decay) * error_sq
            self.sigma_sq = min(self.sigma_sq, MAX_SIGMA_SQ)

        self.y_lag2 = self.y_lag1
        self.y_lag1 = y
        self._n_obs += 1

    def predict(self, horizon: int) -> Prediction:
        """Predict value at horizon steps ahead.

        Args:
            horizon: Steps ahead

        Returns:
            Point prediction with AR(2) extrapolation
        """
        y1, y2 = self.y_lag1, self.y_lag2

        for _ in range(horizon):
            y_new = self.c + self.phi1 * y1 + self.phi2 * y2
            y2 = y1
            y1 = y_new

        variance = np.clip(self.sigma_sq * horizon, 1e-10, 1e10)
        return Prediction(mean=y1, variance=variance)

    def log_likelihood(self, y: float) -> float:
        """Compute log-likelihood of observation.

        Args:
            y: Observed value

        Returns:
            Log probability density
        """
        if self._n_obs < 2:
            return -0.5 * np.log(2 * np.pi * self.sigma_sq)

        pred = self.c + self.phi1 * self.y_lag1 + self.phi2 * self.y_lag2
        return -0.5 * np.log(2 * np.pi * self.sigma_sq) - 0.5 * (y - pred) ** 2 / self.sigma_sq

    def reset(self, partial: float = 1.0) -> None:
        """Reset parameters toward priors.

        Args:
            partial: Interpolation weight (1.0 = full reset)
        """
        self.theta = (
            partial * np.array([self.prior_c, self.prior_phi1, self.prior_phi2])
            + (1 - partial) * self.theta
        )
        self.c, self.phi1, self.phi2 = self.theta
        self.P = partial * np.eye(3) * 10.0 + (1 - partial) * self.P
        self.sigma_sq = partial * self.prior_sigma_sq + (1 - partial) * self.sigma_sq

    @property
    def n_parameters(self) -> int:
        """Number of learnable parameters."""
        return 4

    @property
    def group(self) -> str:
        """Model group."""
        return "dynamic"


class MA1Model(TemporalModel):
    """MA(1) model.

    Captures one-period shock effects. Useful for inventory adjustments
    and filtered signals.

    Model: y_t = theta * epsilon_{t-1} + epsilon_t

    State:
        theta: MA coefficient
        last_error: Previous prediction error
        sigma_sq: Innovation variance
    """

    def __init__(
        self,
        theta: float = 0.5,
        lr: float = 0.01,
        decay: float = 0.95,
        config: AEGISConfig | None = None,
    ) -> None:
        """Initialize MA1Model.

        Args:
            theta: Initial MA coefficient
            lr: Learning rate for gradient updates
            decay: EWMA decay for variance estimation
            config: Optional AEGIS configuration for robust estimation
        """
        self.theta: float = theta
        self.prior_theta: float = theta
        self.lr: float = lr
        self.decay: float = decay

        self.last_error: float = 0.0
        self.sigma_sq: float = 1.0
        self.prior_sigma_sq: float = 1.0
        self._n_obs: int = 0
        self._use_robust: bool = config.use_robust_estimation if config else False
        self._outlier_threshold: float = config.outlier_threshold if config else 5.0

    def update(self, y: float, t: int) -> None:
        """Update model with new observation.

        Args:
            y: Observed value
            t: Time index
        """
        pred = self.theta * self.last_error
        error = y - pred

        grad = -error * self.last_error
        self.theta = np.clip(self.theta - self.lr * grad, -0.99, 0.99)

        if self._use_robust:
            from aegis.models.robust import robust_weight

            weight = robust_weight(error, np.sqrt(self.sigma_sq), self._outlier_threshold)
            self.sigma_sq = self.decay * self.sigma_sq + (1 - self.decay) * weight * error**2
        else:
            self.sigma_sq = self.decay * self.sigma_sq + (1 - self.decay) * error**2
        self.sigma_sq = min(self.sigma_sq, MAX_SIGMA_SQ)
        self.last_error = error
        self._n_obs += 1

    def predict(self, horizon: int) -> Prediction:
        """Predict value at horizon steps ahead.

        Args:
            horizon: Steps ahead

        Returns:
            Point prediction. At h=1 uses MA(1) term, h>1 predicts 0.
        """
        # MA(1): E[y_{t+1}] = theta * last_error, E[y_{t+k}] = 0 for k > 1
        if horizon == 1:
            mean = self.theta * self.last_error
        else:
            mean = 0.0  # MA(1) has no predictive power beyond 1 step
        raw_variance = self.sigma_sq * (1 + (horizon - 1) * (1 + self.theta**2))
        variance = np.clip(raw_variance, 1e-10, 1e10)

        return Prediction(mean=mean, variance=variance)

    def log_likelihood(self, y: float) -> float:
        """Compute log-likelihood of observation.

        Uses predict(1) for consistency with other models.

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
        self.theta = partial * self.prior_theta + (1 - partial) * self.theta
        self.last_error = 0.0
        self.sigma_sq = partial * self.prior_sigma_sq + (1 - partial) * self.sigma_sq

    @property
    def n_parameters(self) -> int:
        """Number of learnable parameters."""
        return 2

    @property
    def group(self) -> str:
        """Model group."""
        return "dynamic"
