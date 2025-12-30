"""Reversion models for AEGIS.

Reversion models capture mean-reverting dynamics:
- MeanReversionModel: AR(1) toward learned mean
- AsymmetricMeanReversionModel: Different speeds above/below mean
- ThresholdARModel: Regime-dependent dynamics based on threshold
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from aegis.core.prediction import Prediction
from aegis.models.base import MAX_SIGMA_SQ, TemporalModel

if TYPE_CHECKING:
    from aegis.config import AEGISConfig


class MeanReversionModel(TemporalModel):
    """Mean reversion (AR(1) toward mean) model.

    Models: y_t = mu + phi * (y_{t-1} - mu) + epsilon

    Uses online learning for mu and phi with Bayesian parameter tracking.

    State:
        mu: Estimated long-run mean
        phi: Autoregressive coefficient (mean reversion speed)
        sigma_sq: Innovation variance
        phi_var: Variance of phi estimate (for epistemic value)
    """

    def __init__(
        self,
        mu: float = 0.0,
        phi: float = 0.9,
        sigma_sq: float = 1.0,
        decay: float = 0.94,
        config: AEGISConfig | None = None,
    ) -> None:
        """Initialize MeanReversionModel.

        Args:
            mu: Initial mean estimate
            phi: Initial AR coefficient
            sigma_sq: Initial variance estimate
            decay: EWMA decay for parameter estimation
            config: Optional AEGIS configuration for robust estimation
        """
        self.mu: float = mu
        self.phi: float = phi
        self.sigma_sq: float = sigma_sq
        self.decay: float = decay

        self.prior_mu: float = mu
        self.prior_phi: float = phi
        self.prior_sigma_sq: float = sigma_sq

        self.phi_var: float = 0.1
        self._last_y: float = 0.0
        self._initialized: bool = False
        self._n_obs: int = 0
        self._use_robust: bool = config.use_robust_estimation if config else False
        self._outlier_threshold: float = config.outlier_threshold if config else 5.0

        self._sum_y: float = 0.0
        self._sum_y_sq: float = 0.0
        self._sum_xy: float = 0.0
        self._sum_x_sq: float = 0.0

    def update(self, y: float, t: int) -> None:
        """Update model with new observation.

        Args:
            y: Observed value
            t: Time index
        """
        if not self._initialized:
            self._last_y = y
            self.mu = y
            self._initialized = True
            self._n_obs += 1
            return

        x = self._last_y - self.mu

        error = y - self.mu - self.phi * x
        if self._use_robust:
            from aegis.models.robust import robust_weight

            weight = robust_weight(error, np.sqrt(self.sigma_sq), self._outlier_threshold)
            self.sigma_sq = self.decay * self.sigma_sq + (1 - self.decay) * weight * error**2
        else:
            self.sigma_sq = self.decay * self.sigma_sq + (1 - self.decay) * error**2
        self.sigma_sq = min(self.sigma_sq, MAX_SIGMA_SQ)

        self._sum_y = self.decay * self._sum_y + y
        self._sum_y_sq = self.decay * self._sum_y_sq + y**2
        self._sum_xy = self.decay * self._sum_xy + x * (y - self.mu)
        self._sum_x_sq = self.decay * self._sum_x_sq + x**2

        if self._sum_x_sq > 1e-6:
            phi_new = np.clip(self._sum_xy / self._sum_x_sq, 0.0, 0.999)
            self.phi = 0.9 * self.phi + 0.1 * phi_new
            self.phi_var = self.sigma_sq / max(self._sum_x_sq, 1e-6)

        self.mu = self.decay * self.mu + (1 - self.decay) * y

        self._last_y = y
        self._n_obs += 1

    def predict(self, horizon: int) -> Prediction:
        """Predict value at horizon steps ahead.

        Args:
            horizon: Steps ahead

        Returns:
            Point prediction: mu + phi^h * (last_y - mu).
            Variance accounts for mean-reversion dynamics.
        """
        x = self._last_y - self.mu

        if abs(self.phi) < 0.999:
            mean = self.mu + x * (self.phi**horizon)
            variance = self.sigma_sq * (1 - self.phi ** (2 * horizon)) / (1 - self.phi**2)
        else:
            mean = self._last_y  # phi ≈ 1, acts like random walk
            variance = self.sigma_sq * horizon

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
        self.mu = partial * self.prior_mu + (1 - partial) * self.mu
        self.phi = partial * self.prior_phi + (1 - partial) * self.phi
        self.sigma_sq = partial * self.prior_sigma_sq + (1 - partial) * self.sigma_sq
        self.phi_var = partial * 0.1 + (1 - partial) * self.phi_var

    def epistemic_value(self) -> float:
        """Expected information gain about phi from next observation.

        Returns:
            Epistemic value (higher near regime uncertainty)
        """
        x = self._last_y - self.mu
        if self.sigma_sq < 1e-10:
            return 0.0
        return 0.5 * np.log(1 + self.phi_var * x**2 / self.sigma_sq)

    @property
    def n_parameters(self) -> int:
        """Number of learnable parameters."""
        return 3

    @property
    def group(self) -> str:
        """Model group."""
        return "reversion"


class AsymmetricMeanReversionModel(TemporalModel):
    """Asymmetric mean reversion model.

    Different reversion speeds above vs below the mean.
    Models:
        if y_{t-1} > mu: y_t = mu + phi_up * (y_{t-1} - mu) + eps
        if y_{t-1} <= mu: y_t = mu + phi_down * (y_{t-1} - mu) + eps

    State:
        mu: Estimated long-run mean
        phi_up: AR coefficient when above mean
        phi_down: AR coefficient when below mean
        sigma_sq: Innovation variance
    """

    def __init__(
        self,
        mu: float = 0.0,
        phi_up: float = 0.8,
        phi_down: float = 0.9,
        sigma_sq: float = 1.0,
        decay: float = 0.94,
        config: AEGISConfig | None = None,
    ) -> None:
        """Initialize AsymmetricMeanReversionModel.

        Args:
            mu: Initial mean estimate
            phi_up: Initial AR coefficient above mean
            phi_down: Initial AR coefficient below mean
            sigma_sq: Initial variance estimate
            decay: EWMA decay for parameter estimation
            config: Optional AEGIS configuration for robust estimation
        """
        self.mu: float = mu
        self.phi_up: float = phi_up
        self.phi_down: float = phi_down
        self.sigma_sq: float = sigma_sq
        self.decay: float = decay

        self.prior_mu: float = mu
        self.prior_phi_up: float = phi_up
        self.prior_phi_down: float = phi_down
        self.prior_sigma_sq: float = sigma_sq

        self._last_y: float = 0.0
        self._initialized: bool = False
        self._n_obs: int = 0
        self._use_robust: bool = config.use_robust_estimation if config else False
        self._outlier_threshold: float = config.outlier_threshold if config else 5.0

        self._sum_xy_up: float = 0.0
        self._sum_x_sq_up: float = 0.0
        self._sum_xy_down: float = 0.0
        self._sum_x_sq_down: float = 0.0

    def update(self, y: float, t: int) -> None:
        """Update model with new observation.

        Args:
            y: Observed value
            t: Time index
        """
        if not self._initialized:
            self._last_y = y
            self.mu = y
            self._initialized = True
            self._n_obs += 1
            return

        x = self._last_y - self.mu
        above = self._last_y > self.mu

        if above:
            phi = self.phi_up
        else:
            phi = self.phi_down

        error = y - self.mu - phi * x
        if self._use_robust:
            from aegis.models.robust import robust_weight

            weight = robust_weight(error, np.sqrt(self.sigma_sq), self._outlier_threshold)
            self.sigma_sq = self.decay * self.sigma_sq + (1 - self.decay) * weight * error**2
        else:
            self.sigma_sq = self.decay * self.sigma_sq + (1 - self.decay) * error**2
        self.sigma_sq = min(self.sigma_sq, MAX_SIGMA_SQ)

        if above:
            self._sum_xy_up = self.decay * self._sum_xy_up + x * (y - self.mu)
            self._sum_x_sq_up = self.decay * self._sum_x_sq_up + x**2
            if self._sum_x_sq_up > 1e-6:
                phi_new = np.clip(self._sum_xy_up / self._sum_x_sq_up, 0.0, 0.999)
                self.phi_up = 0.9 * self.phi_up + 0.1 * phi_new
        else:
            self._sum_xy_down = self.decay * self._sum_xy_down + x * (y - self.mu)
            self._sum_x_sq_down = self.decay * self._sum_x_sq_down + x**2
            if self._sum_x_sq_down > 1e-6:
                phi_new = np.clip(self._sum_xy_down / self._sum_x_sq_down, 0.0, 0.999)
                self.phi_down = 0.9 * self.phi_down + 0.1 * phi_new

        self.mu = self.decay * self.mu + (1 - self.decay) * y
        self._last_y = y
        self._n_obs += 1

    def predict(self, horizon: int) -> Prediction:
        """Predict value at horizon steps ahead.

        Args:
            horizon: Steps ahead

        Returns:
            Point prediction using appropriate phi.
        """
        x = self._last_y - self.mu
        phi = self.phi_up if self._last_y > self.mu else self.phi_down

        if abs(phi) < 0.999:
            mean = self.mu + x * (phi**horizon)
            variance = self.sigma_sq * (1 - phi ** (2 * horizon)) / (1 - phi**2)
        else:
            mean = self._last_y
            variance = self.sigma_sq * horizon

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
        self.mu = partial * self.prior_mu + (1 - partial) * self.mu
        self.phi_up = partial * self.prior_phi_up + (1 - partial) * self.phi_up
        self.phi_down = partial * self.prior_phi_down + (1 - partial) * self.phi_down
        self.sigma_sq = partial * self.prior_sigma_sq + (1 - partial) * self.sigma_sq

    @property
    def n_parameters(self) -> int:
        """Number of learnable parameters."""
        return 4

    @property
    def group(self) -> str:
        """Model group."""
        return "reversion"


class ThresholdARModel(TemporalModel):
    """Threshold autoregressive model.

    Different AR dynamics above vs below a threshold tau.
    Models:
        if y_{t-1} < tau: y_t = phi_low * y_{t-1} + eps
        if y_{t-1} >= tau: y_t = phi_high * y_{t-1} + eps

    State:
        tau: Threshold value
        phi_low: AR coefficient below threshold
        phi_high: AR coefficient above threshold
        sigma_sq: Innovation variance
    """

    def __init__(
        self,
        tau: float = 0.0,
        phi_low: float = 0.5,
        phi_high: float = 0.9,
        sigma_sq: float = 1.0,
        decay: float = 0.94,
        config: AEGISConfig | None = None,
    ) -> None:
        """Initialize ThresholdARModel.

        Args:
            tau: Threshold value
            phi_low: Initial AR coefficient below threshold
            phi_high: Initial AR coefficient above threshold
            sigma_sq: Initial variance estimate
            decay: EWMA decay for parameter estimation
            config: Optional AEGIS configuration for robust estimation
        """
        self.tau: float = tau
        self.phi_low: float = phi_low
        self.phi_high: float = phi_high
        self.sigma_sq: float = sigma_sq
        self.decay: float = decay

        self.prior_phi_low: float = phi_low
        self.prior_phi_high: float = phi_high
        self.prior_sigma_sq: float = sigma_sq

        self.phi_low_var: float = 0.1
        self.phi_high_var: float = 0.1

        self._last_y: float = 0.0
        self._initialized: bool = False
        self._n_obs: int = 0
        self._use_robust: bool = config.use_robust_estimation if config else False
        self._outlier_threshold: float = config.outlier_threshold if config else 5.0

        self._sum_xy_low: float = 0.0
        self._sum_x_sq_low: float = 0.0
        self._sum_xy_high: float = 0.0
        self._sum_x_sq_high: float = 0.0

    def update(self, y: float, t: int) -> None:
        """Update model with new observation.

        Args:
            y: Observed value
            t: Time index
        """
        if not self._initialized:
            self._last_y = y
            self._initialized = True
            self._n_obs += 1
            return

        below = self._last_y < self.tau
        phi = self.phi_low if below else self.phi_high

        error = y - phi * self._last_y
        if self._use_robust:
            from aegis.models.robust import robust_weight

            weight = robust_weight(error, np.sqrt(self.sigma_sq), self._outlier_threshold)
            self.sigma_sq = self.decay * self.sigma_sq + (1 - self.decay) * weight * error**2
        else:
            self.sigma_sq = self.decay * self.sigma_sq + (1 - self.decay) * error**2
        self.sigma_sq = min(self.sigma_sq, MAX_SIGMA_SQ)

        if below:
            self._sum_xy_low = self.decay * self._sum_xy_low + self._last_y * y
            self._sum_x_sq_low = self.decay * self._sum_x_sq_low + self._last_y**2
            if self._sum_x_sq_low > 1e-6:
                phi_new = np.clip(self._sum_xy_low / self._sum_x_sq_low, 0.0, 0.999)
                self.phi_low = 0.9 * self.phi_low + 0.1 * phi_new
                self.phi_low_var = self.sigma_sq / max(self._sum_x_sq_low, 1e-6)
        else:
            self._sum_xy_high = self.decay * self._sum_xy_high + self._last_y * y
            self._sum_x_sq_high = self.decay * self._sum_x_sq_high + self._last_y**2
            if self._sum_x_sq_high > 1e-6:
                phi_new = np.clip(self._sum_xy_high / self._sum_x_sq_high, 0.0, 0.999)
                self.phi_high = 0.9 * self.phi_high + 0.1 * phi_new
                self.phi_high_var = self.sigma_sq / max(self._sum_x_sq_high, 1e-6)

        self._last_y = y
        self._n_obs += 1

    def predict(self, horizon: int) -> Prediction:
        """Predict value at horizon steps ahead.

        Args:
            horizon: Steps ahead

        Returns:
            Point prediction using appropriate phi.
        """
        phi = self.phi_low if self._last_y < self.tau else self.phi_high

        if abs(phi) < 0.999:
            mean = self._last_y * (phi**horizon)
            variance = self.sigma_sq * (1 - phi ** (2 * horizon)) / (1 - phi**2)
        else:
            mean = self._last_y
            variance = self.sigma_sq * horizon

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
        self.phi_low = partial * self.prior_phi_low + (1 - partial) * self.phi_low
        self.phi_high = partial * self.prior_phi_high + (1 - partial) * self.phi_high
        self.sigma_sq = partial * self.prior_sigma_sq + (1 - partial) * self.sigma_sq

    def epistemic_value(self) -> float:
        """Expected information gain about regime parameters.

        Peaks near threshold where regime assignment is uncertain.

        Returns:
            Epistemic value
        """
        distance_to_threshold = abs(self._last_y - self.tau)

        proximity_weight = np.exp(-(distance_to_threshold**2) / (2 * max(self.sigma_sq, 0.1)))

        if self._last_y < self.tau:
            phi_var = self.phi_low_var
        else:
            phi_var = self.phi_high_var

        if self.sigma_sq < 1e-10:
            return 0.0

        param_uncertainty = 0.5 * np.log(1 + phi_var / max(self.sigma_sq, 1e-6))
        return param_uncertainty + proximity_weight

    @property
    def n_parameters(self) -> int:
        """Number of learnable parameters."""
        return 4

    @property
    def group(self) -> str:
        """Model group."""
        return "reversion"


class LevelAwareMeanReversionModel(TemporalModel):
    """Mean reversion model that tracks levels internally.

    Receives returns (differences) as input but maintains an internal
    cumulative level estimate to detect mean reversion at the level.

    This addresses the issue where standard mean reversion on returns
    learns φ≈0 for true mean-reverting levels (like OU processes).

    State:
        level: Cumulative level estimate
        mu: Estimated long-run mean level
        phi: Autoregressive coefficient (reversion speed)
        sigma_sq: Innovation variance
    """

    def __init__(
        self,
        mu: float = 0.0,
        phi: float = 0.9,
        sigma_sq: float = 1.0,
        decay: float = 0.94,
        config: AEGISConfig | None = None,
    ) -> None:
        """Initialize LevelAwareMeanReversionModel.

        Args:
            mu: Initial mean level estimate
            phi: Initial AR coefficient
            sigma_sq: Initial variance estimate
            decay: EWMA decay for parameter estimation
            config: Optional AEGIS configuration for robust estimation
        """
        self.level: float = 0.0
        self.mu: float = mu
        self.phi: float = phi
        self.sigma_sq: float = sigma_sq
        self.decay: float = decay

        self.prior_mu: float = mu
        self.prior_phi: float = phi
        self.prior_sigma_sq: float = sigma_sq

        self._n_obs: int = 0
        self._sum_xy: float = 0.0
        self._sum_x_sq: float = 0.0
        self._use_robust: bool = config.use_robust_estimation if config else False
        self._outlier_threshold: float = config.outlier_threshold if config else 5.0

    def update(self, y: float, t: int) -> None:
        """Update model with new observation (a return/difference).

        Args:
            y: Observed return (y_t - y_{t-1})
            t: Time index
        """
        self.level += y

        deviation = self.level - self.mu

        expected_next_level = self.mu + self.phi * deviation
        expected_return = expected_next_level - self.level

        error = y - expected_return
        if self._use_robust:
            from aegis.models.robust import robust_weight

            weight = robust_weight(error, np.sqrt(self.sigma_sq), self._outlier_threshold)
            self.sigma_sq = self.decay * self.sigma_sq + (1 - self.decay) * weight * error**2
        else:
            self.sigma_sq = self.decay * self.sigma_sq + (1 - self.decay) * error**2
        self.sigma_sq = min(self.sigma_sq, MAX_SIGMA_SQ)

        self._sum_xy = self.decay * self._sum_xy + deviation * y
        self._sum_x_sq = self.decay * self._sum_x_sq + deviation**2

        if self._sum_x_sq > 1e-6:
            phi_implied = self._sum_xy / self._sum_x_sq + 1.0
            phi_new = np.clip(phi_implied, 0.0, 0.999)
            self.phi = 0.9 * self.phi + 0.1 * phi_new

        self.mu = self.decay * self.mu + (1 - self.decay) * self.level
        self._n_obs += 1

    def predict(self, horizon: int) -> Prediction:
        """Predict expected return at horizon steps ahead.

        Args:
            horizon: Steps ahead

        Returns:
            Expected return at time t+horizon.
        """
        deviation = self.level - self.mu

        if abs(self.phi) < 0.999:
            # Expected return at horizon h is proportional to phi^(h-1)
            expected_return = (self.phi - 1.0) * deviation * (self.phi ** (horizon - 1))
            variance = self.sigma_sq * (1 - self.phi ** (2 * horizon)) / (1 - self.phi**2)
        else:
            expected_return = 0.0  # phi ≈ 1 means no expected return
            variance = self.sigma_sq * horizon

        return Prediction(mean=expected_return, variance=max(variance, 1e-10))

    def log_likelihood(self, y: float) -> float:
        """Compute log-likelihood of return observation.

        Args:
            y: Observed return

        Returns:
            Log probability density
        """
        deviation = self.level - self.mu
        expected_return = (self.phi - 1.0) * deviation

        return (
            -0.5 * np.log(2 * np.pi * self.sigma_sq)
            - 0.5 * (y - expected_return) ** 2 / self.sigma_sq
        )

    def reset(self, partial: float = 1.0) -> None:
        """Reset parameters toward priors.

        Args:
            partial: Interpolation weight (1.0 = full reset)
        """
        self.mu = partial * self.prior_mu + (1 - partial) * self.mu
        self.phi = partial * self.prior_phi + (1 - partial) * self.phi
        self.sigma_sq = partial * self.prior_sigma_sq + (1 - partial) * self.sigma_sq
        if partial > 0.5:
            self.level = self.mu

    @property
    def n_parameters(self) -> int:
        """Number of learnable parameters."""
        return 3

    @property
    def group(self) -> str:
        """Model group."""
        return "reversion"

    @property
    def name(self) -> str:
        """Model name."""
        return "LevelAwareMeanReversionModel"
