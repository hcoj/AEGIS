"""Special models for AEGIS.

Special models capture non-standard dynamics:
- JumpDiffusionModel: Random walk with occasional discrete jumps
- ChangePointModel: Explicit regime change detection
"""

import numpy as np

from aegis.core.prediction import Prediction
from aegis.models.base import TemporalModel


class JumpDiffusionModel(TemporalModel):
    """Jump diffusion model.

    Distinguishes continuous volatility from discrete jumps.
    Uses Beta prior on jump probability with Bayesian updates.

    State:
        lambda_a, lambda_b: Beta prior parameters for jump probability
        sigma_sq_diff: Diffusion (continuous) variance
        mu_jump: Mean jump size
        sigma_sq_jump: Jump size variance
    """

    def __init__(
        self,
        jump_threshold: float = 3.0,
        decay: float = 0.95,
    ) -> None:
        """Initialize JumpDiffusionModel.

        Args:
            jump_threshold: Number of std devs to classify as jump
            decay: EWMA decay for variance estimation
        """
        self.jump_threshold: float = jump_threshold
        self.decay: float = decay

        self.lambda_a: float = 1.0
        self.lambda_b: float = 49.0

        self.sigma_sq_diff: float = 1.0
        self.mu_jump: float = 0.0
        self.sigma_sq_jump: float = 10.0

        self.last_y: float = 0.0
        self.recent_jumps: list[float] = []
        self._n_obs: int = 0

    def lambda_mean(self) -> float:
        """Return posterior mean of jump probability."""
        return self.lambda_a / (self.lambda_a + self.lambda_b)

    def lambda_var(self) -> float:
        """Return posterior variance of jump probability."""
        a, b = self.lambda_a, self.lambda_b
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    def update(self, y: float, t: int) -> None:
        """Update model with new observation.

        Args:
            y: Observed value
            t: Time index
        """
        if self._n_obs > 0:
            innovation = y - self.last_y

            is_jump = abs(innovation) > self.jump_threshold * np.sqrt(self.sigma_sq_diff)

            if is_jump:
                self.recent_jumps.append(innovation)
                if len(self.recent_jumps) > 20:
                    self.recent_jumps.pop(0)

                if len(self.recent_jumps) >= 3:
                    self.mu_jump = np.mean(self.recent_jumps)
                    self.sigma_sq_jump = np.var(self.recent_jumps) + 1e-6

                self.lambda_a += 1
            else:
                self.sigma_sq_diff = (
                    self.decay * self.sigma_sq_diff + (1 - self.decay) * innovation**2
                )
                self.lambda_b += 1

        self.last_y = y
        self._n_obs += 1

    def predict(self, horizon: int) -> Prediction:
        """Predict future value.

        Args:
            horizon: Steps ahead

        Returns:
            Prediction including jump risk
        """
        lam = self.lambda_mean()
        mean = self.last_y + horizon * lam * self.mu_jump

        var_per_step = self.sigma_sq_diff + lam * (self.mu_jump**2 + self.sigma_sq_jump)
        variance = var_per_step * horizon

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
        self.lambda_a = partial * 1.0 + (1 - partial) * self.lambda_a
        self.lambda_b = partial * 49.0 + (1 - partial) * self.lambda_b
        self.recent_jumps = []

    def epistemic_value(self) -> float:
        """Expected information gain about jump probability.

        Returns:
            Epistemic value (higher when lambda uncertain)
        """
        n_eff = self.lambda_a + self.lambda_b - 2
        if n_eff > 0:
            return 0.5 * np.log(1 + 1.0 / n_eff)
        return 1.0

    @property
    def n_parameters(self) -> int:
        """Number of learnable parameters."""
        return 4

    @property
    def group(self) -> str:
        """Model group."""
        return "special"


class ChangePointModel(TemporalModel):
    """Change point model.

    Explicitly models probability of regime change.
    High epistemic value when change point probability is uncertain.

    State:
        hazard_rate: Prior probability of change
        hazard_a, hazard_b: Beta prior on hazard
        mu: Current regime mean
        sigma_sq: Current regime variance
        regime_n: Observations in current regime
        run_length: Steps since last change
    """

    def __init__(
        self,
        hazard_rate: float = 0.01,
        decay: float = 0.9,
    ) -> None:
        """Initialize ChangePointModel.

        Args:
            hazard_rate: Prior probability of change per step
            decay: EWMA decay for variance estimation
        """
        self.hazard_rate: float = hazard_rate
        self.decay: float = decay

        self.hazard_a: float = 1.0
        self.hazard_b: float = 99.0

        self.mu: float = 0.0
        self.sigma_sq: float = 1.0
        self.prior_sigma_sq: float = 1.0

        self.regime_sum: float = 0.0
        self.regime_sum_sq: float = 0.0
        self.regime_n: int = 0

        self.last_y: float = 0.0
        self.run_length: int = 0
        self._n_obs: int = 0
        self._rng: np.random.Generator = np.random.default_rng()

    def update(self, y: float, t: int) -> None:
        """Update model with new observation.

        Args:
            y: Observed value
            t: Time index
        """
        if self._n_obs > 0:
            pred_mean = self.mu
            pred_var = self.sigma_sq * (1 + 1.0 / max(self.regime_n, 1))

            error = y - pred_mean
            ll_current = -0.5 * np.log(2 * np.pi * pred_var) - error**2 / (2 * pred_var)

            ll_new = -0.5 * np.log(2 * np.pi * self.sigma_sq) - y**2 / (2 * self.sigma_sq)

            hazard = self.hazard_a / (self.hazard_a + self.hazard_b)

            p_change = (
                hazard
                * np.exp(ll_new)
                / (hazard * np.exp(ll_new) + (1 - hazard) * np.exp(ll_current) + 1e-10)
            )

            if self._rng.random() < p_change or p_change > 0.5:
                self.regime_sum = y
                self.regime_sum_sq = y**2
                self.regime_n = 1
                self.run_length = 0
                self.hazard_a += 1
            else:
                self.regime_sum += y
                self.regime_sum_sq += y**2
                self.regime_n += 1
                self.run_length += 1
                self.hazard_b += 1

            if self.regime_n > 0:
                self.mu = self.regime_sum / self.regime_n
                if self.regime_n > 1:
                    var = (self.regime_sum_sq - self.regime_n * self.mu**2) / (self.regime_n - 1)
                    self.sigma_sq = self.decay * self.sigma_sq + (1 - self.decay) * max(var, 0.01)
        else:
            self.mu = y
            self.regime_sum = y
            self.regime_sum_sq = y**2
            self.regime_n = 1

        self.last_y = y
        self._n_obs += 1

    def predict(self, horizon: int) -> Prediction:
        """Predict future value.

        Args:
            horizon: Steps ahead

        Returns:
            Prediction with change risk in variance
        """
        variance = self.sigma_sq * (1 + horizon * self.hazard_rate)
        return Prediction(mean=self.mu, variance=max(variance, 1e-10))

    def log_likelihood(self, y: float) -> float:
        """Compute log-likelihood of observation.

        Args:
            y: Observed value

        Returns:
            Log probability density
        """
        pred_var = self.sigma_sq * (1 + 1.0 / max(self.regime_n, 1))
        return -0.5 * np.log(2 * np.pi * pred_var) - 0.5 * (y - self.mu) ** 2 / pred_var

    def reset(self, partial: float = 1.0) -> None:
        """Reset parameters toward priors.

        Args:
            partial: Interpolation weight (1.0 = full reset)
        """
        self.regime_n = int(self.regime_n * (1 - partial))
        self.hazard_a = partial * 1.0 + (1 - partial) * self.hazard_a
        self.hazard_b = partial * 99.0 + (1 - partial) * self.hazard_b
        self.sigma_sq = partial * self.prior_sigma_sq + (1 - partial) * self.sigma_sq

    def epistemic_value(self) -> float:
        """Expected information gain about hazard and regime mean.

        Returns:
            Epistemic value
        """
        hazard_info = 0.5 * np.log(1 + 1.0 / max(self.hazard_a + self.hazard_b, 1))

        if self.regime_n > 0:
            mean_info = 0.5 * np.log(1 + 1.0 / self.regime_n)
        else:
            mean_info = 1.0

        return hazard_info + mean_info

    @property
    def n_parameters(self) -> int:
        """Number of learnable parameters."""
        return 3

    @property
    def group(self) -> str:
        """Model group."""
        return "special"
