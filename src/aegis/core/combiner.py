"""Model combiner for AEGIS.

Implements Expected Free Energy (EFE) based model weighting
using softmax over cumulative log-likelihood scores.
"""

import numpy as np

from aegis.config import AEGISConfig
from aegis.core.prediction import Prediction
from aegis.models.base import TemporalModel


class EFEModelCombiner:
    """EFE-based model combiner.

    Weights models by softmax over cumulative scores.
    Scores combine log-likelihood (pragmatic) and epistemic value.

    Attributes:
        n_models: Number of models being combined
        config: AEGIS configuration
        cumulative_scores: Running sum of scores per model
    """

    def __init__(self, n_models: int, config: AEGISConfig) -> None:
        """Initialize EFEModelCombiner.

        Args:
            n_models: Number of models to combine
            config: AEGIS configuration with temperature, forgetting, etc.
        """
        self.n_models: int = n_models
        self.config: AEGISConfig = config

        self.cumulative_scores: np.ndarray = np.zeros(n_models)
        self.last_pragmatic: np.ndarray = np.zeros(n_models)
        self.last_epistemic: np.ndarray | None = None
        self._n_obs: int = 0

        self.current_forget: float = config.likelihood_forget
        self._surprise_ema: float = 0.0

    def get_weights(self) -> np.ndarray:
        """Compute current model weights via softmax.

        If entropy_penalty_weight > 0, adaptively lower temperature
        when weights are spread out to encourage concentration.

        Returns:
            Array of model weights summing to 1
        """
        base_temp = max(self.config.temperature, 1e-10)

        if self.config.entropy_penalty_weight > 0 and self.n_models > 1:
            scores_base = self.cumulative_scores / base_temp
            max_score = np.max(scores_base)
            exp_scores = np.exp(scores_base - max_score)
            weights_base = exp_scores / np.sum(exp_scores)

            weights_clipped = np.clip(weights_base, 1e-10, 1.0)
            entropy = -np.sum(weights_clipped * np.log(weights_clipped))
            max_entropy = np.log(self.n_models)

            entropy_ratio = entropy / max_entropy

            temp_factor = 1.0 - self.config.entropy_penalty_weight * entropy_ratio
            temp_factor = max(temp_factor, 0.2)

            effective_temp = base_temp * temp_factor
        else:
            effective_temp = base_temp

        scores = self.cumulative_scores / effective_temp

        max_score = np.max(scores)
        exp_scores = np.exp(scores - max_score)
        weights = exp_scores / np.sum(exp_scores)

        return weights

    def update(self, models: list[TemporalModel], y: float, t: int) -> None:
        """Update model scores with new observation.

        Args:
            models: List of temporal models
            y: Observed value
            t: Time index
        """
        self._n_obs += 1

        pragmatic = np.array([m.log_likelihood(y) for m in models])
        self.last_pragmatic = pragmatic

        if self.config.use_epistemic_value:
            epistemic = np.array([m.epistemic_value() for m in models])
            self.last_epistemic = epistemic
        else:
            epistemic = np.zeros(self.n_models)
            self.last_epistemic = None

        scores = pragmatic + self.config.epistemic_weight * epistemic

        # Apply BIC-like complexity penalty: penalize models with more parameters
        if self.config.complexity_penalty_weight > 0:
            n_params = np.array([m.n_parameters for m in models])
            # Penalty: -0.5 * k * log(n) / n, where k = n_parameters, n = observations
            # This is the per-observation BIC penalty
            n_effective = max(self._n_obs, 10)  # Avoid log(0)
            penalty = -0.5 * n_params * np.log(n_effective) / n_effective
            scores += self.config.complexity_penalty_weight * penalty

        if self.config.use_adaptive_forgetting:
            best_ll = np.max(pragmatic)
            surprise = max(0, -best_ll - 1.0)

            surprise_decay = 0.95
            self._surprise_ema = (
                surprise_decay * self._surprise_ema + (1 - surprise_decay) * surprise
            )

            surprise_threshold = 2.0
            if self._surprise_ema > surprise_threshold:
                forget_reduction = min(0.1, (self._surprise_ema - surprise_threshold) * 0.05)
                self.current_forget = max(
                    self.config.min_forget, self.config.likelihood_forget - forget_reduction
                )
            else:
                recovery_rate = 0.05
                gap = self.config.likelihood_forget - self.current_forget
                self.current_forget = min(
                    self.config.likelihood_forget,
                    self.current_forget + recovery_rate * gap + 0.001,
                )

            forget = self.current_forget
        else:
            forget = self.config.likelihood_forget

        self.cumulative_scores = forget * self.cumulative_scores + scores

        for m in models:
            m.update(y, t)

    def combine_predictions(self, predictions: list[Prediction]) -> Prediction:
        """Combine predictions using law of total variance.

        Combined mean is weighted average of means.
        Combined variance includes within-model and between-model variance.

        Args:
            predictions: List of predictions from each model

        Returns:
            Combined prediction
        """
        weights = self.get_weights()

        means = np.array([p.mean for p in predictions])
        variances = np.array([p.variance for p in predictions])

        # Clamp means to prevent overflow when computing squared differences
        max_mean = 1e6
        clamped_means = np.clip(means, -max_mean, max_mean)

        combined_mean = np.sum(weights * clamped_means)

        within_var = np.sum(weights * variances)

        between_var = np.sum(weights * (clamped_means - combined_mean) ** 2)

        combined_variance = within_var + between_var

        return Prediction(mean=combined_mean, variance=np.clip(combined_variance, 1e-10, 1e10))

    def reset_scores(self, partial: float = 1.0) -> None:
        """Reset cumulative scores.

        Args:
            partial: Interpolation weight (1.0 = full reset to uniform)
        """
        self.cumulative_scores = (1 - partial) * self.cumulative_scores
