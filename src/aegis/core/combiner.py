"""Model combiner for AEGIS.

Implements Expected Free Energy (EFE) based model weighting
using softmax over cumulative log-likelihood scores.
"""

import numpy as np

from aegis.config import AEGISConfig
from aegis.core.prediction import Prediction
from aegis.core.prediction_buffer import PredictionBuffer
from aegis.models.base import TemporalModel


def compute_multi_horizon_likelihood(
    y: float,
    predictions: dict[int, tuple[float, float]],
) -> float:
    """Compute weighted log-likelihood across multiple horizons.

    Args:
        y: Observed value
        predictions: Dict mapping horizon -> (mean, variance)

    Returns:
        Average log-likelihood across available horizons
    """
    if not predictions:
        return 0.0

    total_ll = 0.0
    count = 0

    for horizon, (mean, variance) in predictions.items():
        # Gaussian log-likelihood
        var_safe = max(variance, 1e-10)
        ll = -0.5 * np.log(2 * np.pi * var_safe) - 0.5 * (y - mean) ** 2 / var_safe
        total_ll += ll
        count += 1

    return total_ll / count if count > 0 else 0.0


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

        # Multi-horizon scoring: store predictions for future evaluation
        scoring_horizons = getattr(config, "scoring_horizons", [1, 4, 16])
        self.scoring_horizons: list[int] = scoring_horizons
        self.prediction_buffer: PredictionBuffer = PredictionBuffer(
            horizons=scoring_horizons,
            n_models=n_models,
            max_history=max(scoring_horizons) + 10,
        )
        self.last_multi_horizon_scores: np.ndarray | None = None

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

        # --- Multi-horizon scoring: evaluate past predictions ---
        multi_horizon_scores = np.zeros(self.n_models)
        max_horizon = max(self.scoring_horizons)

        if t >= max_horizon:
            # Score each model on how well past predictions match current observation
            for model_idx in range(self.n_models):
                predictions: dict[int, tuple[float, float]] = {}
                for h in self.scoring_horizons:
                    pred = self.prediction_buffer.get_for_scoring(
                        model_idx=model_idx, horizon=h, current_t=t
                    )
                    if pred is not None:
                        predictions[h] = pred

                if predictions:
                    multi_horizon_scores[model_idx] = compute_multi_horizon_likelihood(
                        y=y, predictions=predictions
                    )

            self.last_multi_horizon_scores = multi_horizon_scores

            # Normalize multi-horizon scores
            if len(multi_horizon_scores) > 1:
                std = np.std(multi_horizon_scores)
                if std > 1e-10:
                    multi_horizon_scores = (
                        multi_horizon_scores - np.mean(multi_horizon_scores)
                    ) / std

        # --- Current 1-step pragmatic scoring (kept for cold start) ---
        raw_pragmatic = np.array([m.log_likelihood(y) for m in models])
        self.last_pragmatic = raw_pragmatic

        # Z-score normalize log-likelihoods
        if len(raw_pragmatic) > 1:
            std = np.std(raw_pragmatic)
            if std > 1e-10:
                pragmatic = (raw_pragmatic - np.mean(raw_pragmatic)) / std
            else:
                pragmatic = raw_pragmatic
        else:
            pragmatic = raw_pragmatic

        # --- Combine multi-horizon and current pragmatic ---
        if t >= max_horizon:
            # Use multi-horizon as primary scoring once available
            # Multi-horizon captures long-term prediction ability
            # Pragmatic (h=1) is included but weighted less to avoid variance bias
            scores = 0.8 * multi_horizon_scores + 0.2 * pragmatic
        else:
            # Cold start: use only current pragmatic
            scores = pragmatic

        if self.config.use_epistemic_value:
            epistemic = np.array([m.epistemic_value() for m in models])
            self.last_epistemic = epistemic
        else:
            epistemic = np.zeros(self.n_models)
            self.last_epistemic = None

        scores = scores + self.config.epistemic_weight * epistemic

        # Apply BIC-like complexity penalty
        if self.config.complexity_penalty_weight > 0:
            n_params = np.array([m.n_parameters for m in models])
            n_effective = max(self._n_obs, 10)
            penalty = -0.5 * n_params * np.log(n_effective) / n_effective
            scores += self.config.complexity_penalty_weight * penalty

        if self.config.use_adaptive_forgetting:
            best_ll = np.max(raw_pragmatic)
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

        # --- Update models ---
        for m in models:
            m.update(y, t)

        # --- Store predictions for future multi-horizon scoring ---
        for model_idx, m in enumerate(models):
            for h in self.scoring_horizons:
                pred = m.predict(horizon=h)
                self.prediction_buffer.store(
                    model_idx=model_idx,
                    horizon=h,
                    t=t,
                    mean=pred.mean,
                    variance=pred.variance,
                )

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

    def get_score_breakdown(self) -> dict:
        """Get detailed score breakdown for debugging.

        Returns:
            Dictionary with cumulative scores, last log-likelihoods,
            current weights, and observation count.
        """
        max_horizon = max(self.scoring_horizons)
        return {
            "cumulative_scores": self.cumulative_scores.tolist(),
            "last_log_likelihoods": self.last_pragmatic.tolist(),
            "last_epistemic": self.last_epistemic.tolist()
            if self.last_epistemic is not None
            else None,
            "current_weights": self.get_weights().tolist(),
            "n_observations": self._n_obs,
            "current_forget": self.current_forget,
            "surprise_ema": self._surprise_ema,
            "multi_horizon_available": self._n_obs >= max_horizon,
            "scoring_horizons": self.scoring_horizons,
            "last_multi_horizon_scores": self.last_multi_horizon_scores.tolist()
            if self.last_multi_horizon_scores is not None
            else None,
        }
