"""Scale manager for AEGIS.

Manages multi-scale processing for a single stream,
computing returns at multiple lookback periods.
"""

from collections.abc import Callable

import numpy as np

from aegis.config import AEGISConfig
from aegis.core.combiner import EFEModelCombiner
from aegis.core.prediction import Prediction
from aegis.models.base import TemporalModel


class ScaleManager:
    """Manages multi-scale processing for a single stream.

    Computes returns at multiple lookback periods and maintains
    per-scale model banks.

    Attributes:
        config: AEGIS configuration
        scales: List of lookback scales
        history: Raw observation buffer
        scale_models: Per-scale model banks
        scale_combiners: Per-scale model combiners
    """

    def __init__(
        self,
        config: AEGISConfig,
        model_factory: Callable[[], list[TemporalModel]],
    ) -> None:
        """Initialize ScaleManager.

        Args:
            config: System configuration
            model_factory: Callable that returns a list of TemporalModels
        """
        self.config: AEGISConfig = config
        self.scales: list[int] = list(config.scales)
        self.max_scale: int = max(self.scales)

        self.history: list[float] = []

        self.scale_models: dict[int, list[TemporalModel]] = {}
        self.scale_combiners: dict[int, EFEModelCombiner] = {}

        for scale in self.scales:
            models = model_factory()
            self.scale_models[scale] = models
            self.scale_combiners[scale] = EFEModelCombiner(n_models=len(models), config=config)

        self.scale_weights: np.ndarray = np.ones(len(self.scales)) / len(self.scales)
        self.scale_errors: dict[int, float] = {s: 1.0 for s in self.scales}

        self.t: int = 0

    def observe(self, y: float) -> None:
        """Process new observation.

        Args:
            y: Observed value
        """
        self.history.append(y)

        if len(self.history) > self.max_scale + 10:
            self.history = self.history[-(self.max_scale + 10) :]

        for scale in self.scales:
            if len(self.history) > scale:
                r = self.history[-1] - self.history[-1 - scale]

                models = self.scale_models[scale]
                combiner = self.scale_combiners[scale]
                combiner.update(models, r, self.t)

        self.t += 1

    def predict(self, horizon: int) -> Prediction:
        """Generate combined prediction across all scales.

        Uses horizon-aware filtering: only scales <= horizon contribute to the mean
        (to avoid interpolation errors), but all scales contribute to variance
        (to preserve calibration).

        Args:
            horizon: Steps ahead to predict

        Returns:
            Combined prediction in levels
        """
        if len(self.history) < 2:
            return Prediction(mean=0.0, variance=1.0)

        # Collect predictions from all available scales
        all_predictions: list[Prediction] = []
        all_weights: list[float] = []
        all_scales: list[int] = []

        # Also track which are valid for mean estimation (scale <= horizon)
        mean_predictions: list[Prediction] = []
        mean_weights: list[float] = []
        mean_scales: list[int] = []

        for i, scale in enumerate(self.scales):
            if len(self.history) > scale:
                models = self.scale_models[scale]
                combiner = self.scale_combiners[scale]

                model_preds = [m.predict(horizon) for m in models]
                scale_pred = combiner.combine_predictions(model_preds)

                all_predictions.append(scale_pred)
                all_weights.append(self.scale_weights[i])
                all_scales.append(scale)

                # Only use scales <= horizon for mean (avoid interpolation errors)
                if scale <= horizon:
                    mean_predictions.append(scale_pred)
                    mean_weights.append(self.scale_weights[i])
                    mean_scales.append(scale)

        if not mean_predictions:
            return Prediction(mean=0.0, variance=1.0)

        # Compute mean using only horizon-appropriate scales
        # Use inverse-variance weighting: scales with high uncertainty get less weight
        mean_variances = np.array([p.variance for p in mean_predictions])
        inv_var_weights = 1.0 / (mean_variances + 1e-10)
        # Combine with scale_weights (based on historical h=1 accuracy)
        mean_weights_arr = np.array(mean_weights) * inv_var_weights
        mean_weights_arr /= mean_weights_arr.sum()

        mean_level_changes = np.array([p.mean / s for p, s in zip(mean_predictions, mean_scales)])
        max_level_change = 1e6
        clamped_mean_changes = np.clip(mean_level_changes, -max_level_change, max_level_change)
        combined_level_change = np.sum(mean_weights_arr * clamped_mean_changes)

        # Compute variance using ALL scales (preserves calibration)
        all_weights_arr = np.array(all_weights)
        all_weights_arr /= all_weights_arr.sum()

        all_level_changes = np.array([p.mean / s for p, s in zip(all_predictions, all_scales)])
        all_level_variances = np.array([p.variance for p in all_predictions])
        clamped_all_changes = np.clip(all_level_changes, -max_level_change, max_level_change)

        within_var = np.sum(all_weights_arr * all_level_variances)
        between_var = np.sum(all_weights_arr * (clamped_all_changes - combined_level_change) ** 2)

        level_mean = self.history[-1] + combined_level_change
        level_var = within_var + between_var

        return Prediction(
            mean=level_mean,
            variance=np.clip(level_var, self.config.min_variance, self.config.max_variance),
        )

    def predict_at_scale(self, scale: int, horizon: int) -> Prediction:
        """Get prediction from a specific scale.

        Args:
            scale: Scale to predict from
            horizon: Steps ahead

        Returns:
            Prediction from specified scale
        """
        if scale not in self.scale_models:
            return Prediction(mean=0.0, variance=1.0)

        models = self.scale_models[scale]
        combiner = self.scale_combiners[scale]

        model_preds = [m.predict(horizon) for m in models]
        return combiner.combine_predictions(model_preds)

    def update_scale_weights(self, observed: float) -> None:
        """Update scale weights based on prediction accuracy.

        Args:
            observed: Observed value for error computation
        """
        decay = 0.95

        for i, scale in enumerate(self.scales):
            if len(self.history) > scale:
                pred = self.predict_at_scale(scale, horizon=1)
                expected_return = (
                    self.history[-1] - self.history[-1 - scale]
                    if len(self.history) > scale
                    else 0.0
                )
                error = (expected_return - pred.mean) ** 2

                self.scale_errors[scale] = decay * self.scale_errors[scale] + (1 - decay) * error

        errors = np.array([self.scale_errors[s] for s in self.scales])
        inv_errors = 1.0 / (errors + 1e-6)
        self.scale_weights = inv_errors / inv_errors.sum()

    def trigger_break_adaptation(self) -> None:
        """Reset models after detected break."""
        for scale in self.scales:
            for model in self.scale_models[scale]:
                model.reset(partial=0.5)
            self.scale_combiners[scale].reset_scores(partial=0.5)

    def get_horizon_scale_weights(self, horizon: int) -> np.ndarray:
        """Get horizon-specific scale weights.

        Weights scales based on proximity to the target horizon.
        Short horizons favor short scales, long horizons favor long scales.

        Args:
            horizon: Target prediction horizon

        Returns:
            Array of weights, one per scale, summing to 1.0
        """
        log_horizon = np.log(max(horizon, 1))
        log_scales = np.log(np.array(self.scales, dtype=float))

        distances = np.abs(log_scales - log_horizon)

        temperature = 1.0
        proximity_weights = np.exp(-distances / temperature)

        combined = proximity_weights * self.scale_weights
        return combined / combined.sum()

    def get_diagnostics(self) -> dict:
        """Get diagnostic information.

        Returns:
            Dictionary with scale weights and per-scale diagnostics
        """
        result: dict = {
            "scale_weights": self.scale_weights.copy(),
            "per_scale": {},
        }

        for scale in self.scales:
            combiner = self.scale_combiners[scale]
            models = self.scale_models[scale]
            model_names = [m.name for m in models]
            weights = combiner.get_weights()

            result["per_scale"][scale] = {
                "weights": weights,
                "cumulative_scores": combiner.cumulative_scores.copy(),
                "model_names": model_names,
                "score_breakdown": combiner.get_score_breakdown(),
            }

        return result
