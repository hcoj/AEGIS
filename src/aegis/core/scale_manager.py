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

        Args:
            horizon: Steps ahead to predict

        Returns:
            Combined prediction in levels
        """
        if len(self.history) < 2:
            return Prediction(mean=0.0, variance=1.0)

        predictions: list[Prediction] = []
        weights: list[float] = []
        active_scales: list[int] = []

        for i, scale in enumerate(self.scales):
            if len(self.history) > scale:
                models = self.scale_models[scale]
                combiner = self.scale_combiners[scale]

                model_preds = [m.predict(horizon) for m in models]
                scale_pred = combiner.combine_predictions(model_preds)

                predictions.append(scale_pred)
                weights.append(self.scale_weights[i])
                active_scales.append(scale)

        if not predictions:
            return Prediction(mean=0.0, variance=1.0)

        weights_arr = np.array(weights)
        weights_arr /= weights_arr.sum()

        # Models now return cumulative change over horizon.
        # Scale s models predict cumulative s-period returns; divide by s to get level change.
        level_changes = np.array([p.mean / s for p, s in zip(predictions, active_scales)])
        variances = np.array([p.variance for p in predictions])

        combined_level_change = np.sum(weights_arr * level_changes)
        within_var = np.sum(weights_arr * variances)
        between_var = np.sum(weights_arr * (level_changes - combined_level_change) ** 2)

        level_mean = self.history[-1] + combined_level_change
        level_var = within_var + between_var

        return Prediction(mean=level_mean, variance=max(level_var, self.config.min_variance))

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
            result["per_scale"][scale] = {
                "weights": combiner.get_weights(),
                "cumulative_scores": combiner.cumulative_scores.copy(),
            }

        return result
