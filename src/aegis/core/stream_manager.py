"""Stream manager for AEGIS.

Manages all processing for a single data stream including
multi-scale processing, break detection, and uncertainty calibration.
"""

from collections.abc import Callable

import numpy as np

from aegis.config import AEGISConfig
from aegis.core.break_detector import CUSUMBreakDetector
from aegis.core.prediction import Prediction
from aegis.core.quantile_tracker import HorizonAwareQuantileTracker
from aegis.core.scale_manager import ScaleManager
from aegis.models.base import TemporalModel


class StreamManager:
    """Manages all processing for a single data stream.

    Integrates multi-scale processing, break detection, and
    uncertainty calibration.

    Attributes:
        name: Stream identifier
        config: AEGIS configuration
        scale_manager: Multi-scale processor
        break_detector: Regime break detector
        quantile_tracker: Uncertainty calibration
    """

    def __init__(
        self,
        name: str,
        config: AEGISConfig,
        model_factory: Callable[[], list[TemporalModel]],
    ) -> None:
        """Initialize StreamManager.

        Args:
            name: Stream identifier
            config: AEGIS configuration
            model_factory: Callable that returns a list of TemporalModels
        """
        self.name: str = name
        self.config: AEGISConfig = config

        self.scale_manager: ScaleManager = ScaleManager(config, model_factory)

        self.break_detector: CUSUMBreakDetector = CUSUMBreakDetector(
            threshold=config.break_threshold
        )

        self.quantile_tracker: HorizonAwareQuantileTracker = HorizonAwareQuantileTracker(
            target_coverage=config.target_coverage
        )

        self.volatility: float = 1.0
        self.long_run_vol: float = 1.0

        self.t: int = 0
        self.last_prediction: Prediction | None = None
        self.last_h1_prediction: Prediction | None = None
        self.in_break_adaptation: bool = False
        self.break_countdown: int = 0

    def observe(self, y: float, t: int | None = None) -> None:
        """Process new observation.

        Args:
            y: Observed value
            t: Optional time index
        """
        if t is not None:
            self.t = t

        if self.last_h1_prediction is not None:
            error = y - self.last_h1_prediction.mean
            std = self.last_h1_prediction.std

            self.volatility = (
                self.config.volatility_decay * self.volatility
                + (1 - self.config.volatility_decay) * error**2
            )
            self.long_run_vol = 0.999 * self.long_run_vol + 0.001 * error**2

            if std > 0:
                self.quantile_tracker.update(y, self.last_h1_prediction.mean, std, horizon=1)

            if self.break_detector.update(error):
                self._handle_break()

        self.scale_manager.observe(y)
        self.scale_manager.update_scale_weights(y)

        if self.break_countdown > 0:
            self.break_countdown -= 1
            if self.break_countdown == 0:
                self.in_break_adaptation = False

        self.t += 1

    def predict(self, horizon: int = 1) -> Prediction:
        """Generate prediction with calibrated uncertainty.

        Args:
            horizon: Steps ahead to predict

        Returns:
            Prediction with calibrated intervals
        """
        pred = self.scale_manager.predict(horizon)

        vol_ratio = np.sqrt(self.volatility / (self.long_run_vol + 1e-10))
        scaled_var = pred.variance * max(vol_ratio, 0.1)

        if self.config.use_quantile_calibration:
            std = np.sqrt(scaled_var)
            q_low, q_high = self.quantile_tracker.get_quantiles(horizon)
            interval_lower = pred.mean + q_low * std
            interval_upper = pred.mean + q_high * std
        else:
            interval_lower = None
            interval_upper = None

        self.last_prediction = Prediction(
            mean=pred.mean,
            variance=scaled_var,
            interval_lower=interval_lower,
            interval_upper=interval_upper,
        )

        if horizon == 1:
            self.last_h1_prediction = self.last_prediction

        return self.last_prediction

    def _handle_break(self) -> None:
        """Handle detected regime break."""
        self.scale_manager.trigger_break_adaptation()
        self.break_detector.reset()
        self.in_break_adaptation = True
        self.break_countdown = self.config.post_break_duration

    def get_diagnostics(self) -> dict:
        """Get diagnostic information.

        Returns:
            Dictionary with model weights, volatility, and break status
        """
        scale_diag = self.scale_manager.get_diagnostics()

        model_weights: dict[str, float] = {}
        group_weights: dict[str, float] = {}

        for scale, data in scale_diag["per_scale"].items():
            models = self.scale_manager.scale_models[scale]
            weights = data["weights"]

            for i, model in enumerate(models):
                name = f"{model.name}_s{scale}"
                model_weights[name] = float(weights[i])

                group = model.group
                if group not in group_weights:
                    group_weights[group] = 0.0
                group_weights[group] += float(weights[i])

        total = sum(group_weights.values())
        if total > 0:
            group_weights = {k: v / total for k, v in group_weights.items()}

        per_scale_summary: dict = {}
        for scale, data in scale_diag["per_scale"].items():
            model_names = data.get("model_names", [])
            weights = data["weights"]
            score_breakdown = data.get("score_breakdown", {})
            top_3_idx = np.argsort(weights)[::-1][:3]
            top_3 = [(model_names[i], float(weights[i])) for i in top_3_idx if i < len(model_names)]
            per_scale_summary[scale] = {
                "top_models": top_3,
                "n_observations": score_breakdown.get("n_observations", 0),
                "surprise_ema": score_breakdown.get("surprise_ema", 0.0),
            }

        return {
            "model_weights": model_weights,
            "group_weights": group_weights,
            "top_models": sorted(model_weights.items(), key=lambda x: -x[1])[:10],
            "scale_weights": scale_diag["scale_weights"],
            "per_scale_summary": per_scale_summary,
            "volatility": self.volatility,
            "in_break_adaptation": self.in_break_adaptation,
            "quantile_multipliers": self.quantile_tracker.get_quantiles(horizon=1),
        }
