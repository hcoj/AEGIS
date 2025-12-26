# AEGIS Appendix A: Core Implementation

## Concrete Implementations of Core Components

---

## Contents

1. [Data Structures](#1-data-structures)
2. [Model Combiner](#2-model-combiner)
3. [Scale Manager](#3-scale-manager)
4. [Stream Manager](#4-stream-manager)
5. [Cross-Stream Regression](#5-cross-stream-regression)
6. [Break Detection](#6-break-detection)
7. [Quantile Calibration](#7-quantile-calibration)
8. [Factor Model](#8-factor-model)
9. [Main System Class](#9-main-system-class)

---

## 1. Data Structures

### 1.1 Prediction

```python
from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy.stats import norm


@dataclass
class Prediction:
    """Point prediction with uncertainty."""
    mean: float
    variance: float
    
    # Optional calibrated interval (from quantile tracker)
    interval_lower: Optional[float] = None
    interval_upper: Optional[float] = None
    
    @property
    def std(self) -> float:
        return np.sqrt(self.variance)
    
    def interval(self, level: float = 0.95) -> tuple[float, float]:
        """
        Get prediction interval at specified confidence level.
        
        Uses calibrated interval if available, otherwise Gaussian.
        """
        if self.interval_lower is not None and self.interval_upper is not None:
            return self.interval_lower, self.interval_upper
        
        z = norm.ppf(0.5 + level / 2)
        return self.mean - z * self.std, self.mean + z * self.std
```

### 1.2 Configuration

```python
from dataclasses import dataclass, field
from typing import List


@dataclass
class AEGISConfig:
    """Configuration for AEGIS system."""
    
    # -------------------------------------------------------------------------
    # Phase selection
    # -------------------------------------------------------------------------
    use_epistemic_value: bool = False  # False for Phase 1, True for Phase 2
    epistemic_weight: float = 1.0      # α parameter for EFE
    
    # -------------------------------------------------------------------------
    # Multi-scale
    # -------------------------------------------------------------------------
    scales: List[int] = field(
        default_factory=lambda: [1, 2, 4, 8, 16, 32, 64]
    )
    
    # -------------------------------------------------------------------------
    # Model-specific
    # -------------------------------------------------------------------------
    oscillator_periods: List[int] = field(
        default_factory=lambda: [4, 8, 16, 32, 64, 128, 256]
    )
    seasonal_periods: List[int] = field(
        default_factory=lambda: [7, 12]
    )
    
    # -------------------------------------------------------------------------
    # Model combination
    # -------------------------------------------------------------------------
    likelihood_forget: float = 0.99
    temperature: float = 1.0
    
    # -------------------------------------------------------------------------
    # Volatility
    # -------------------------------------------------------------------------
    volatility_decay: float = 0.94
    
    # -------------------------------------------------------------------------
    # Cross-stream
    # -------------------------------------------------------------------------
    cross_stream_lags: int = 5
    include_lag_zero: bool = False
    cross_stream_forget: float = 0.99
    n_factors: int = 3
    
    # -------------------------------------------------------------------------
    # Regime adaptation
    # -------------------------------------------------------------------------
    break_threshold: float = 3.0
    post_break_forget: float = 0.9
    post_break_duration: int = 50
    
    # -------------------------------------------------------------------------
    # Calibration
    # -------------------------------------------------------------------------
    target_coverage: float = 0.95
    use_quantile_calibration: bool = True
    
    # -------------------------------------------------------------------------
    # Robustness
    # -------------------------------------------------------------------------
    outlier_threshold: float = 5.0
    min_variance: float = 1e-10
    max_epistemic_value: float = 10.0  # Cap to prevent domination
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        assert 0 < self.likelihood_forget <= 1.0
        assert self.temperature > 0
        assert self.epistemic_weight >= 0
        assert all(s > 0 for s in self.scales)
        assert 0 < self.target_coverage < 1
```

---

## 2. Model Combiner

### 2.1 EFE Model Combiner

```python
from abc import ABC, abstractmethod
from typing import List
import numpy as np

from aegis.models.base import TemporalModel, Prediction


class EFEModelCombiner:
    """
    Model combiner using Expected Free Energy.
    
    Combines pragmatic value (expected accuracy) with epistemic value
    (expected information gain) to weight models.
    """
    
    def __init__(
        self,
        n_models: int,
        config: AEGISConfig
    ):
        self.n_models = n_models
        self.config = config
        
        # Cumulative scores for each model
        self.cumulative_scores = np.zeros(n_models)
        
        # Track individual components for diagnostics
        self.last_pragmatic = np.zeros(n_models)
        self.last_epistemic = np.zeros(n_models)
        self.last_weights = np.ones(n_models) / n_models
    
    def update(
        self,
        models: List[TemporalModel],
        y_observed: float
    ) -> np.ndarray:
        """
        Update combination weights based on observed value.
        
        Args:
            models: List of temporal models
            y_observed: The observed value
            
        Returns:
            Updated weight vector
        """
        pragmatic = np.zeros(self.n_models)
        epistemic = np.zeros(self.n_models)
        
        for i, model in enumerate(models):
            # Compute log-likelihood (actual, not expected)
            pragmatic[i] = model.log_likelihood(y_observed)
            
            # Get epistemic value if Phase 2
            if self.config.use_epistemic_value:
                ev = model.epistemic_value()
                epistemic[i] = min(ev, self.config.max_epistemic_value)
            
            # Update model with observation
            model.update(y_observed, t=-1)  # t managed externally
        
        # Store for diagnostics
        self.last_pragmatic = pragmatic.copy()
        self.last_epistemic = epistemic.copy()
        
        # Update cumulative scores with forgetting
        alpha = self.config.epistemic_weight
        self.cumulative_scores *= self.config.likelihood_forget
        self.cumulative_scores += pragmatic + alpha * epistemic
        
        # Compute weights via softmax
        self.last_weights = self._softmax(self.cumulative_scores)
        
        return self.last_weights
    
    def _softmax(self, scores: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        scaled = scores / self.config.temperature
        scaled -= np.max(scaled)
        exp_scores = np.exp(scaled)
        return exp_scores / np.sum(exp_scores)
    
    def get_weights(self) -> np.ndarray:
        """Get current model weights."""
        return self.last_weights.copy()
    
    def combine_predictions(
        self,
        predictions: List[Prediction]
    ) -> Prediction:
        """
        Combine model predictions using current weights.
        
        Uses law of total variance to account for model disagreement.
        """
        weights = self.last_weights
        
        # Weighted mean
        means = np.array([p.mean for p in predictions])
        combined_mean = np.sum(weights * means)
        
        # Law of total variance
        variances = np.array([p.variance for p in predictions])
        within_var = np.sum(weights * variances)
        between_var = np.sum(weights * (means - combined_mean)**2)
        combined_var = within_var + between_var
        
        # Ensure minimum variance
        combined_var = max(combined_var, self.config.min_variance)
        
        return Prediction(mean=combined_mean, variance=combined_var)
    
    def reset_toward_uniform(self, strength: float = 0.5) -> None:
        """
        Reset weights toward uniform distribution.
        
        Used after regime breaks.
        """
        self.cumulative_scores *= (1 - strength)
    
    def get_diagnostics(self) -> dict:
        """Get diagnostic information."""
        return {
            "weights": self.last_weights.copy(),
            "pragmatic_values": self.last_pragmatic.copy(),
            "epistemic_values": self.last_epistemic.copy(),
            "cumulative_scores": self.cumulative_scores.copy()
        }
```

---

## 3. Scale Manager

```python
from typing import Dict, List, Optional
import numpy as np

from aegis.models.base import TemporalModel, Prediction


class ScaleManager:
    """
    Manages multi-scale processing for a single stream.
    
    Computes returns at multiple lookback periods and maintains
    per-scale model banks.
    """
    
    def __init__(
        self,
        config: AEGISConfig,
        model_factory: callable
    ):
        """
        Args:
            config: System configuration
            model_factory: Callable that returns a list of TemporalModels
        """
        self.config = config
        self.scales = config.scales
        self.max_scale = max(self.scales)
        
        # History buffer for computing multi-scale returns
        self.history: List[float] = []
        
        # Per-scale model banks and combiners
        self.scale_models: Dict[int, List[TemporalModel]] = {}
        self.scale_combiners: Dict[int, EFEModelCombiner] = {}
        
        for scale in self.scales:
            models = model_factory()
            self.scale_models[scale] = models
            self.scale_combiners[scale] = EFEModelCombiner(
                n_models=len(models),
                config=config
            )
        
        # Scale weights (learned from performance)
        self.scale_weights = np.ones(len(self.scales)) / len(self.scales)
        self.scale_errors: Dict[int, float] = {s: 1.0 for s in self.scales}
        
        self.t = 0
    
    def observe(self, y: float) -> None:
        """
        Process new observation.
        
        Updates history and all per-scale models.
        """
        self.history.append(y)
        
        # Trim history to necessary length
        if len(self.history) > self.max_scale + 10:
            self.history = self.history[-(self.max_scale + 10):]
        
        # Update each scale
        for scale in self.scales:
            if len(self.history) > scale:
                # Compute return at this scale
                r = self.history[-1] - self.history[-1 - scale]
                
                # Update models and combiner
                models = self.scale_models[scale]
                combiner = self.scale_combiners[scale]
                combiner.update(models, r)
        
        self.t += 1
    
    def predict(self, horizon: int) -> Prediction:
        """
        Generate combined prediction across all scales.
        """
        if len(self.history) < 2:
            return Prediction(mean=0.0, variance=1.0)
        
        predictions = []
        weights = []
        
        for i, scale in enumerate(self.scales):
            if len(self.history) > scale:
                # Get per-scale combined prediction
                models = self.scale_models[scale]
                combiner = self.scale_combiners[scale]
                
                model_preds = [m.predict(horizon) for m in models]
                scale_pred = combiner.combine_predictions(model_preds)
                
                predictions.append(scale_pred)
                weights.append(self.scale_weights[i])
        
        if not predictions:
            return Prediction(mean=0.0, variance=1.0)
        
        # Combine across scales
        weights = np.array(weights)
        weights /= weights.sum()
        
        means = np.array([p.mean for p in predictions])
        variances = np.array([p.variance for p in predictions])
        
        combined_mean = np.sum(weights * means)
        within_var = np.sum(weights * variances)
        between_var = np.sum(weights * (means - combined_mean)**2)
        
        # Convert return prediction to level prediction
        level_mean = self.history[-1] + combined_mean
        level_var = within_var + between_var
        
        return Prediction(mean=level_mean, variance=max(level_var, self.config.min_variance))
    
    def update_scale_weights(self, observed: float) -> None:
        """
        Update scale weights based on prediction accuracy.
        """
        decay = 0.95
        
        for i, scale in enumerate(self.scales):
            if len(self.history) > scale:
                # Get prediction that was made
                pred = self.predict_at_scale(scale, horizon=1)
                error = (observed - pred.mean)**2
                
                self.scale_errors[scale] = decay * self.scale_errors[scale] + (1 - decay) * error
        
        # Convert errors to weights (lower error = higher weight)
        errors = np.array([self.scale_errors[s] for s in self.scales])
        inv_errors = 1.0 / (errors + 1e-6)
        self.scale_weights = inv_errors / inv_errors.sum()
    
    def predict_at_scale(self, scale: int, horizon: int) -> Prediction:
        """Get prediction from a specific scale."""
        if scale not in self.scale_models:
            return Prediction(mean=0.0, variance=1.0)
        
        models = self.scale_models[scale]
        combiner = self.scale_combiners[scale]
        
        model_preds = [m.predict(horizon) for m in models]
        return combiner.combine_predictions(model_preds)
    
    def trigger_break_adaptation(self) -> None:
        """Reset models after detected break."""
        for scale in self.scales:
            for model in self.scale_models[scale]:
                model.reset(partial=0.5)
            self.scale_combiners[scale].reset_toward_uniform(strength=0.5)
    
    def get_diagnostics(self) -> dict:
        """Get diagnostic information."""
        result = {
            "scale_weights": self.scale_weights.copy(),
            "per_scale": {}
        }
        
        for scale in self.scales:
            combiner = self.scale_combiners[scale]
            result["per_scale"][scale] = combiner.get_diagnostics()
        
        return result
```

---

## 4. Stream Manager

```python
from typing import Optional
import numpy as np

from aegis.models.base import Prediction


class StreamManager:
    """
    Manages all processing for a single data stream.
    
    Integrates multi-scale processing, break detection, and
    uncertainty calibration.
    """
    
    def __init__(
        self,
        name: str,
        config: AEGISConfig,
        model_factory: callable
    ):
        self.name = name
        self.config = config
        
        # Multi-scale processing
        self.scale_manager = ScaleManager(config, model_factory)
        
        # Break detection
        self.break_detector = CUSUMBreakDetector(
            threshold=config.break_threshold
        )
        
        # Uncertainty calibration
        self.quantile_tracker = QuantileTracker(
            target_coverage=config.target_coverage
        )
        
        # Volatility tracking
        self.volatility = 1.0
        self.long_run_vol = 1.0
        
        # State
        self.t = 0
        self.last_prediction: Optional[Prediction] = None
        self.in_break_adaptation = False
        self.break_countdown = 0
    
    def observe(self, y: float, t: Optional[int] = None) -> None:
        """
        Process new observation.
        """
        if t is not None:
            self.t = t
        
        # Check prediction error if we made a prediction
        if self.last_prediction is not None:
            error = y - self.last_prediction.mean
            std = self.last_prediction.std
            
            # Update volatility
            self.volatility = (
                self.config.volatility_decay * self.volatility +
                (1 - self.config.volatility_decay) * error**2
            )
            self.long_run_vol = 0.999 * self.long_run_vol + 0.001 * error**2
            
            # Update quantile tracker
            if std > 0:
                self.quantile_tracker.update(error / std)
            
            # Check for break
            if self.break_detector.update(error):
                self._handle_break()
        
        # Update scale manager
        self.scale_manager.observe(y)
        self.scale_manager.update_scale_weights(y)
        
        # Manage break adaptation period
        if self.break_countdown > 0:
            self.break_countdown -= 1
            if self.break_countdown == 0:
                self.in_break_adaptation = False
        
        self.t += 1
    
    def predict(self, horizon: int = 1) -> Prediction:
        """
        Generate prediction with calibrated uncertainty.
        """
        # Get base prediction from scale manager
        pred = self.scale_manager.predict(horizon)
        
        # Apply volatility scaling
        vol_ratio = np.sqrt(self.volatility / (self.long_run_vol + 1e-10))
        scaled_var = pred.variance * vol_ratio
        
        # Apply quantile calibration
        if self.config.use_quantile_calibration:
            std = np.sqrt(scaled_var)
            q_low, q_high = self.quantile_tracker.get_interval_multipliers()
            interval_lower = pred.mean + q_low * std
            interval_upper = pred.mean + q_high * std
        else:
            interval_lower = None
            interval_upper = None
        
        self.last_prediction = Prediction(
            mean=pred.mean,
            variance=scaled_var,
            interval_lower=interval_lower,
            interval_upper=interval_upper
        )
        
        return self.last_prediction
    
    def _handle_break(self) -> None:
        """Handle detected regime break."""
        self.scale_manager.trigger_break_adaptation()
        self.break_detector.reset()
        self.in_break_adaptation = True
        self.break_countdown = self.config.post_break_duration
    
    def get_diagnostics(self) -> dict:
        """Get diagnostic information."""
        scale_diag = self.scale_manager.get_diagnostics()
        
        # Aggregate model weights across scales
        model_weights = {}
        group_weights = {}
        epistemic_values = {}
        
        for scale, data in scale_diag["per_scale"].items():
            models = self.scale_manager.scale_models[scale]
            weights = data["weights"]
            ev = data["epistemic_values"]
            
            for i, model in enumerate(models):
                name = f"{model.name}_s{scale}"
                model_weights[name] = weights[i]
                
                group = model.group
                if group not in group_weights:
                    group_weights[group] = 0.0
                group_weights[group] += weights[i]
                
                if self.config.use_epistemic_value:
                    epistemic_values[name] = ev[i]
        
        # Normalise group weights
        total = sum(group_weights.values())
        if total > 0:
            group_weights = {k: v / total for k, v in group_weights.items()}
        
        result = {
            "model_weights": model_weights,
            "group_weights": group_weights,
            "top_models": sorted(model_weights.items(), key=lambda x: -x[1])[:10],
            "scale_weights": scale_diag["scale_weights"],
            "volatility": self.volatility,
            "in_break_adaptation": self.in_break_adaptation,
            "quantile_multipliers": self.quantile_tracker.get_interval_multipliers()
        }
        
        if self.config.use_epistemic_value:
            result["epistemic_values"] = epistemic_values
        
        return result
```

---

## 5. Cross-Stream Regression

```python
from typing import Dict, List, Optional
import numpy as np


class CrossStreamRegression:
    """
    Captures relationships between multiple streams via residual regression.
    
    Operates on prediction residuals to avoid double-counting temporal structure.
    """
    
    def __init__(
        self,
        stream_names: List[str],
        config: AEGISConfig
    ):
        self.stream_names = stream_names
        self.n_streams = len(stream_names)
        self.config = config
        self.lags = config.cross_stream_lags
        self.include_lag_zero = config.include_lag_zero
        
        # Number of features per predictor stream
        n_lags = self.lags + (1 if self.include_lag_zero else 0)
        self.n_features = (self.n_streams - 1) * n_lags
        
        # Regression coefficients for each target stream
        # coefficients[target] = array of shape (n_features,)
        self.coefficients: Dict[str, np.ndarray] = {
            name: np.zeros(self.n_features)
            for name in stream_names
        }
        
        # RLS covariance matrices
        self.P: Dict[str, np.ndarray] = {
            name: np.eye(self.n_features) * 100.0
            for name in stream_names
        }
        
        # History of residuals
        self.residual_history: Dict[str, List[float]] = {
            name: [] for name in stream_names
        }
        
        # Current residuals (for lag-0)
        self.current_residuals: Dict[str, Optional[float]] = {
            name: None for name in stream_names
        }
    
    def update(
        self,
        stream_name: str,
        residual: float,
        observed_order: List[str]
    ) -> float:
        """
        Update regression and return adjusted prediction.
        
        Args:
            stream_name: Name of stream being updated
            residual: Prediction residual (observed - predicted)
            observed_order: Order in which streams were observed this period
            
        Returns:
            Cross-stream adjustment to apply to prediction
        """
        # Store current residual
        self.current_residuals[stream_name] = residual
        
        # Build feature vector
        features = self._build_features(stream_name, observed_order)
        
        if features is not None and len(self.residual_history[stream_name]) > self.lags:
            # RLS update
            beta = self.coefficients[stream_name]
            P = self.P[stream_name]
            
            # Prediction
            pred = np.dot(features, beta)
            error = residual - pred
            
            # Update (RLS with forgetting)
            forget = self.config.cross_stream_forget
            Pf = P @ features
            denom = forget + np.dot(features, Pf)
            gain = Pf / denom
            
            self.coefficients[stream_name] = beta + gain * error
            self.P[stream_name] = (P - np.outer(gain, Pf)) / forget
        else:
            pred = 0.0
        
        # Update history
        self.residual_history[stream_name].append(residual)
        if len(self.residual_history[stream_name]) > self.lags + 10:
            self.residual_history[stream_name] = \
                self.residual_history[stream_name][-(self.lags + 10):]
        
        return pred
    
    def _build_features(
        self,
        target: str,
        observed_order: List[str]
    ) -> Optional[np.ndarray]:
        """Build feature vector for prediction."""
        features = []
        
        for source in self.stream_names:
            if source == target:
                continue
            
            hist = self.residual_history[source]
            
            # Lagged features
            for lag in range(1, self.lags + 1):
                if len(hist) >= lag:
                    features.append(hist[-lag])
                else:
                    features.append(0.0)
            
            # Lag-0 (contemporaneous)
            if self.include_lag_zero:
                # Only include if source was observed before target this period
                if source in observed_order:
                    source_idx = observed_order.index(source)
                    target_idx = observed_order.index(target) if target in observed_order else len(observed_order)
                    
                    if source_idx < target_idx and self.current_residuals[source] is not None:
                        features.append(self.current_residuals[source])
                    else:
                        features.append(0.0)
                else:
                    features.append(0.0)
        
        if len(features) == 0:
            return None
        
        return np.array(features)
    
    def end_period(self) -> None:
        """Called at end of each time period to reset current residuals."""
        for name in self.stream_names:
            self.current_residuals[name] = None
    
    def get_diagnostics(self) -> dict:
        """Get diagnostic information."""
        return {
            "coefficients": {
                name: coef.copy()
                for name, coef in self.coefficients.items()
            }
        }
```

---

## 6. Break Detection

```python
import numpy as np


class CUSUMBreakDetector:
    """
    CUSUM-based regime break detection.
    
    Accumulates prediction errors and triggers when cumulative
    sum exceeds threshold.
    """
    
    def __init__(
        self,
        threshold: float = 3.0,
        drift: float = 1.5
    ):
        """
        Args:
            threshold: Number of standard deviations for break signal
            drift: Allowance subtracted each step (in std devs)
        """
        self.threshold = threshold
        self.drift = drift
        
        # Running statistics
        self.cusum_pos = 0.0
        self.cusum_neg = 0.0
        self.sigma = 1.0
        self.n_obs = 0
    
    def update(self, error: float) -> bool:
        """
        Update detector with prediction error.
        
        Returns True if break detected.
        """
        self.n_obs += 1
        
        # Update volatility estimate
        self.sigma = 0.95 * self.sigma + 0.05 * abs(error)
        
        # Standardise error
        z = error / (self.sigma + 1e-10)
        
        # Update CUSUMs
        self.cusum_pos = max(0, self.cusum_pos + z - self.drift)
        self.cusum_neg = max(0, self.cusum_neg - z - self.drift)
        
        # Check for break
        if self.cusum_pos > self.threshold or self.cusum_neg > self.threshold:
            return True
        
        return False
    
    def reset(self) -> None:
        """Reset detector state."""
        self.cusum_pos = 0.0
        self.cusum_neg = 0.0
    
    def get_state(self) -> dict:
        """Get current detector state."""
        return {
            "cusum_pos": self.cusum_pos,
            "cusum_neg": self.cusum_neg,
            "sigma": self.sigma
        }
```

---

## 7. Quantile Calibration

```python
import numpy as np


class QuantileTracker:
    """
    Tracks empirical quantiles for calibrated prediction intervals.
    
    Adjusts interval widths based on actual coverage, correcting
    for non-Gaussian error distributions.
    """
    
    def __init__(
        self,
        target_coverage: float = 0.95,
        learning_rate: float = 0.02
    ):
        self.target_coverage = target_coverage
        self.alpha_low = (1 - target_coverage) / 2   # e.g., 0.025
        self.alpha_high = 1 - self.alpha_low          # e.g., 0.975
        self.lr = learning_rate
        
        # Quantile estimates (in units of standard deviations)
        # Initialise at Gaussian values
        from scipy.stats import norm
        self.q_low = norm.ppf(self.alpha_low)    # ~ -1.96
        self.q_high = norm.ppf(self.alpha_high)  # ~ +1.96
        self.median = 0.0
        
        self.n_obs = 0
    
    def update(self, standardised_error: float) -> None:
        """
        Update quantile estimates with new standardised error.
        
        Args:
            standardised_error: (observed - predicted) / predicted_std
        """
        self.n_obs += 1
        z = standardised_error
        
        # Quantile regression update
        # If observation below quantile, we need to lower the quantile
        if z < self.q_low:
            self.q_low -= self.lr * (1 - self.alpha_low)
        else:
            self.q_low += self.lr * self.alpha_low
        
        if z < self.q_high:
            self.q_high -= self.lr * (1 - self.alpha_high)
        else:
            self.q_high += self.lr * self.alpha_high
        
        # Median
        if z < self.median:
            self.median -= self.lr * 0.5
        else:
            self.median += self.lr * 0.5
        
        # Ensure ordering
        self.q_low = min(self.q_low, self.median - 0.1)
        self.q_high = max(self.q_high, self.median + 0.1)
    
    def get_interval_multipliers(self) -> tuple[float, float]:
        """
        Get multipliers for prediction standard deviation.
        
        Returns (lower_mult, upper_mult) such that:
            interval = [mean + lower_mult * std, mean + upper_mult * std]
        """
        return self.q_low, self.q_high
    
    def get_diagnostics(self) -> dict:
        """Get current state."""
        return {
            "q_low": self.q_low,
            "q_high": self.q_high,
            "median": self.median,
            "n_obs": self.n_obs,
            "interval_width": self.q_high - self.q_low
        }
```

---

## 8. Factor Model

```python
from typing import List, Optional
import numpy as np


class OnlinePCA:
    """
    Online PCA for factor extraction from multiple streams.
    
    Uses incremental SVD to maintain factor loadings without
    storing full history.
    """
    
    def __init__(
        self,
        n_streams: int,
        n_factors: int = 3,
        forget: float = 0.99
    ):
        self.n_streams = n_streams
        self.n_factors = min(n_factors, n_streams)
        self.forget = forget
        
        # Factor loadings (n_streams x n_factors)
        self.loadings = np.random.randn(n_streams, self.n_factors) * 0.1
        
        # Running covariance estimate
        self.cov = np.eye(n_streams)
        self.mean = np.zeros(n_streams)
        
        self.n_obs = 0
    
    def update(self, observations: np.ndarray) -> None:
        """
        Update factor model with new observations.
        
        Args:
            observations: Array of shape (n_streams,)
        """
        self.n_obs += 1
        x = observations
        
        # Update mean
        self.mean = self.forget * self.mean + (1 - self.forget) * x
        
        # Update covariance
        centered = x - self.mean
        self.cov = self.forget * self.cov + (1 - self.forget) * np.outer(centered, centered)
        
        # Periodically update loadings via eigendecomposition
        if self.n_obs % 50 == 0:
            self._update_loadings()
    
    def _update_loadings(self) -> None:
        """Update factor loadings from current covariance."""
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(self.cov)
            # Sort by eigenvalue descending
            idx = np.argsort(eigenvalues)[::-1]
            self.loadings = eigenvectors[:, idx[:self.n_factors]]
        except np.linalg.LinAlgError:
            pass  # Keep existing loadings if decomposition fails
    
    def get_factors(self, observations: np.ndarray) -> np.ndarray:
        """
        Project observations onto factors.
        
        Args:
            observations: Array of shape (n_streams,)
            
        Returns:
            Factor scores of shape (n_factors,)
        """
        centered = observations - self.mean
        return self.loadings.T @ centered
    
    def reconstruct(self, factors: np.ndarray) -> np.ndarray:
        """
        Reconstruct observations from factors.
        
        Args:
            factors: Array of shape (n_factors,)
            
        Returns:
            Reconstructed observations of shape (n_streams,)
        """
        return self.loadings @ factors + self.mean
    
    def get_diagnostics(self) -> dict:
        """Get diagnostic information."""
        eigenvalues = np.linalg.eigvalsh(self.cov)
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        return {
            "loadings": self.loadings.copy(),
            "explained_variance": eigenvalues[:self.n_factors],
            "total_variance": eigenvalues.sum()
        }
```

---

## 9. Main System Class

```python
from typing import Dict, Optional, List
import numpy as np


class AEGIS:
    """
    Main AEGIS system class.
    
    Manages multiple streams with cross-stream integration.
    """
    
    def __init__(
        self,
        config: Optional[AEGISConfig] = None,
        model_factory: Optional[callable] = None
    ):
        self.config = config or AEGISConfig()
        self.config.validate()
        
        # Default model factory creates standard model bank
        if model_factory is None:
            from aegis.models import create_model_bank
            model_factory = lambda: create_model_bank(self.config)
        
        self.model_factory = model_factory
        
        # Stream managers
        self.streams: Dict[str, StreamManager] = {}
        self.stream_order: List[str] = []
        
        # Cross-stream integration (initialised when streams added)
        self.cross_stream: Optional[CrossStreamRegression] = None
        self.factor_model: Optional[OnlinePCA] = None
        
        # Observation tracking
        self.t = 0
        self.observed_this_period: List[str] = []
    
    def add_stream(self, name: str) -> None:
        """Add a new data stream."""
        self.streams[name] = StreamManager(
            name=name,
            config=self.config,
            model_factory=self.model_factory
        )
        self.stream_order.append(name)
        
        # Reinitialise cross-stream components
        if len(self.streams) > 1:
            self.cross_stream = CrossStreamRegression(
                stream_names=list(self.streams.keys()),
                config=self.config
            )
            
            if len(self.streams) > 3:
                self.factor_model = OnlinePCA(
                    n_streams=len(self.streams),
                    n_factors=self.config.n_factors
                )
    
    def observe(self, stream_name: str, value: float, t: Optional[int] = None) -> None:
        """
        Record observation for a stream.
        
        Args:
            stream_name: Name of the stream
            value: Observed value
            t: Optional time index (auto-incremented if not provided)
        """
        if stream_name not in self.streams:
            raise ValueError(f"Unknown stream: {stream_name}")
        
        if t is not None:
            self.t = t
        
        # Track observation order for lag-0 cross-stream effects
        self.observed_this_period.append(stream_name)
        
        # Get stream manager
        stream = self.streams[stream_name]
        
        # Compute cross-stream adjustment if available
        if self.cross_stream is not None and stream.last_prediction is not None:
            residual = value - stream.last_prediction.mean
            adjustment = self.cross_stream.update(
                stream_name,
                residual,
                self.observed_this_period
            )
        else:
            adjustment = 0.0
        
        # Update stream
        stream.observe(value, self.t)
        
        # Update factor model if available
        if self.factor_model is not None:
            if len(self.observed_this_period) == len(self.streams):
                obs_vector = np.array([
                    self.streams[name].scale_manager.history[-1]
                    for name in self.stream_order
                ])
                self.factor_model.update(obs_vector)
    
    def predict(self, stream_name: str, horizon: int = 1) -> Prediction:
        """
        Generate prediction for a stream.
        
        Args:
            stream_name: Name of the stream
            horizon: Steps ahead to predict
            
        Returns:
            Prediction with mean and calibrated uncertainty
        """
        if stream_name not in self.streams:
            raise ValueError(f"Unknown stream: {stream_name}")
        
        return self.streams[stream_name].predict(horizon)
    
    def end_period(self) -> None:
        """
        Signal end of observation period.
        
        Call after all streams have been observed for the current time step.
        """
        if self.cross_stream is not None:
            self.cross_stream.end_period()
        
        self.observed_this_period = []
        self.t += 1
    
    def get_diagnostics(self, stream_name: str) -> dict:
        """Get diagnostic information for a stream."""
        if stream_name not in self.streams:
            raise ValueError(f"Unknown stream: {stream_name}")
        
        diag = self.streams[stream_name].get_diagnostics()
        
        # Add cross-stream info
        if self.cross_stream is not None:
            diag["cross_stream"] = self.cross_stream.get_diagnostics()
        
        if self.factor_model is not None:
            diag["factor_model"] = self.factor_model.get_diagnostics()
        
        return diag
    
    def get_all_diagnostics(self) -> dict:
        """Get diagnostic information for all streams."""
        return {
            name: self.get_diagnostics(name)
            for name in self.streams
        }
```

---

*End of Appendix A*
