# AEGIS Appendix B: Model Specifications

## Complete Model Implementations

---

## Contents

1. [Model Overview](#1-model-overview)
2. [Persistence Models](#2-persistence-models)
3. [Trend Models](#3-trend-models)
4. [Mean Reversion Models](#4-mean-reversion-models)
5. [Periodic Models](#5-periodic-models)
6. [Dynamic Models](#6-dynamic-models)
7. [Jump and Regime Models](#7-jump-and-regime-models)
8. [Variance Models](#8-variance-models)
9. [FEP-Native Models](#9-fep-native-models)
10. [Model Factory](#10-model-factory)

---

## 1. Model Overview

### 1.1 Model Groups

| Group | Models | Epistemic Value |
|-------|--------|-----------------|
| Persistence | RandomWalk, LocalLevel | Minimal after burn-in |
| Trend | LocalTrend, DampedTrend | Moderate (slope uncertainty) |
| Reversion | MeanReversion, AsymmetricMR, ThresholdAR | High when far from equilibrium |
| Periodic | OscillatorBank, SeasonalDummy | Low (coefficients converge) |
| Dynamic | AR2, MA1, ARMA | Moderate |
| Jump/Regime | JumpDiffusion, ChangePoint | High near regime boundaries |
| Variance | VolatilityTracker, LevelDependentVol, QuantileTracker | Low (auxiliary models) |
| FEP-Native | GaussianProcessApprox, MixtureOfExperts, HierarchicalMR | Naturally high epistemic value |

### 1.2 Base Class Reference

All models inherit from `TemporalModel` (defined in main specification):

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

@dataclass
class Prediction:
    mean: float
    variance: float
    
    @property
    def std(self) -> float:
        return np.sqrt(self.variance)

class TemporalModel(ABC):
    @abstractmethod
    def update(self, y: float, t: int) -> None: ...
    
    @abstractmethod
    def predict(self, horizon: int) -> Prediction: ...
    
    @abstractmethod
    def log_likelihood(self, y: float) -> float: ...
    
    @abstractmethod
    def reset(self, partial: float = 1.0) -> None: ...
    
    def pragmatic_value(self) -> float:
        pred = self.predict(horizon=1)
        return -0.5 * np.log(2 * np.pi * pred.variance) - 0.5
    
    def epistemic_value(self) -> float:
        return 0.0
    
    @property
    def name(self) -> str:
        return self.__class__.__name__
    
    @property
    def n_parameters(self) -> int:
        return 0
    
    @property
    def group(self) -> str:
        return "unknown"
```

---

## 2. Persistence Models

### 2.1 Random Walk

**Structure**: $\hat{y}_{t+1} = y_t$

```python
class RandomWalkModel(TemporalModel):
    """
    Random walk: best prediction is current value.
    
    Optimal for unpredictable series. Variance scales linearly with horizon.
    """
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha  # Variance learning rate
        self.sigma_sq = 1.0
        self.last_value = 0.0
        self.n_obs = 0
    
    def update(self, y: float, t: int) -> None:
        if self.n_obs > 0:
            innovation = y - self.last_value
            self.sigma_sq = (1 - self.alpha) * self.sigma_sq + self.alpha * innovation**2
        self.last_value = y
        self.n_obs += 1
    
    def predict(self, horizon: int) -> Prediction:
        return Prediction(
            mean=self.last_value,
            variance=self.sigma_sq * horizon
        )
    
    def log_likelihood(self, y: float) -> float:
        var = self.sigma_sq
        return -0.5 * np.log(2 * np.pi * var) - (y - self.last_value)**2 / (2 * var)
    
    def reset(self, partial: float = 1.0) -> None:
        self.sigma_sq = partial * self.sigma_sq + (1 - partial) * 1.0
    
    @property
    def n_parameters(self) -> int:
        return 1
    
    @property
    def group(self) -> str:
        return "persistence"
```

### 2.2 Local Level (Exponential Smoothing)

**Structure**: $\ell_t = \alpha y_t + (1-\alpha) \ell_{t-1}$

```python
class LocalLevelModel(TemporalModel):
    """
    Local level model: exponentially smoothed estimate of level.
    
    Good for noisy observations of a slowly-moving underlying value.
    """
    
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.level = 0.0
        self.sigma_sq = 1.0
        self.n_obs = 0
    
    def update(self, y: float, t: int) -> None:
        if self.n_obs == 0:
            self.level = y
        else:
            error = y - self.level
            self.sigma_sq = 0.95 * self.sigma_sq + 0.05 * error**2
            self.level = self.alpha * y + (1 - self.alpha) * self.level
        self.n_obs += 1
    
    def predict(self, horizon: int) -> Prediction:
        # Level stays constant; variance grows
        var_mult = 1 + (horizon - 1) * self.alpha**2 / (2 - self.alpha)
        return Prediction(
            mean=self.level,
            variance=self.sigma_sq * var_mult
        )
    
    def log_likelihood(self, y: float) -> float:
        var = self.sigma_sq
        return -0.5 * np.log(2 * np.pi * var) - (y - self.level)**2 / (2 * var)
    
    def reset(self, partial: float = 1.0) -> None:
        self.sigma_sq = partial * self.sigma_sq + (1 - partial) * 1.0
    
    @property
    def n_parameters(self) -> int:
        return 2
    
    @property
    def group(self) -> str:
        return "persistence"
```

---

## 3. Trend Models

### 3.1 Local Trend (Holt's Method)

**Structure**: Level plus slope, both updated exponentially.

```python
class LocalTrendModel(TemporalModel):
    """
    Local trend model: exponentially smoothed level and slope.
    
    Captures trending behaviour with adaptive slope estimation.
    """
    
    def __init__(self, alpha: float = 0.1, beta: float = 0.05):
        self.alpha = alpha  # Level smoothing
        self.beta = beta    # Slope smoothing
        self.level = 0.0
        self.slope = 0.0
        self.sigma_sq = 1.0
        self.n_obs = 0
    
    def update(self, y: float, t: int) -> None:
        if self.n_obs == 0:
            self.level = y
            self.n_obs += 1
            return
        
        if self.n_obs == 1:
            self.slope = y - self.level
            self.level = y
            self.n_obs += 1
            return
        
        # Prediction error
        pred = self.level + self.slope
        error = y - pred
        self.sigma_sq = 0.95 * self.sigma_sq + 0.05 * error**2
        
        # Update level and slope
        old_level = self.level
        self.level = self.alpha * y + (1 - self.alpha) * (self.level + self.slope)
        self.slope = self.beta * (self.level - old_level) + (1 - self.beta) * self.slope
        
        self.n_obs += 1
    
    def predict(self, horizon: int) -> Prediction:
        mean = self.level + horizon * self.slope
        # Variance grows with horizon
        var = self.sigma_sq * (1 + 0.5 * (horizon - 1))
        return Prediction(mean=mean, variance=var)
    
    def log_likelihood(self, y: float) -> float:
        pred = self.level + self.slope
        var = self.sigma_sq
        return -0.5 * np.log(2 * np.pi * var) - (y - pred)**2 / (2 * var)
    
    def reset(self, partial: float = 1.0) -> None:
        self.slope = partial * self.slope
        self.sigma_sq = partial * self.sigma_sq + (1 - partial) * 1.0
    
    @property
    def n_parameters(self) -> int:
        return 3
    
    @property
    def group(self) -> str:
        return "trend"
```

### 3.2 Damped Trend

**Structure**: Trend that decelerates over time.

```python
class DampedTrendModel(TemporalModel):
    """
    Damped trend model: slope decays toward zero over forecast horizon.
    
    Better for growth processes that naturally decelerate.
    """
    
    def __init__(self, alpha: float = 0.1, beta: float = 0.05, phi: float = 0.9):
        self.alpha = alpha
        self.beta = beta
        self.phi = phi  # Damping factor
        self.level = 0.0
        self.slope = 0.0
        self.sigma_sq = 1.0
        self.n_obs = 0
    
    def update(self, y: float, t: int) -> None:
        if self.n_obs == 0:
            self.level = y
            self.n_obs += 1
            return
        
        if self.n_obs == 1:
            self.slope = y - self.level
            self.level = y
            self.n_obs += 1
            return
        
        # Prediction with damping
        pred = self.level + self.phi * self.slope
        error = y - pred
        self.sigma_sq = 0.95 * self.sigma_sq + 0.05 * error**2
        
        old_level = self.level
        self.level = self.alpha * y + (1 - self.alpha) * (self.level + self.phi * self.slope)
        self.slope = self.beta * (self.level - old_level) + (1 - self.beta) * self.phi * self.slope
        
        self.n_obs += 1
    
    def predict(self, horizon: int) -> Prediction:
        if abs(self.phi - 1.0) < 1e-6:
            mean = self.level + horizon * self.slope
        else:
            # Sum of geometric series
            mean = self.level + self.phi * (1 - self.phi**horizon) / (1 - self.phi) * self.slope
        
        var = self.sigma_sq * (1 + 0.3 * horizon * (1 - 0.5 * (1 - self.phi**horizon)))
        return Prediction(mean=mean, variance=var)
    
    def log_likelihood(self, y: float) -> float:
        pred = self.level + self.phi * self.slope
        var = self.sigma_sq
        return -0.5 * np.log(2 * np.pi * var) - (y - pred)**2 / (2 * var)
    
    def reset(self, partial: float = 1.0) -> None:
        self.slope = partial * self.slope
        self.sigma_sq = partial * self.sigma_sq + (1 - partial) * 1.0
    
    @property
    def n_parameters(self) -> int:
        return 4
    
    @property
    def group(self) -> str:
        return "trend"
```

---

## 4. Mean Reversion Models

### 4.1 Mean Reversion (AR(1) toward mean)

**Structure**: $\hat{y}_{t+h} = \mu + \phi^h (y_t - \mu)$

```python
class MeanReversionModel(TemporalModel):
    """
    Mean reversion model: deviation from long-run mean decays exponentially.
    
    Includes epistemic value based on parameter uncertainty.
    """
    
    def __init__(self, phi: float = 0.9):
        self.mu = 0.0           # Long-run mean
        self.phi = phi          # Decay rate
        self.phi_var = 0.01     # Uncertainty over phi
        self.sigma_sq = 1.0
        self.last_y = 0.0
        self.n_obs = 0
        
        # Learning rates
        self.mu_lr = 0.001
        self.phi_lr = 0.01
    
    def update(self, y: float, t: int) -> None:
        if self.n_obs > 0:
            deviation = self.last_y - self.mu
            pred = self.mu + self.phi * deviation
            error = y - pred
            
            # Bayesian update for phi (linear regression form)
            x = deviation
            if abs(x) > 1e-6:
                gain = (self.phi_var * x) / (self.sigma_sq + self.phi_var * x**2)
                self.phi += gain * error
                self.phi_var *= (1 - gain * x)
                self.phi_var = max(self.phi_var, 1e-6)
            
            # Constrain phi
            self.phi = np.clip(self.phi, 0.0, 0.999)
            
            # Update variance and mean
            self.sigma_sq = 0.95 * self.sigma_sq + 0.05 * error**2
            self.mu = (1 - self.mu_lr) * self.mu + self.mu_lr * y
        
        self.last_y = y
        self.n_obs += 1
    
    def predict(self, horizon: int) -> Prediction:
        deviation = self.last_y - self.mu
        mean = self.mu + (self.phi ** horizon) * deviation
        
        # Variance formula for AR(1)
        if abs(self.phi) < 0.999:
            var = self.sigma_sq * (1 - self.phi ** (2 * horizon)) / (1 - self.phi**2)
        else:
            var = self.sigma_sq * horizon
        
        return Prediction(mean=mean, variance=max(var, self.sigma_sq * 0.1))
    
    def log_likelihood(self, y: float) -> float:
        pred = self.predict(horizon=1)
        return -0.5 * np.log(2 * np.pi * pred.variance) - (y - pred.mean)**2 / (2 * pred.variance)
    
    def reset(self, partial: float = 1.0) -> None:
        self.phi = partial * self.phi + (1 - partial) * 0.9
        self.phi_var = partial * self.phi_var + (1 - partial) * 0.01
        self.sigma_sq = partial * self.sigma_sq + (1 - partial) * 1.0
    
    def epistemic_value(self) -> float:
        """Information gain about phi from next observation."""
        deviation = self.last_y - self.mu
        x = deviation
        
        # Expected variance reduction
        if self.phi_var > 1e-8 and abs(x) > 1e-6:
            var_reduction = (self.phi_var**2 * x**2) / (self.sigma_sq + self.phi_var * x**2)
            return 0.5 * np.log(1 + var_reduction / self.phi_var)
        return 0.0
    
    @property
    def n_parameters(self) -> int:
        return 3
    
    @property
    def group(self) -> str:
        return "reversion"
```

### 4.2 Asymmetric Mean Reversion

**Structure**: Different reversion speeds above and below mean.

```python
class AsymmetricMeanReversionModel(TemporalModel):
    """
    Asymmetric mean reversion: different speeds for positive/negative deviations.
    
    Captures asymmetric dynamics like interest rates or unemployment.
    """
    
    def __init__(self):
        self.mu = 0.0
        self.phi_up = 0.9       # Reversion from above
        self.phi_down = 0.9     # Reversion from below
        self.phi_up_var = 0.01
        self.phi_down_var = 0.01
        self.sigma_sq = 1.0
        self.last_y = 0.0
        self.n_obs = 0
        self.mu_lr = 0.001
    
    def update(self, y: float, t: int) -> None:
        if self.n_obs > 0:
            deviation = self.last_y - self.mu
            
            if deviation > 0:
                phi = self.phi_up
                phi_var = self.phi_up_var
                pred = self.mu + phi * deviation
                error = y - pred
                
                # Update phi_up
                x = deviation
                if abs(x) > 1e-6:
                    gain = (phi_var * x) / (self.sigma_sq + phi_var * x**2)
                    self.phi_up += gain * error
                    self.phi_up_var *= (1 - gain * x)
                    self.phi_up_var = max(self.phi_up_var, 1e-6)
                self.phi_up = np.clip(self.phi_up, 0.0, 0.999)
            else:
                phi = self.phi_down
                phi_var = self.phi_down_var
                pred = self.mu + phi * deviation
                error = y - pred
                
                # Update phi_down
                x = deviation
                if abs(x) > 1e-6:
                    gain = (phi_var * x) / (self.sigma_sq + phi_var * x**2)
                    self.phi_down += gain * error
                    self.phi_down_var *= (1 - gain * x)
                    self.phi_down_var = max(self.phi_down_var, 1e-6)
                self.phi_down = np.clip(self.phi_down, 0.0, 0.999)
            
            self.sigma_sq = 0.95 * self.sigma_sq + 0.05 * error**2
            self.mu = (1 - self.mu_lr) * self.mu + self.mu_lr * y
        
        self.last_y = y
        self.n_obs += 1
    
    def predict(self, horizon: int) -> Prediction:
        deviation = self.last_y - self.mu
        y = self.last_y
        
        for _ in range(horizon):
            dev = y - self.mu
            if dev > 0:
                y = self.mu + self.phi_up * dev
            else:
                y = self.mu + self.phi_down * dev
        
        avg_phi = (self.phi_up + self.phi_down) / 2
        if abs(avg_phi) < 0.999:
            var = self.sigma_sq * (1 - avg_phi ** (2 * horizon)) / (1 - avg_phi**2 + 1e-10)
        else:
            var = self.sigma_sq * horizon
        
        return Prediction(mean=y, variance=max(var, self.sigma_sq * 0.1))
    
    def log_likelihood(self, y: float) -> float:
        pred = self.predict(horizon=1)
        return -0.5 * np.log(2 * np.pi * pred.variance) - (y - pred.mean)**2 / (2 * pred.variance)
    
    def reset(self, partial: float = 1.0) -> None:
        self.phi_up = partial * self.phi_up + (1 - partial) * 0.9
        self.phi_down = partial * self.phi_down + (1 - partial) * 0.9
        self.phi_up_var = partial * self.phi_up_var + (1 - partial) * 0.01
        self.phi_down_var = partial * self.phi_down_var + (1 - partial) * 0.01
    
    def epistemic_value(self) -> float:
        deviation = self.last_y - self.mu
        
        if deviation > 0:
            phi_var = self.phi_up_var
        else:
            phi_var = self.phi_down_var
        
        x = abs(deviation)
        if phi_var > 1e-8 and x > 1e-6:
            var_reduction = (phi_var**2 * x**2) / (self.sigma_sq + phi_var * x**2)
            return 0.5 * np.log(1 + var_reduction / phi_var)
        return 0.0
    
    @property
    def n_parameters(self) -> int:
        return 4
    
    @property
    def group(self) -> str:
        return "reversion"
```

### 4.3 Threshold AR

**Structure**: Different dynamics above and below threshold.

```python
class ThresholdARModel(TemporalModel):
    """
    Threshold autoregression: regime-dependent dynamics.
    
    High epistemic value near threshold where observations are most informative.
    """
    
    def __init__(self, tau: float = 0.0):
        self.tau = tau          # Threshold
        self.tau_var = 1.0      # Uncertainty over threshold
        self.c_low, self.phi_low = 0.0, 0.9
        self.c_high, self.phi_high = 0.0, 0.9
        self.phi_low_var = 0.01
        self.phi_high_var = 0.01
        self.sigma_sq = 1.0
        self.last_y = 0.0
        self.n_obs = 0
        self.sum_y = 0.0
        self.lr = 0.01
    
    def update(self, y: float, t: int) -> None:
        if self.n_obs > 0:
            if self.last_y < self.tau:
                pred = self.c_low + self.phi_low * self.last_y
                error = y - pred
                
                # Update low regime
                self.c_low += self.lr * error
                x = self.last_y
                if abs(x) > 1e-6:
                    gain = (self.phi_low_var * x) / (self.sigma_sq + self.phi_low_var * x**2)
                    self.phi_low += gain * error
                    self.phi_low_var *= (1 - gain * x)
                    self.phi_low_var = max(self.phi_low_var, 1e-6)
                self.phi_low = np.clip(self.phi_low, -0.99, 0.99)
            else:
                pred = self.c_high + self.phi_high * self.last_y
                error = y - pred
                
                # Update high regime
                self.c_high += self.lr * error
                x = self.last_y
                if abs(x) > 1e-6:
                    gain = (self.phi_high_var * x) / (self.sigma_sq + self.phi_high_var * x**2)
                    self.phi_high += gain * error
                    self.phi_high_var *= (1 - gain * x)
                    self.phi_high_var = max(self.phi_high_var, 1e-6)
                self.phi_high = np.clip(self.phi_high, -0.99, 0.99)
            
            self.sigma_sq = 0.95 * self.sigma_sq + 0.05 * error**2
            
            # Adapt threshold toward median
            self.sum_y += y
            if self.n_obs > 50 and self.n_obs % 50 == 0:
                self.tau = 0.95 * self.tau + 0.05 * (self.sum_y / self.n_obs)
                self.tau_var *= 0.99  # Threshold uncertainty decreases
        
        self.last_y = y
        self.n_obs += 1
    
    def predict(self, horizon: int) -> Prediction:
        y = self.last_y
        for _ in range(horizon):
            if y < self.tau:
                y = self.c_low + self.phi_low * y
            else:
                y = self.c_high + self.phi_high * y
        return Prediction(mean=y, variance=self.sigma_sq * horizon)
    
    def log_likelihood(self, y: float) -> float:
        if self.last_y < self.tau:
            pred = self.c_low + self.phi_low * self.last_y
        else:
            pred = self.c_high + self.phi_high * self.last_y
        var = self.sigma_sq
        return -0.5 * np.log(2 * np.pi * var) - (y - pred)**2 / (2 * var)
    
    def reset(self, partial: float = 1.0) -> None:
        self.phi_low = partial * self.phi_low + (1 - partial) * 0.9
        self.phi_high = partial * self.phi_high + (1 - partial) * 0.9
        self.phi_low_var = partial * self.phi_low_var + (1 - partial) * 0.01
        self.phi_high_var = partial * self.phi_high_var + (1 - partial) * 0.01
        self.tau_var = partial * self.tau_var + (1 - partial) * 1.0
    
    def epistemic_value(self) -> float:
        distance_to_threshold = abs(self.last_y - self.tau)
        threshold_std = np.sqrt(self.tau_var)
        
        # Near threshold: very informative about tau
        if distance_to_threshold < 2 * threshold_std:
            tau_info = 0.5 * np.log(1 + 1.0 / (1 + distance_to_threshold**2))
        else:
            tau_info = 0.0
        
        # Phi information
        if self.last_y < self.tau:
            phi_var = self.phi_low_var
        else:
            phi_var = self.phi_high_var
        
        x = abs(self.last_y)
        if phi_var > 1e-8 and x > 1e-6:
            phi_info = 0.5 * np.log(1 + phi_var * x**2 / self.sigma_sq)
        else:
            phi_info = 0.0
        
        return tau_info + phi_info
    
    @property
    def n_parameters(self) -> int:
        return 6
    
    @property
    def group(self) -> str:
        return "reversion"
```

---

## 5. Periodic Models

### 5.1 Oscillator Bank

**Structure**: Sum of sinusoids at predetermined frequencies.

```python
class OscillatorBankModel(TemporalModel):
    """
    Bank of oscillators at different frequencies.
    
    Captures periodic structure by learning amplitude and phase at each frequency.
    """
    
    def __init__(self, periods: list = None):
        self.periods = periods or [4, 8, 16, 32, 64, 128, 256]
        self.n_freqs = len(self.periods)
        
        # Coefficients: a_k cos(wt) + b_k sin(wt)
        self.a = np.zeros(self.n_freqs)
        self.b = np.zeros(self.n_freqs)
        
        self.sigma_sq = 1.0
        self.t = 0
        self.lr = 0.01
    
    def update(self, y: float, t: int) -> None:
        self.t = t
        
        # Compute prediction
        pred = self._compute_prediction(t)
        error = y - pred
        
        self.sigma_sq = 0.95 * self.sigma_sq + 0.05 * error**2
        
        # Gradient update for each frequency
        for k, period in enumerate(self.periods):
            omega = 2 * np.pi / period
            cos_term = np.cos(omega * t)
            sin_term = np.sin(omega * t)
            
            self.a[k] += self.lr * error * cos_term
            self.b[k] += self.lr * error * sin_term
    
    def _compute_prediction(self, t: int) -> float:
        pred = 0.0
        for k, period in enumerate(self.periods):
            omega = 2 * np.pi / period
            pred += self.a[k] * np.cos(omega * t) + self.b[k] * np.sin(omega * t)
        return pred
    
    def predict(self, horizon: int) -> Prediction:
        future_t = self.t + horizon
        mean = self._compute_prediction(future_t)
        return Prediction(mean=mean, variance=self.sigma_sq)
    
    def log_likelihood(self, y: float) -> float:
        pred = self._compute_prediction(self.t + 1)
        var = self.sigma_sq
        return -0.5 * np.log(2 * np.pi * var) - (y - pred)**2 / (2 * var)
    
    def reset(self, partial: float = 1.0) -> None:
        self.a *= partial
        self.b *= partial
        self.sigma_sq = partial * self.sigma_sq + (1 - partial) * 1.0
    
    @property
    def n_parameters(self) -> int:
        return 2 * self.n_freqs + 1
    
    @property
    def group(self) -> str:
        return "periodic"
```

### 5.2 Seasonal Dummy

**Structure**: Separate mean for each seasonal period.

```python
class SeasonalDummyModel(TemporalModel):
    """
    Seasonal dummy model: separate means for each position in the cycle.
    
    Better than sinusoids for sharp seasonal patterns.
    """
    
    def __init__(self, period: int):
        self.period = period
        self.means = np.zeros(period)
        self.counts = np.zeros(period)
        self.sigma_sq = 1.0
        self.t = 0
        self.forget = 0.99
    
    def update(self, y: float, t: int) -> None:
        self.t = t
        s = t % self.period
        
        error = y - self.means[s]
        self.sigma_sq = 0.95 * self.sigma_sq + 0.05 * error**2
        
        # Update seasonal mean with forgetting
        self.counts[s] = self.forget * self.counts[s] + 1
        alpha = 1.0 / self.counts[s]
        self.means[s] = (1 - alpha) * self.means[s] + alpha * y
    
    def predict(self, horizon: int) -> Prediction:
        s = (self.t + horizon) % self.period
        return Prediction(mean=self.means[s], variance=self.sigma_sq)
    
    def log_likelihood(self, y: float) -> float:
        s = (self.t + 1) % self.period
        pred = self.means[s]
        var = self.sigma_sq
        return -0.5 * np.log(2 * np.pi * var) - (y - pred)**2 / (2 * var)
    
    def reset(self, partial: float = 1.0) -> None:
        self.means *= partial
        self.counts *= partial
        self.sigma_sq = partial * self.sigma_sq + (1 - partial) * 1.0
    
    @property
    def n_parameters(self) -> int:
        return self.period + 1
    
    @property
    def group(self) -> str:
        return "periodic"
```

---

## 6. Dynamic Models

### 6.1 AR(2)

**Structure**: Two-lag autoregression.

```python
class AR2Model(TemporalModel):
    """
    AR(2) model: captures richer autocorrelation structure.
    
    Can model oscillatory behaviour via complex roots.
    """
    
    def __init__(self):
        self.c = 0.0
        self.phi1 = 0.5
        self.phi2 = 0.3
        self.sigma_sq = 1.0
        self.y_lag1 = 0.0
        self.y_lag2 = 0.0
        self.n_obs = 0
        
        # RLS components
        self.P = np.eye(3) * 10.0  # Covariance for [c, phi1, phi2]
        self.theta = np.array([0.0, 0.5, 0.3])
        self.forget = 0.99
    
    def update(self, y: float, t: int) -> None:
        if self.n_obs >= 2:
            # Feature vector
            x = np.array([1.0, self.y_lag1, self.y_lag2])
            
            # Prediction
            pred = np.dot(x, self.theta)
            error = y - pred
            
            # RLS update
            Px = self.P @ x
            denom = self.forget + np.dot(x, Px)
            gain = Px / denom
            
            self.theta = self.theta + gain * error
            self.P = (self.P - np.outer(gain, Px)) / self.forget
            
            # Extract parameters
            self.c, self.phi1, self.phi2 = self.theta
            
            # Ensure stationarity (approximate)
            if abs(self.phi1) + abs(self.phi2) > 0.99:
                scale = 0.99 / (abs(self.phi1) + abs(self.phi2) + 1e-6)
                self.phi1 *= scale
                self.phi2 *= scale
                self.theta[1:] = [self.phi1, self.phi2]
            
            self.sigma_sq = 0.95 * self.sigma_sq + 0.05 * error**2
        
        # Update lags
        self.y_lag2 = self.y_lag1
        self.y_lag1 = y
        self.n_obs += 1
    
    def predict(self, horizon: int) -> Prediction:
        y1, y2 = self.y_lag1, self.y_lag2
        
        for _ in range(horizon):
            y_new = self.c + self.phi1 * y1 + self.phi2 * y2
            y2 = y1
            y1 = y_new
        
        return Prediction(mean=y1, variance=self.sigma_sq * horizon)
    
    def log_likelihood(self, y: float) -> float:
        if self.n_obs < 2:
            return -0.5 * np.log(2 * np.pi * self.sigma_sq)
        pred = self.c + self.phi1 * self.y_lag1 + self.phi2 * self.y_lag2
        var = self.sigma_sq
        return -0.5 * np.log(2 * np.pi * var) - (y - pred)**2 / (2 * var)
    
    def reset(self, partial: float = 1.0) -> None:
        self.theta = partial * self.theta + (1 - partial) * np.array([0.0, 0.5, 0.3])
        self.c, self.phi1, self.phi2 = self.theta
        self.P = partial * self.P + (1 - partial) * np.eye(3) * 10.0
    
    @property
    def n_parameters(self) -> int:
        return 4
    
    @property
    def group(self) -> str:
        return "dynamic"
```

### 6.2 MA(1)

**Structure**: Prediction depends on previous forecast error.

```python
class MA1Model(TemporalModel):
    """
    MA(1) model: captures one-period shock effects.
    
    Useful for inventory adjustments and filtered signals.
    """
    
    def __init__(self, theta: float = 0.5):
        self.theta = theta
        self.last_error = 0.0
        self.sigma_sq = 1.0
        self.n_obs = 0
        self.lr = 0.01
    
    def update(self, y: float, t: int) -> None:
        pred = self.theta * self.last_error
        error = y - pred
        
        # Gradient update for theta
        grad = -error * self.last_error
        self.theta = np.clip(self.theta - self.lr * grad, -0.99, 0.99)
        
        self.sigma_sq = 0.95 * self.sigma_sq + 0.05 * error**2
        self.last_error = error
        self.n_obs += 1
    
    def predict(self, horizon: int) -> Prediction:
        if horizon == 1:
            mean = self.theta * self.last_error
            var = self.sigma_sq
        else:
            # MA(1) is unpredictable beyond h=1
            mean = 0.0
            var = self.sigma_sq * (1 + self.theta**2)
        
        return Prediction(mean=mean, variance=var)
    
    def log_likelihood(self, y: float) -> float:
        pred = self.theta * self.last_error
        var = self.sigma_sq
        return -0.5 * np.log(2 * np.pi * var) - (y - pred)**2 / (2 * var)
    
    def reset(self, partial: float = 1.0) -> None:
        self.theta = partial * self.theta + (1 - partial) * 0.5
        self.last_error = 0.0
    
    @property
    def n_parameters(self) -> int:
        return 2
    
    @property
    def group(self) -> str:
        return "dynamic"
```

---

## 7. Jump and Regime Models

### 7.1 Jump Diffusion

**Structure**: Random walk plus occasional jumps.

```python
class JumpDiffusionModel(TemporalModel):
    """
    Jump diffusion: distinguishes continuous volatility from discrete jumps.
    
    Variance estimates include jump risk even during quiet periods.
    """
    
    def __init__(self):
        # Beta prior on jump probability
        self.lambda_a = 1.0  # Prior shape
        self.lambda_b = 49.0  # Prior: ~2% jump rate
        
        self.sigma_sq_diff = 1.0
        self.mu_jump = 0.0
        self.sigma_sq_jump = 10.0
        self.last_y = 0.0
        self.n_obs = 0
        
        self.jump_threshold = 3.0  # Std devs
        self.recent_jumps = []
    
    def lambda_mean(self) -> float:
        return self.lambda_a / (self.lambda_a + self.lambda_b)
    
    def lambda_var(self) -> float:
        a, b = self.lambda_a, self.lambda_b
        return (a * b) / ((a + b)**2 * (a + b + 1))
    
    def update(self, y: float, t: int) -> None:
        if self.n_obs > 0:
            innovation = y - self.last_y
            
            # Classify as jump or diffusion
            is_jump = abs(innovation) > self.jump_threshold * np.sqrt(self.sigma_sq_diff)
            
            if is_jump:
                # Update jump parameters
                self.recent_jumps.append(innovation)
                if len(self.recent_jumps) > 20:
                    self.recent_jumps.pop(0)
                
                if len(self.recent_jumps) >= 3:
                    self.mu_jump = np.mean(self.recent_jumps)
                    self.sigma_sq_jump = np.var(self.recent_jumps) + 1e-6
                
                # Beta update: observed jump
                self.lambda_a += 1
            else:
                # Update diffusion variance
                self.sigma_sq_diff = 0.95 * self.sigma_sq_diff + 0.05 * innovation**2
                
                # Beta update: no jump
                self.lambda_b += 1
        
        self.last_y = y
        self.n_obs += 1
    
    def predict(self, horizon: int) -> Prediction:
        lam = self.lambda_mean()
        mean = self.last_y + horizon * lam * self.mu_jump
        
        # Variance includes both diffusion and jump risk
        var_per_step = self.sigma_sq_diff + lam * (self.mu_jump**2 + self.sigma_sq_jump)
        var = var_per_step * horizon
        
        return Prediction(mean=mean, variance=var)
    
    def log_likelihood(self, y: float) -> float:
        pred = self.predict(horizon=1)
        return -0.5 * np.log(2 * np.pi * pred.variance) - (y - pred.mean)**2 / (2 * pred.variance)
    
    def reset(self, partial: float = 1.0) -> None:
        self.lambda_a = partial * self.lambda_a + (1 - partial) * 1.0
        self.lambda_b = partial * self.lambda_b + (1 - partial) * 49.0
        self.recent_jumps = []
    
    def epistemic_value(self) -> float:
        # Information about lambda
        n_eff = self.lambda_a + self.lambda_b - 2
        if n_eff > 0:
            return 0.5 * np.log(1 + 1.0 / n_eff)
        return 1.0
    
    @property
    def n_parameters(self) -> int:
        return 4
    
    @property
    def group(self) -> str:
        return "special"
```

### 7.2 Change Point Model

**Structure**: Detects and adapts to structural changes.

```python
class ChangePointModel(TemporalModel):
    """
    Change point model: explicitly models probability of regime change.
    
    High epistemic value when change point probability is uncertain.
    """
    
    def __init__(self, hazard_rate: float = 0.01):
        self.hazard_rate = hazard_rate  # Prior probability of change
        self.hazard_a = 1.0  # Beta prior on hazard
        self.hazard_b = 99.0
        
        # Current regime parameters
        self.mu = 0.0
        self.sigma_sq = 1.0
        
        # Sufficient statistics for current regime
        self.regime_sum = 0.0
        self.regime_sum_sq = 0.0
        self.regime_n = 0
        
        self.last_y = 0.0
        self.n_obs = 0
        
        # Run length distribution (simplified)
        self.run_length = 0
    
    def update(self, y: float, t: int) -> None:
        if self.n_obs > 0:
            # Compute predictive probability
            pred_mean = self.mu
            pred_var = self.sigma_sq * (1 + 1.0 / (self.regime_n + 1))
            
            error = y - pred_mean
            ll_current = -0.5 * np.log(2 * np.pi * pred_var) - error**2 / (2 * pred_var)
            
            # Prior predictive (for new regime)
            ll_new = -0.5 * np.log(2 * np.pi * self.sigma_sq) - y**2 / (2 * self.sigma_sq)
            
            # Change point probability
            hazard = self.hazard_a / (self.hazard_a + self.hazard_b)
            
            # Posterior probability of change (Bayes rule)
            p_change = hazard * np.exp(ll_new) / (
                hazard * np.exp(ll_new) + (1 - hazard) * np.exp(ll_current) + 1e-10
            )
            
            if np.random.random() < p_change or p_change > 0.5:
                # New regime
                self.regime_sum = y
                self.regime_sum_sq = y**2
                self.regime_n = 1
                self.run_length = 0
                
                # Update hazard posterior
                self.hazard_a += 1
            else:
                # Continue current regime
                self.regime_sum += y
                self.regime_sum_sq += y**2
                self.regime_n += 1
                self.run_length += 1
                
                self.hazard_b += 1
            
            # Update regime parameters
            if self.regime_n > 0:
                self.mu = self.regime_sum / self.regime_n
                if self.regime_n > 1:
                    var = (self.regime_sum_sq - self.regime_n * self.mu**2) / (self.regime_n - 1)
                    self.sigma_sq = 0.9 * self.sigma_sq + 0.1 * max(var, 0.01)
        else:
            self.mu = y
            self.regime_sum = y
            self.regime_sum_sq = y**2
            self.regime_n = 1
        
        self.last_y = y
        self.n_obs += 1
    
    def predict(self, horizon: int) -> Prediction:
        return Prediction(mean=self.mu, variance=self.sigma_sq * (1 + horizon * self.hazard_rate))
    
    def log_likelihood(self, y: float) -> float:
        pred_var = self.sigma_sq * (1 + 1.0 / (self.regime_n + 1))
        return -0.5 * np.log(2 * np.pi * pred_var) - (y - self.mu)**2 / (2 * pred_var)
    
    def reset(self, partial: float = 1.0) -> None:
        self.regime_n = int(self.regime_n * partial)
        self.hazard_a = partial * self.hazard_a + (1 - partial) * 1.0
        self.hazard_b = partial * self.hazard_b + (1 - partial) * 99.0
    
    def epistemic_value(self) -> float:
        # Information about hazard rate and current regime mean
        hazard_info = 0.5 * np.log(1 + 1.0 / (self.hazard_a + self.hazard_b))
        
        # Regime mean uncertainty
        if self.regime_n > 0:
            mean_info = 0.5 * np.log(1 + 1.0 / self.regime_n)
        else:
            mean_info = 1.0
        
        return hazard_info + mean_info
    
    @property
    def n_parameters(self) -> int:
        return 3
    
    @property
    def group(self) -> str:
        return "special"
```

---

## 8. Variance Models

### 8.1 Volatility Tracker (EWMA)

```python
class VolatilityTrackerModel(TemporalModel):
    """
    EWMA volatility tracker.
    
    Doesn't predict levels; provides volatility scaling for other models.
    """
    
    def __init__(self, decay: float = 0.94):
        self.decay = decay
        self.sigma_sq = 1.0
        self.long_run_var = 1.0
        self.last_y = 0.0
        self.n_obs = 0
    
    def update(self, y: float, t: int) -> None:
        if self.n_obs > 0:
            innovation = y - self.last_y
            self.sigma_sq = self.decay * self.sigma_sq + (1 - self.decay) * innovation**2
            self.long_run_var = 0.999 * self.long_run_var + 0.001 * innovation**2
        
        self.last_y = y
        self.n_obs += 1
    
    def predict(self, horizon: int) -> Prediction:
        # Predicts persistence with current volatility
        return Prediction(mean=self.last_y, variance=self.sigma_sq * horizon)
    
    def log_likelihood(self, y: float) -> float:
        var = self.sigma_sq
        return -0.5 * np.log(2 * np.pi * var) - (y - self.last_y)**2 / (2 * var)
    
    def reset(self, partial: float = 1.0) -> None:
        self.sigma_sq = partial * self.sigma_sq + (1 - partial) * self.long_run_var
    
    def get_volatility_ratio(self) -> float:
        """Current volatility relative to long-run."""
        return np.sqrt(self.sigma_sq / (self.long_run_var + 1e-10))
    
    @property
    def n_parameters(self) -> int:
        return 1
    
    @property
    def group(self) -> str:
        return "variance"
```

### 8.2 Level-Dependent Volatility

```python
class LevelDependentVolModel(TemporalModel):
    """
    Level-dependent volatility: variance scales with signal level.
    
    Useful for count data and prices where percentage volatility is stable.
    """
    
    def __init__(self, gamma: float = 0.5):
        self.gamma = gamma  # Power relationship
        self.sigma_sq_base = 1.0
        self.last_y = 1.0
        self.n_obs = 0
    
    def update(self, y: float, t: int) -> None:
        if self.n_obs > 0 and abs(self.last_y) > 0.01:
            innovation = y - self.last_y
            # Estimate base variance (normalised by level)
            level_factor = abs(self.last_y) ** self.gamma
            normalised_sq = (innovation / level_factor) ** 2
            self.sigma_sq_base = 0.95 * self.sigma_sq_base + 0.05 * normalised_sq
        
        self.last_y = y
        self.n_obs += 1
    
    def predict(self, horizon: int) -> Prediction:
        level_factor = max(abs(self.last_y), 0.01) ** self.gamma
        var = self.sigma_sq_base * level_factor**2 * horizon
        return Prediction(mean=self.last_y, variance=var)
    
    def log_likelihood(self, y: float) -> float:
        pred = self.predict(horizon=1)
        return -0.5 * np.log(2 * np.pi * pred.variance) - (y - pred.mean)**2 / (2 * pred.variance)
    
    def reset(self, partial: float = 1.0) -> None:
        self.sigma_sq_base = partial * self.sigma_sq_base + (1 - partial) * 1.0
    
    @property
    def n_parameters(self) -> int:
        return 2
    
    @property
    def group(self) -> str:
        return "variance"
```

---

## 9. FEP-Native Models

These models are designed with the Free Energy Principle in mind, naturally providing well-defined epistemic value.

### 9.1 Gaussian Process Approximation

**Rationale**: GP regression provides natural uncertainty quantification. Epistemic value is high in regions with sparse data.

```python
class GaussianProcessApproxModel(TemporalModel):
    """
    Sparse Gaussian Process approximation for time series.
    
    Naturally provides epistemic value through posterior uncertainty.
    Uses a small set of inducing points for computational tractability.
    """
    
    def __init__(self, n_inducing: int = 20, length_scale: float = 10.0):
        self.n_inducing = n_inducing
        self.length_scale = length_scale
        
        # Inducing points and values
        self.inducing_t = np.linspace(0, 100, n_inducing)
        self.inducing_y = np.zeros(n_inducing)
        self.inducing_var = np.ones(n_inducing)
        
        # Noise variance
        self.sigma_sq = 1.0
        
        self.t = 0
        self.last_y = 0.0
        self.n_obs = 0
    
    def _kernel(self, t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
        """Squared exponential kernel."""
        diff = t1[:, None] - t2[None, :]
        return np.exp(-0.5 * diff**2 / self.length_scale**2)
    
    def update(self, y: float, t: int) -> None:
        self.t = t
        self.last_y = y
        self.n_obs += 1
        
        # Update inducing points (sliding window)
        if t > self.inducing_t[-1]:
            # Shift inducing points forward
            shift = t - self.inducing_t[-1] + 10
            self.inducing_t += shift
        
        # Find nearest inducing point and update
        distances = np.abs(self.inducing_t - t)
        nearest = np.argmin(distances)
        
        # Bayesian update at nearest inducing point
        prior_var = self.inducing_var[nearest]
        obs_precision = 1.0 / self.sigma_sq
        
        post_precision = 1.0 / prior_var + obs_precision
        post_var = 1.0 / post_precision
        post_mean = post_var * (self.inducing_y[nearest] / prior_var + y * obs_precision)
        
        self.inducing_y[nearest] = post_mean
        self.inducing_var[nearest] = post_var
        
        # Update noise estimate
        pred = self._predict_at(t)
        error = y - pred
        self.sigma_sq = 0.95 * self.sigma_sq + 0.05 * error**2
    
    def _predict_at(self, t: float) -> float:
        """Predict at a single time point."""
        k_star = np.exp(-0.5 * (self.inducing_t - t)**2 / self.length_scale**2)
        return np.sum(k_star * self.inducing_y) / (np.sum(k_star) + 1e-6)
    
    def _variance_at(self, t: float) -> float:
        """Posterior variance at a single time point."""
        k_star = np.exp(-0.5 * (self.inducing_t - t)**2 / self.length_scale**2)
        weights = k_star / (np.sum(k_star) + 1e-6)
        
        # Variance from inducing points plus interpolation uncertainty
        var_from_inducing = np.sum(weights**2 * self.inducing_var)
        
        # Distance-based uncertainty
        min_dist = np.min(np.abs(self.inducing_t - t))
        distance_var = 1.0 - np.exp(-0.5 * min_dist**2 / self.length_scale**2)
        
        return var_from_inducing + distance_var + self.sigma_sq
    
    def predict(self, horizon: int) -> Prediction:
        future_t = self.t + horizon
        mean = self._predict_at(future_t)
        var = self._variance_at(future_t)
        return Prediction(mean=mean, variance=var)
    
    def log_likelihood(self, y: float) -> float:
        pred = self.predict(horizon=1)
        return -0.5 * np.log(2 * np.pi * pred.variance) - (y - pred.mean)**2 / (2 * pred.variance)
    
    def reset(self, partial: float = 1.0) -> None:
        self.inducing_y *= partial
        self.inducing_var = partial * self.inducing_var + (1 - partial) * 1.0
    
    def epistemic_value(self) -> float:
        """Epistemic value from posterior uncertainty."""
        future_t = self.t + 1
        var = self._variance_at(future_t)
        
        # High variance = high epistemic value
        return 0.5 * np.log(var / self.sigma_sq + 1)
    
    @property
    def n_parameters(self) -> int:
        return self.n_inducing * 2 + 2
    
    @property
    def group(self) -> str:
        return "special"
```

### 9.2 Mixture of Experts

**Rationale**: Uncertainty about which expert is active provides natural epistemic value.

```python
class MixtureOfExpertsModel(TemporalModel):
    """
    Mixture of experts: combines multiple simple models.
    
    Epistemic value from uncertainty about which expert is appropriate.
    """
    
    def __init__(self, n_experts: int = 3):
        self.n_experts = n_experts
        
        # Expert parameters (each is a simple AR(1))
        self.expert_phi = np.random.uniform(0.5, 0.99, n_experts)
        self.expert_mu = np.zeros(n_experts)
        
        # Dirichlet posterior on mixture weights
        self.alpha = np.ones(n_experts)  # Symmetric prior
        
        self.sigma_sq = 1.0
        self.last_y = 0.0
        self.n_obs = 0
    
    def _expert_predictions(self) -> np.ndarray:
        """Get predictions from each expert."""
        preds = np.zeros(self.n_experts)
        for i in range(self.n_experts):
            deviation = self.last_y - self.expert_mu[i]
            preds[i] = self.expert_mu[i] + self.expert_phi[i] * deviation
        return preds
    
    def _mixture_weights(self) -> np.ndarray:
        """Posterior mean mixture weights."""
        return self.alpha / np.sum(self.alpha)
    
    def update(self, y: float, t: int) -> None:
        if self.n_obs > 0:
            # Get expert predictions
            preds = self._expert_predictions()
            
            # Compute likelihoods
            errors = y - preds
            log_likes = -0.5 * errors**2 / self.sigma_sq
            
            # Update Dirichlet posterior (multiplicative update)
            likes = np.exp(log_likes - np.max(log_likes))
            self.alpha += likes / (np.sum(likes) + 1e-10)
            
            # Update expert parameters
            weights = self._mixture_weights()
            for i in range(self.n_experts):
                if weights[i] > 0.1:  # Only update active experts
                    # Update mu
                    self.expert_mu[i] = 0.999 * self.expert_mu[i] + 0.001 * y
                    
                    # Update phi (gradient step)
                    deviation = self.last_y - self.expert_mu[i]
                    if abs(deviation) > 1e-6:
                        grad = -errors[i] * deviation / self.sigma_sq
                        self.expert_phi[i] -= 0.01 * grad
                        self.expert_phi[i] = np.clip(self.expert_phi[i], 0.0, 0.999)
            
            # Update variance
            combined_pred = np.sum(weights * preds)
            error = y - combined_pred
            self.sigma_sq = 0.95 * self.sigma_sq + 0.05 * error**2
        
        self.last_y = y
        self.n_obs += 1
    
    def predict(self, horizon: int) -> Prediction:
        preds = self._expert_predictions()
        weights = self._mixture_weights()
        
        # Project forward
        future_preds = np.zeros(self.n_experts)
        for i in range(self.n_experts):
            y = self.last_y
            for _ in range(horizon):
                y = self.expert_mu[i] + self.expert_phi[i] * (y - self.expert_mu[i])
            future_preds[i] = y
        
        mean = np.sum(weights * future_preds)
        
        # Variance includes model uncertainty
        within_var = self.sigma_sq * horizon
        between_var = np.sum(weights * (future_preds - mean)**2)
        
        return Prediction(mean=mean, variance=within_var + between_var)
    
    def log_likelihood(self, y: float) -> float:
        pred = self.predict(horizon=1)
        return -0.5 * np.log(2 * np.pi * pred.variance) - (y - pred.mean)**2 / (2 * pred.variance)
    
    def reset(self, partial: float = 1.0) -> None:
        self.alpha = partial * self.alpha + (1 - partial) * np.ones(self.n_experts)
    
    def epistemic_value(self) -> float:
        """Entropy of mixture weights indicates epistemic uncertainty."""
        weights = self._mixture_weights()
        
        # Entropy (higher = more uncertainty)
        entropy = -np.sum(weights * np.log(weights + 1e-10))
        max_entropy = np.log(self.n_experts)
        
        # Normalise to [0, 1] range, then scale
        normalised = entropy / max_entropy
        return normalised * 0.5  # Scale factor
    
    @property
    def n_parameters(self) -> int:
        return 3 * self.n_experts + 1
    
    @property
    def group(self) -> str:
        return "special"
```

### 9.3 Hierarchical Mean Reversion

**Rationale**: Uncertainty at multiple levels provides rich epistemic structure.

```python
class HierarchicalMeanReversionModel(TemporalModel):
    """
    Hierarchical mean reversion with uncertainty at multiple levels.
    
    Level 1: Fast dynamics around local mean
    Level 2: Slow drift of local mean toward global mean
    Level 3: Very slow adaptation of global mean
    
    Epistemic value from uncertainty at each level.
    """
    
    def __init__(self):
        # Level 1: Fast mean reversion
        self.local_mean = 0.0
        self.local_mean_var = 1.0  # Uncertainty
        self.phi_fast = 0.8
        
        # Level 2: Slow mean reversion
        self.global_mean = 0.0
        self.global_mean_var = 2.0
        self.phi_slow = 0.99
        
        # Observation noise
        self.sigma_sq = 1.0
        
        self.last_y = 0.0
        self.n_obs = 0
    
    def update(self, y: float, t: int) -> None:
        if self.n_obs > 0:
            # Prediction from current state
            pred = self.local_mean + self.phi_fast * (self.last_y - self.local_mean)
            error = y - pred
            
            # Update Level 1: local mean
            # Kalman-style update
            obs_precision = 1.0 / self.sigma_sq
            prior_precision = 1.0 / self.local_mean_var
            
            post_precision = prior_precision + obs_precision * (1 - self.phi_fast)**2
            post_var = 1.0 / post_precision
            
            innovation_to_mean = error * (1 - self.phi_fast)
            self.local_mean += post_var * obs_precision * innovation_to_mean
            self.local_mean_var = post_var + 0.01  # Process noise
            
            # Update Level 2: global mean pulls local mean
            mean_error = self.local_mean - self.global_mean
            self.global_mean += 0.01 * mean_error
            self.global_mean_var = 0.99 * self.global_mean_var + 0.01 * mean_error**2
            
            # Local mean reverts to global
            self.local_mean = self.phi_slow * self.local_mean + (1 - self.phi_slow) * self.global_mean
            
            # Update observation noise
            self.sigma_sq = 0.95 * self.sigma_sq + 0.05 * error**2
        
        self.last_y = y
        self.n_obs += 1
    
    def predict(self, horizon: int) -> Prediction:
        # Fast reversion to local mean, which slowly reverts to global
        y = self.last_y
        local = self.local_mean
        
        for _ in range(horizon):
            y = local + self.phi_fast * (y - local)
            local = self.phi_slow * local + (1 - self.phi_slow) * self.global_mean
        
        # Variance includes all levels of uncertainty
        var = (
            self.sigma_sq * horizon +
            self.local_mean_var * (1 - self.phi_fast**(2*horizon)) +
            self.global_mean_var * (1 - self.phi_slow**(2*horizon)) * 0.1
        )
        
        return Prediction(mean=y, variance=var)
    
    def log_likelihood(self, y: float) -> float:
        pred = self.predict(horizon=1)
        return -0.5 * np.log(2 * np.pi * pred.variance) - (y - pred.mean)**2 / (2 * pred.variance)
    
    def reset(self, partial: float = 1.0) -> None:
        self.local_mean_var = partial * self.local_mean_var + (1 - partial) * 1.0
        self.global_mean_var = partial * self.global_mean_var + (1 - partial) * 2.0
    
    def epistemic_value(self) -> float:
        """Epistemic value from uncertainty at all levels."""
        level1_info = 0.5 * np.log(1 + self.local_mean_var / self.sigma_sq)
        level2_info = 0.5 * np.log(1 + self.global_mean_var / self.local_mean_var)
        
        return level1_info + 0.5 * level2_info  # Discount higher levels
    
    @property
    def n_parameters(self) -> int:
        return 5
    
    @property
    def group(self) -> str:
        return "reversion"
```

---

## 10. Model Factory

```python
from typing import List
from aegis.config import AEGISConfig
from aegis.models.base import TemporalModel


def create_model_bank(config: AEGISConfig) -> List[TemporalModel]:
    """
    Create the standard model bank for AEGIS.
    
    Returns models organised by group for a single scale.
    """
    models = []
    
    # Persistence
    models.append(RandomWalkModel())
    models.append(LocalLevelModel(alpha=0.1))
    models.append(LocalLevelModel(alpha=0.3))
    
    # Trend
    models.append(LocalTrendModel())
    models.append(DampedTrendModel(phi=0.9))
    models.append(DampedTrendModel(phi=0.98))
    
    # Reversion
    models.append(MeanReversionModel(phi=0.9))
    models.append(MeanReversionModel(phi=0.99))
    models.append(AsymmetricMeanReversionModel())
    models.append(ThresholdARModel())
    
    # Periodic
    models.append(OscillatorBankModel(periods=config.oscillator_periods))
    for period in config.seasonal_periods:
        models.append(SeasonalDummyModel(period=period))
    
    # Dynamic
    models.append(AR2Model())
    models.append(MA1Model())
    
    # Special
    models.append(JumpDiffusionModel())
    models.append(ChangePointModel())
    
    # Variance
    models.append(VolatilityTrackerModel())
    
    # FEP-Native (optional, for Phase 2)
    if config.use_epistemic_value:
        models.append(MixtureOfExpertsModel(n_experts=3))
        models.append(HierarchicalMeanReversionModel())
    
    return models


def get_model_groups() -> dict:
    """Return mapping of model names to groups."""
    return {
        "RandomWalkModel": "persistence",
        "LocalLevelModel": "persistence",
        "LocalTrendModel": "trend",
        "DampedTrendModel": "trend",
        "MeanReversionModel": "reversion",
        "AsymmetricMeanReversionModel": "reversion",
        "ThresholdARModel": "reversion",
        "OscillatorBankModel": "periodic",
        "SeasonalDummyModel": "periodic",
        "AR2Model": "dynamic",
        "MA1Model": "dynamic",
        "JumpDiffusionModel": "special",
        "ChangePointModel": "special",
        "VolatilityTrackerModel": "variance",
        "LevelDependentVolModel": "variance",
        "GaussianProcessApproxModel": "special",
        "MixtureOfExpertsModel": "special",
        "HierarchicalMeanReversionModel": "reversion"
    }
```

---

*End of Appendix B*
