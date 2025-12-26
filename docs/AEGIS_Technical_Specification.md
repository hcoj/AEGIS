# AEGIS: Active Epistemic Generative Inference System

## Technical Specification

**A System for Multi-stream Time Series Prediction Using Structured Model Ensembles with Expected Free Energy Weighting**

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Theoretical Foundation](#2-theoretical-foundation)
3. [Architecture](#3-architecture)
4. [Core Interfaces](#4-core-interfaces)
5. [Model Combination](#5-model-combination)
6. [Multi-Scale Processing](#6-multi-scale-processing)
7. [Cross-Stream Integration](#7-cross-stream-integration)
8. [Uncertainty Quantification](#8-uncertainty-quantification)
9. [Regime Adaptation](#9-regime-adaptation)
10. [Configuration](#10-configuration)
11. [API Overview](#11-api-overview)

**Appendices (separate documents):**
- Appendix A: Core Implementation
- Appendix B: Model Specifications
- Appendix C: Implementation Plan
- Appendix D: Signal Taxonomy

---

## 1. Introduction

### 1.1 Design Philosophy

**Core principle**: Time series exhibit a finite vocabulary of behaviours. Rather than searching for structure, we enumerate known structures and weight them by both predictive performance and epistemic value.

The behaviours that matter:
- **Persistence**: Tomorrow looks like today
- **Trend**: Consistent directional drift (with optional damping)
- **Mean-reversion**: Pull toward a centre (symmetric or asymmetric)
- **Oscillation**: Cyclical patterns at various frequencies
- **Volatility clustering**: Time-varying uncertainty
- **Threshold effects**: Regime-dependent dynamics
- **Jumps**: Occasional large discrete moves
- **Moving average effects**: Shock propagation

These structures are well-understood. Each has a known mathematical form with a small number of parameters. We don't need to discover that mean-reversion exists; we need to detect whether it's present and estimate its strength.

### 1.2 The AEGIS Approach

AEGIS combines two insights:

**Structured models instead of black boxes**: Each model has 1-4 learnable parameters, not dozens. Learning a decay rate is tractable. Learning a full dynamics matrix from scratch is not.

**Expected free energy instead of pure accuracy**: Traditional ensemble methods weight models by past accuracy alone. AEGIS incorporates epistemic value—how much the next observation would inform us about each model's parameters. This accelerates adaptation during regime changes and naturally balances exploitation (using what we know) with exploration (learning what we don't).

### 1.3 Key Capabilities

AEGIS handles:
- Single and multiple correlated time series
- Mixed frequencies (daily + monthly data)
- Level prediction and uncertainty estimation
- Regime changes and structural breaks
- Heavy-tailed observations
- Threshold and asymmetric dynamics
- Jump processes
- Contemporaneous cross-stream relationships

AEGIS does not handle:
- Truly novel dynamics outside the model vocabulary
- Categorical or non-numeric data
- Chaotic dynamics (beyond short-term local approximation)

### 1.4 Implementation Phases

AEGIS is implemented in two phases:

**Phase 1** implements the core architecture with log-likelihood model weighting. This provides a fully functional forecasting system using accuracy-based model combination.

**Phase 2** extends Phase 1 with expected free energy weighting. Models that track parameter uncertainty contribute epistemic value to the weighting, improving adaptation during non-stationary periods.

Both phases use identical model interfaces. The difference is whether `epistemic_value()` contributes to model weights.

---

## 2. Theoretical Foundation

### 2.1 The Free Energy Principle

The Free Energy Principle (FEP) provides a unifying framework for understanding adaptive systems. Under FEP, systems that persist must minimise surprise—the negative log-probability of observations under their generative model.

For forecasting, this translates to: prefer models that assign high probability to observed data.

But pure surprise minimisation is backward-looking. The expected free energy extends this to forward-looking model selection.

### 2.2 Expected Free Energy

Expected free energy decomposes into two terms:

$$G = \underbrace{-\mathbb{E}[\ln P(o|\pi)]}_{\text{pragmatic value}} - \underbrace{\mathbb{E}[D_{KL}[P(\theta|o) \| P(\theta)]]}_{\text{epistemic value}}$$

**Pragmatic value** measures expected predictive accuracy. Models with tighter predictive distributions score higher.

**Epistemic value** measures expected information gain. Models whose parameters would be informed by the next observation score higher.

### 2.3 Application to Model Weighting

For a model $m$ with parameters $\theta_m$:

**Pragmatic value:**
$$\mathcal{P}_m = \mathbb{E}_{p(y|m)}[\ln p(y|\theta_m)]$$

For Gaussian predictions with variance $\sigma^2_m$:
$$\mathcal{P}_m = -\frac{1}{2}\ln(2\pi\sigma^2_m) - \frac{1}{2}$$

**Epistemic value:**
$$\mathcal{E}_m = \mathbb{E}_{p(y|m)}[D_{KL}[p(\theta_m|y) \| p(\theta_m)]]$$

This measures how much the posterior over parameters would change given the next observation.

**Combined weighting:**

Model weights are computed via softmax over expected free energy:

$$w_m = \frac{\exp(\mathcal{P}_m + \alpha \cdot \mathcal{E}_m)}{\sum_{m'} \exp(\mathcal{P}_{m'} + \alpha \cdot \mathcal{E}_{m'})}$$

where $\alpha \geq 0$ controls the weight on epistemic value.

- When $\alpha = 0$: Pure pragmatic weighting (Phase 1 behaviour)
- When $\alpha > 0$: Epistemic bonus for models poised to learn (Phase 2 behaviour)

### 2.4 Why Epistemic Value Matters

Consider a regime change. All models have stale parameters. Under pure pragmatic weighting, the model that happened to be least wrong recently dominates—even if it's learning nothing useful about the new regime.

With epistemic weighting, models whose parameters are uncertain but would become informed by new data receive elevated weight. This accelerates adaptation.

During stationary periods, epistemic values shrink as parameters converge, and the system naturally reverts to accuracy-weighted averaging.

### 2.5 Computing Epistemic Value

For models with Gaussian posteriors over parameters, epistemic value has a closed form.

If $\theta \sim \mathcal{N}(\mu_\theta, \Sigma_\theta)$ and the model is linear in parameters:

$$\mathcal{E} = \frac{1}{2} \ln\left(1 + \frac{\phi(x)^\top \Sigma_\theta \phi(x)}{\sigma^2}\right)$$

where $\phi(x)$ is the feature vector and $\sigma^2$ is observation noise.

Epistemic value is high when:
- Parameter uncertainty ($\Sigma_\theta$) is large
- The current state ($\phi(x)$) is informative about parameters
- Observation noise ($\sigma^2$) is low

For nonlinear models, we use Laplace approximations or empirical estimates.

---

## 3. Architecture

### 3.1 System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         Raw Input                               │
│                            │                                    │
│                            ▼                                    │
│              ┌─────────────────────────┐                        │
│              │   Multi-Scale Layer     │                        │
│              │   (returns at various   │                        │
│              │    lookback periods)    │                        │
│              └─────────────────────────┘                        │
│                            │                                    │
│         ┌──────────────────┼──────────────────┐                 │
│         ▼                  ▼                  ▼                 │
│   ┌───────────┐      ┌───────────┐      ┌───────────┐          │
│   │  Scale 1  │      │  Scale 4  │      │  Scale 16 │  ...     │
│   │  Models   │      │  Models   │      │  Models   │          │
│   └─────┬─────┘      └─────┬─────┘      └─────┬─────┘          │
│         │                  │                  │                 │
│         └──────────────────┼──────────────────┘                 │
│                            ▼                                    │
│              ┌─────────────────────────┐                        │
│              │   Prediction Combiner   │                        │
│              │   (EFE-weighted avg)    │                        │
│              └─────────────────────────┘                        │
│                            │                                    │
│                            ▼                                    │
│              ┌─────────────────────────┐                        │
│              │  Cross-Stream Layer     │                        │
│              │  (residual regression)  │                        │
│              └─────────────────────────┘                        │
│                            │                                    │
│                            ▼                                    │
│              ┌─────────────────────────┐                        │
│              │  Uncertainty Layer      │                        │
│              │  (calibrated intervals) │                        │
│              └─────────────────────────┘                        │
│                            │                                    │
│                            ▼                                    │
│                     Final Prediction                            │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Components

**Multi-Scale Layer**: Converts raw prices/levels into returns at multiple lookback periods. A 16-step return exposes slow mean-reversion as a direct signal, avoiding the need for eigenvalues near 1.

**Per-Scale Model Bank**: At each scale, temporal models make predictions. Each model has a specific structure with learnable parameters.

**Prediction Combiner**: Weights model predictions by expected free energy. In Phase 1, this reduces to log-likelihood weighting. In Phase 2, epistemic value contributes.

**Cross-Stream Layer**: After per-stream temporal modelling, captures relationships between streams via residual regression.

**Uncertainty Layer**: Combines Gaussian variance estimates with quantile tracking for calibrated prediction intervals.

### 3.3 Data Flow

1. Raw observation arrives
2. Multi-scale layer computes returns at each lookback period
3. Each scale's model bank updates parameters and makes predictions
4. Combiner weights predictions by expected free energy
5. Cross-stream layer adjusts predictions using information from other streams
6. Uncertainty layer produces calibrated prediction intervals
7. Final prediction and uncertainty interval produced

---

## 4. Core Interfaces

### 4.1 Prediction

```python
@dataclass
class Prediction:
    """Point prediction with uncertainty."""
    mean: float
    variance: float
    
    @property
    def std(self) -> float:
        return np.sqrt(self.variance)
    
    def interval(self, level: float = 0.95) -> tuple[float, float]:
        """Gaussian confidence interval."""
        z = norm.ppf(0.5 + level / 2)
        return self.mean - z * self.std, self.mean + z * self.std
```

### 4.2 Temporal Model

All temporal models implement this abstract base class:

```python
class TemporalModel(ABC):
    """
    Abstract base class for all temporal models.
    
    Core interface (must implement):
        update, predict, log_likelihood, reset
    
    EFE interface (have working defaults):
        pragmatic_value, epistemic_value
    """
    
    # -------------------------------------------------------------------------
    # Core forecasting interface
    # -------------------------------------------------------------------------
    
    @abstractmethod
    def update(self, y: float, t: int) -> None:
        """
        Update model state with new observation.
        
        Args:
            y: Observed value
            t: Time index
        """
        ...
    
    @abstractmethod
    def predict(self, horizon: int) -> Prediction:
        """
        Generate prediction for given horizon.
        
        Args:
            horizon: Steps ahead to predict
            
        Returns:
            Prediction with mean and variance
        """
        ...
    
    @abstractmethod
    def log_likelihood(self, y: float) -> float:
        """
        Compute log-likelihood of observation under current predictive distribution.
        
        Args:
            y: Observed value
            
        Returns:
            Log probability density at y
        """
        ...
    
    @abstractmethod
    def reset(self, partial: float = 1.0) -> None:
        """
        Reset parameters toward priors.
        
        Used after regime breaks to accelerate adaptation.
        
        Args:
            partial: Interpolation weight. 1.0 = full reset, 0.0 = no change
        """
        ...
    
    # -------------------------------------------------------------------------
    # Expected free energy interface
    # -------------------------------------------------------------------------
    
    def pragmatic_value(self) -> float:
        """
        Expected log-likelihood under own predictive distribution.
        
        Default implementation assumes Gaussian predictive distribution.
        
        Returns:
            Expected log-likelihood (higher is better)
        """
        pred = self.predict(horizon=1)
        return -0.5 * np.log(2 * np.pi * pred.variance) - 0.5
    
    def epistemic_value(self) -> float:
        """
        Expected information gain about parameters from next observation.
        
        Default returns 0.0. Override in models that track parameter uncertainty.
        
        Returns:
            Expected information gain (higher means more to learn)
        """
        return 0.0
    
    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------
    
    @property
    def name(self) -> str:
        """Human-readable model name."""
        return self.__class__.__name__
    
    @property
    def n_parameters(self) -> int:
        """Number of learnable parameters."""
        return 0
    
    @property
    def group(self) -> str:
        """Model group: persistence, trend, reversion, periodic, dynamic, special, variance."""
        return "unknown"
```

### 4.3 Model Combiner

```python
class ModelCombiner(ABC):
    """
    Abstract base class for model combination strategies.
    """
    
    @abstractmethod
    def update(self, models: list[TemporalModel], y_observed: float) -> None:
        """
        Update combination weights based on observed value.
        
        Args:
            models: List of temporal models
            y_observed: The observed value
        """
        ...
    
    @abstractmethod
    def get_weights(self) -> np.ndarray:
        """
        Get current model weights.
        
        Returns:
            Normalised weight vector summing to 1.0
        """
        ...
    
    @abstractmethod
    def combine_predictions(
        self, 
        predictions: list[Prediction]
    ) -> Prediction:
        """
        Combine model predictions into single prediction.
        
        Uses law of total variance to account for model disagreement.
        
        Args:
            predictions: Predictions from each model
            
        Returns:
            Combined prediction
        """
        ...
```

### 4.4 Stream Manager

```python
class StreamManager(ABC):
    """
    Abstract base class for managing a single data stream.
    """
    
    @abstractmethod
    def observe(self, y: float, t: int) -> None:
        """
        Process new observation.
        
        Args:
            y: Observed value
            t: Time index
        """
        ...
    
    @abstractmethod
    def predict(self, horizon: int) -> Prediction:
        """
        Generate prediction for given horizon.
        
        Args:
            horizon: Steps ahead
            
        Returns:
            Combined prediction from all models and scales
        """
        ...
    
    @abstractmethod
    def get_diagnostics(self) -> dict:
        """
        Get diagnostic information about model weights and behaviour.
        
        Returns:
            Dictionary with model weights, group weights, etc.
        """
        ...
```

### 4.5 Break Detector

```python
class BreakDetector(ABC):
    """
    Abstract base class for regime break detection.
    """
    
    @abstractmethod
    def update(self, error: float) -> bool:
        """
        Update detector with prediction error.
        
        Args:
            error: Prediction error (observed - predicted)
            
        Returns:
            True if break detected, False otherwise
        """
        ...
    
    @abstractmethod
    def reset(self) -> None:
        """Reset detector state after break handling."""
        ...
```

---

## 5. Model Combination

### 5.1 Expected Free Energy Combiner

The combiner maintains cumulative scores for each model and computes weights via softmax:

```
Algorithm: EFE Model Combination

For each observation y_t:
    1. For each model m:
        a. Compute pragmatic value P_m = E[log p(y|m)]
        b. Compute epistemic value E_m (model-specific)
        c. Update cumulative score: S_m ← λ·S_m + P_m + α·E_m
    
    2. Compute weights: w_m = softmax(S / T)
    
    3. Combine predictions:
        mean = Σ w_m · μ_m
        variance = Σ w_m · [σ²_m + (μ_m - mean)²]  # Law of total variance
    
    4. Update models with observation
```

Parameters:
- $\lambda$: Forgetting factor (default 0.99)
- $\alpha$: Epistemic weight (0 for Phase 1, typically 1.0 for Phase 2)
- $T$: Temperature for softmax (default 1.0)

### 5.2 Phase 1 vs Phase 2 Behaviour

**Phase 1** ($\alpha = 0$):
- Weights based purely on accumulated log-likelihood
- Equivalent to Bayesian model averaging with exponential forgetting
- All models use default `epistemic_value() → 0.0`

**Phase 2** ($\alpha > 0$):
- Weights include epistemic bonus for uncertain models
- Models with parameter tracking override `epistemic_value()`
- Accelerated adaptation during regime changes

### 5.3 Model Groups

Models are organised into groups for diagnostic purposes:

| Group | Description |
|-------|-------------|
| Persistence | RandomWalk, LocalLevel |
| Trend | LocalTrend, DampedTrend |
| Reversion | MeanReversion, AsymmetricMR, ThresholdAR |
| Periodic | OscillatorBank, SeasonalDummy |
| Dynamic | AR(2), MA(1), ARMA |
| Special | JumpDiffusion, ChangePoint |
| Variance | VolatilityTracker, LevelDependentVol, QuantileTracker |

---

## 6. Multi-Scale Processing

### 6.1 Scale Definition

A scale $s$ corresponds to computing returns over $s$ periods:

$$r_t^{(s)} = y_t - y_{t-s}$$

Default scales: [1, 2, 4, 8, 16, 32, 64]

### 6.2 Rationale

Multi-scale decomposition serves several purposes:

1. **Exposes slow dynamics**: A process with φ = 0.99 (half-life ~69 steps) appears nearly unpredictable at scale 1. At scale 64, its mean-reverting structure becomes visible.

2. **Separates timescales**: Short scales capture momentum; long scales capture reversion. Different models dominate at different scales.

3. **Reduces parameter sensitivity**: Rather than estimating φ near 1.0 precisely, we detect mean-reversion at the scale where it's most apparent.

### 6.3 Per-Scale Model Banks

Each scale maintains a complete model bank. The same model types operate at each scale, but they capture different temporal structure.

### 6.4 Cross-Scale Combination

Scale predictions are combined based on:
- Prediction horizon (short horizons weight short scales)
- Historical accuracy at each scale
- Epistemic value (in Phase 2)

---

## 7. Cross-Stream Integration

### 7.1 Residual Regression

After per-stream temporal modelling, cross-stream regression captures relationships:

$$\hat{r}_t^{(i)} = \hat{r}_{t,\text{temporal}}^{(i)} + \sum_{j \neq i} \sum_{\ell=0}^{L} \beta_{ij\ell} \cdot r_{t-\ell}^{(j)}$$

The regression operates on residuals (observed minus temporal prediction) to avoid double-counting temporal structure.

### 7.2 Lag-0 Support

Cross-stream regression supports contemporaneous (lag-0) relationships:

$$\beta_{ij0} \cdot r_t^{(j)}$$

This requires that stream $j$ is observed before stream $i$ within the same period.

### 7.3 Factor Model

For many streams (>3), online PCA extracts common factors:

$$\mathbf{y}_t = \Lambda \mathbf{f}_t + \boldsymbol{\epsilon}_t$$

Factor dynamics are modelled separately, and factor predictions inform per-stream predictions.

---

## 8. Uncertainty Quantification

### 8.1 Base Variance

Each model provides a variance estimate. The combiner produces a weighted average using the law of total variance:

$$\sigma^2 = \sum_m w_m [\sigma^2_m + (\hat{y}_m - \hat{y})^2]$$

This accounts for both within-model uncertainty and between-model disagreement.

### 8.2 Volatility Scaling

A volatility tracker (EWMA) monitors squared residuals and scales variance estimates by current volatility relative to long-run volatility.

### 8.3 Quantile Calibration

Gaussian intervals may undercover for heavy-tailed data. The quantile tracker monitors empirical coverage and adjusts interval widths:

```
Algorithm: Quantile Calibration

For target coverage (e.g., 95%):
    1. Track empirical quantiles of standardised residuals
    2. Adjust interval multipliers based on actual coverage
    3. Return calibrated intervals: [μ + q_low·σ, μ + q_high·σ]
```

### 8.4 Jump Risk Premium

The jump-diffusion model contributes variance reflecting jump risk, even during quiet periods. This provides honest uncertainty that anticipates rare events.

---

## 9. Regime Adaptation

### 9.1 Break Detection

Large cumulative prediction errors trigger break detection via CUSUM:

$$\text{CUSUM}_t = \max(0, \text{CUSUM}_{t-1} + |e_t| - k)$$

where $k$ is a threshold (default: 1.5σ).

When $\text{CUSUM}_t > h$ (default: 3σ), a break is declared.

### 9.2 Post-Break Adaptation

After a detected break:
1. Forgetting factor temporarily drops (e.g., 0.99 → 0.9)
2. Parameters partially reset toward priors
3. Model weights reset toward uniform

This accelerates learning in the new regime.

### 9.3 Epistemic Value and Breaks

In Phase 2, epistemic value naturally increases after breaks (parameter uncertainty rises). This provides a complementary mechanism: models that can learn quickly from new data receive elevated weight during transitions.

---

## 10. Configuration

### 10.1 Core Configuration

```python
@dataclass
class AEGISConfig:
    """Configuration for AEGIS system."""
    
    # Phase selection
    use_epistemic_value: bool = False  # False for Phase 1, True for Phase 2
    epistemic_weight: float = 1.0      # α parameter
    
    # Scales
    scales: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32, 64])
    
    # Oscillator bank frequencies (in periods)
    oscillator_periods: list[int] = field(
        default_factory=lambda: [4, 8, 16, 32, 64, 128, 256]
    )
    
    # Seasonal periods
    seasonal_periods: list[int] = field(default_factory=lambda: [7, 12])
    
    # Model combination
    likelihood_forget: float = 0.99
    temperature: float = 1.0
    
    # Volatility tracking
    volatility_decay: float = 0.94
    
    # Cross-stream
    cross_stream_lags: int = 5
    include_lag_zero: bool = False
    n_factors: int = 3
    
    # Regime adaptation
    break_threshold: float = 3.0
    post_break_forget: float = 0.9
    post_break_duration: int = 50
    
    # Calibration
    target_coverage: float = 0.95
    use_quantile_calibration: bool = True
    
    # Robustness
    outlier_threshold: float = 5.0
    min_variance: float = 1e-10
```

### 10.2 Phase Configuration

**Phase 1 (accuracy-based weighting):**
```python
config = AEGISConfig(
    use_epistemic_value=False
)
```

**Phase 2 (expected free energy weighting):**
```python
config = AEGISConfig(
    use_epistemic_value=True,
    epistemic_weight=1.0
)
```

### 10.3 Hyperparameter Guidance

| Parameter | Default | Guidance |
|-----------|---------|----------|
| `epistemic_weight` | 1.0 | 0.5-2.0 typical; higher for frequent regime changes |
| `temperature` | 1.0 | <1 for sharper weights; >1 for softer ensemble |
| `likelihood_forget` | 0.99 | Lower for faster adaptation, higher for stability |
| `break_threshold` | 3.0 | Lower for more sensitive detection |

---

## 11. API Overview

### 11.1 Basic Usage

```python
from aegis import AEGIS, AEGISConfig

# Phase 1 configuration (accuracy-based)
config = AEGISConfig(use_epistemic_value=False)
system = AEGIS(config)

# Add streams
system.add_stream("price")
system.add_stream("volume")

# Process observations
for t, (price, volume) in enumerate(data):
    system.observe("price", price, t)
    system.observe("volume", volume, t)
    
    # Get prediction
    pred = system.predict("price", horizon=5)
    print(f"Price in 5 steps: {pred.mean:.2f} ± {pred.std:.2f}")
    
    # Get calibrated interval
    lower, upper = pred.interval(0.95)
    print(f"95% interval: [{lower:.2f}, {upper:.2f}]")
```

### 11.2 Phase 2 Usage

```python
# Phase 2 configuration (expected free energy)
config = AEGISConfig(
    use_epistemic_value=True,
    epistemic_weight=1.0
)
system = AEGIS(config)

# Usage identical to Phase 1
# System automatically uses epistemic value in weighting
```

### 11.3 Diagnostics

```python
# Get diagnostic information
diag = system.get_diagnostics("price")

# Model weights by group
print("Group weights:")
for group, weight in diag["group_weights"].items():
    print(f"  {group}: {weight:.3f}")

# Top individual models
print("Top models:")
for model, weight in diag["top_models"][:5]:
    print(f"  {model}: {weight:.3f}")

# Epistemic values (Phase 2 only)
if "epistemic_values" in diag:
    print("Epistemic values:")
    for model, ev in diag["epistemic_values"].items():
        print(f"  {model}: {ev:.4f}")
```

---

## References

1. Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*, 11(2), 127-138.

2. Parr, T., & Friston, K. J. (2019). Generalised free energy and active inference. *Biological Cybernetics*, 113(5), 495-513.

3. Holt, C. C. (1957). Forecasting seasonals and trends by exponentially weighted moving averages. *ONR Research Memorandum*, 52.

4. Tong, H. (1983). *Threshold models in non-linear time series analysis*. Springer.

5. Merton, R. C. (1976). Option pricing when underlying stock returns are discontinuous. *Journal of Financial Economics*, 3(1-2), 125-144.

---

*End of Technical Specification*

*See Appendices A-D for implementation details, model specifications, implementation plan, and signal taxonomy.*
