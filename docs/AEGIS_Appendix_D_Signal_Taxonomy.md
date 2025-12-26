# AEGIS Appendix D: Signal Taxonomy

## Expected Behaviour Across Signal Types

---

## Contents

1. [Overview](#1-overview)
2. [Deterministic Signals](#2-deterministic-signals)
3. [Simple Stochastic Processes](#3-simple-stochastic-processes)
4. [Composite Signals](#4-composite-signals)
5. [Non-Stationary and Regime-Changing](#5-non-stationary-and-regime-changing)
6. [Heavy-Tailed Signals](#6-heavy-tailed-signals)
7. [Multi-Scale Structure](#7-multi-scale-structure)
8. [Multiple Correlated Series](#8-multiple-correlated-series)
9. [Domain-Specific Applications](#9-domain-specific-applications)
10. [Adversarial and Edge Cases](#10-adversarial-and-edge-cases)
11. [Summary Tables](#11-summary-tables)

---

## 1. Overview

### 1.1 Purpose

This appendix characterises AEGIS's expected behaviour across signal types. For each class:
- Which models dominate
- Prediction quality expectations
- Uncertainty calibration
- Phase 1 vs Phase 2 differences

### 1.2 Evaluation Metrics

**Point Prediction Quality:**
- MAE: Mean Absolute Error
- RMSE: Root Mean Squared Error
- Relative to naive baseline (random walk)

**Uncertainty Calibration:**
- Coverage: Fraction of observations within prediction interval
- Interval width: Sharpness of uncertainty estimates

**Adaptation Speed:**
- Recovery time after regime changes
- Observations until dominant model stabilises

### 1.3 Rating Scale

| Rating | Meaning |
|--------|---------|
| Excellent | Near-optimal; matches theoretical best |
| Good | Substantially better than naive baseline |
| Moderate | Better than naive, room for improvement |
| Poor | Limited improvement over baseline |

---

## 2. Deterministic Signals

### 2.1 Constant Value

**Signal:** $y_t = c$

**Domain examples:** Setpoint of controlled process; calibration signal; baseline reference.

**AEGIS behaviour:**

| Aspect | Expectation |
|--------|-------------|
| Dominant model | LocalLevel (α=0.1) |
| Prediction quality | Excellent |
| Uncertainty | Appropriately tight |
| Convergence | 20-50 observations |

**Mechanism:** LocalLevel converges to $c$ with shrinking variance estimate. RandomWalk also predicts correctly but maintains higher variance.

**Phase 2 difference:** Minimal. Epistemic values shrink as parameters converge; system behaves similarly to Phase 1.

---

### 2.2 Linear Trend

**Signal:** $y_t = at + b$

**Domain examples:** Population growth (short term); linear depreciation; thermal drift.

**AEGIS behaviour:**

| Aspect | Expectation |
|--------|-------------|
| Dominant model | LocalTrend or DampedTrend (φ→1) |
| Prediction quality | Excellent |
| Uncertainty | Appropriate |
| Slope learning | 50-100 observations |

**Mechanism:** LocalTrend tracks slope via exponential smoothing. DampedTrend with high φ behaves similarly.

---

### 2.3 Sinusoidal

**Signal:** $y_t = A\sin(\omega t + \phi)$

**Domain examples:** Tidal measurements; circadian rhythms; seasonal temperature.

**AEGIS behaviour:**

| Aspect | Expectation |
|--------|-------------|
| Dominant model | OscillatorBank |
| Prediction quality | Excellent (if frequency in bank) |
| Uncertainty | Appropriately tight |
| Coefficient learning | 2-3 full cycles |

**Critical factor:** Frequency must be in or near the oscillator bank. Default bank covers periods 4-256.

**If frequency not in bank:** Performance degrades. Multiple oscillators may approximate, but predictions are less accurate.

---

### 2.4 Square Wave / Sharp Seasonal

**Signal:** $y_t = c_{s(t)}$ where $s(t)$ cycles through discrete values

**Domain examples:** Day-of-week retail patterns; shift schedules; on/off cycles.

**AEGIS behaviour:**

| Aspect | Expectation |
|--------|-------------|
| Dominant model | SeasonalDummy |
| Prediction quality | Excellent |
| Uncertainty | Good |
| Pattern learning | 3-5 full cycles |

**Key advantage:** SeasonalDummy captures sharp transitions directly, outperforming sinusoidal approximation.

---

### 2.5 Polynomial Trend

**Signal:** $y_t = at^2 + bt + c$

**Domain examples:** Accelerating growth; parabolic motion; some reaction kinetics.

**AEGIS behaviour:**

| Aspect | Expectation |
|--------|-------------|
| Dominant model | LocalTrend (adapting) |
| Prediction quality | Moderate |
| Short horizon | Good |
| Long horizon | Underestimates curvature |

**Limitation:** No model has polynomial structure. LocalTrend tracks instantaneous slope but underestimates acceleration.

**DampedTrend advantage:** For decelerating quadratics (negative $a$), DampedTrend provides better long-horizon forecasts.

---

## 3. Simple Stochastic Processes

### 3.1 White Noise

**Signal:** $y_t \sim \mathcal{N}(0, \sigma^2)$ i.i.d.

**Domain examples:** Measurement noise; thermal noise; baseline neural activity.

**AEGIS behaviour:**

| Aspect | Expectation |
|--------|-------------|
| Dominant model | RandomWalk (by likelihood) |
| Prediction quality | Optimal (predicts 0) |
| Uncertainty | Excellent calibration |
| Variance estimate | Converges to σ² |

**Mechanism:** All models learn to predict zero. RandomWalk dominates because its variance estimate matches true variance without overfitting.

---

### 3.2 Random Walk

**Signal:** $y_t = y_{t-1} + \epsilon_t$, $\epsilon_t \sim \mathcal{N}(0, \sigma^2)$

**Domain examples:** Stock prices (approximate); Brownian motion; accumulated measurement error.

**AEGIS behaviour:**

| Aspect | Expectation |
|--------|-------------|
| Dominant model | RandomWalk |
| Prediction quality | Excellent (optimal) |
| Uncertainty | Excellent |
| Multi-step variance | Linear in horizon |

**Key point:** This is the canonical baseline. Other models cannot beat RandomWalk on true random walk data.

---

### 3.3 AR(1) Mean-Reverting

**Signal:** $y_t = \phi y_{t-1} + \epsilon_t$, $|\phi| < 1$

**Domain examples:** Temperature deviations; interest rate spreads; homeostatic systems.

**AEGIS behaviour:**

| Aspect | Expectation |
|--------|-------------|
| Dominant model | MeanReversion |
| Prediction quality | Excellent |
| φ estimation | Converges within 100-200 obs |
| Variance formula | Correct AR(1) variance |

**Phase 2 difference:** When far from mean, MeanReversion receives epistemic bonus (deviation informs φ estimation).

---

### 3.4 AR(1) Near Unit Root (φ ≈ 0.99)

**Signal:** $y_t = 0.99 y_{t-1} + \epsilon_t$

**Domain examples:** Persistent economic indicators; slowly-decaying shocks.

**AEGIS behaviour:**

| Aspect | Expectation |
|--------|-------------|
| Dominant model at scale 1 | RandomWalk |
| Dominant model at scale 64 | MeanReversion |
| Prediction quality | Good |

**Multi-scale advantage:** At scale 1, mean-reversion is nearly invisible. At scale 64, the 64-step return shows clear mean-reversion (φ^64 ≈ 0.52), making structure detectable.

---

### 3.5 MA(1)

**Signal:** $y_t = \epsilon_t + \theta \epsilon_{t-1}$

**Domain examples:** Inventory adjustments; filtered sensor readings; shock propagation.

**AEGIS behaviour:**

| Aspect | Expectation |
|--------|-------------|
| Dominant model | MA1 |
| One-step prediction | Good |
| Multi-step prediction | Correctly predicts 0 |
| θ estimation | Converges within 100 obs |

---

### 3.6 ARMA(1,1)

**Signal:** $y_t = \phi y_{t-1} + \epsilon_t + \theta \epsilon_{t-1}$

**AEGIS behaviour:**

| Aspect | Expectation |
|--------|-------------|
| Dominant models | AR2 + MA1 (weighted) |
| Prediction quality | Good |
| Uncertainty | Good |

**Mechanism:** The model bank doesn't include ARMA directly, but combination of AR and MA models captures both components.

---

### 3.7 Ornstein-Uhlenbeck

**Signal:** $dy_t = \theta(\mu - y_t)dt + \sigma dW_t$

**Domain examples:** Interest rates (Vasicek); particle in potential; temperature reversion.

**AEGIS behaviour:**

| Aspect | Expectation |
|--------|-------------|
| Dominant model | MeanReversion |
| Prediction quality | Excellent |
| Parameter estimation | Accurate for θ, μ |

**Note:** Discretised O-U is exactly AR(1) toward mean, which MeanReversion represents directly.

---

## 4. Composite Signals

### 4.1 Trend + Noise

**Signal:** $y_t = at + \epsilon_t$

**AEGIS behaviour:**

| Aspect | Expectation |
|--------|-------------|
| Dominant model | LocalTrend |
| Prediction quality | Excellent |
| Trend/noise separation | Implicit via smoothing |

---

### 4.2 Sine + Noise

**Signal:** $y_t = A\sin(\omega t) + \epsilon_t$

**AEGIS behaviour:**

| Aspect | Expectation |
|--------|-------------|
| Dominant model | OscillatorBank |
| Prediction quality | Good |
| Amplitude estimation | Accurate |
| Variance estimation | Captures noise level |

---

### 4.3 Trend + Seasonality + Noise

**Signal:** $y_t = at + S_t + \epsilon_t$ (classical decomposition)

**Domain examples:** Retail sales; tourism; agricultural production.

**AEGIS behaviour:**

| Aspect | Expectation |
|--------|-------------|
| Dominant models | LocalTrend + SeasonalDummy |
| Prediction quality | Good to Excellent |
| Multi-scale benefit | Trend at long scales, season at matching scale |

**Sharp seasonality:** SeasonalDummy outperforms OscillatorBank for non-sinusoidal patterns.

---

### 4.4 Mean-Reversion + Oscillation

**Signal:** $y_t = \phi y_{t-1} + A\sin(\omega t) + \epsilon_t$

**AEGIS behaviour:**

| Aspect | Expectation |
|--------|-------------|
| Dominant models | MeanReversion + OscillatorBank |
| Prediction quality | Good |
| Weight distribution | Split between groups |

---

## 5. Non-Stationary and Regime-Changing

### 5.1 Random Walk with Drift

**Signal:** $y_t = \mu + y_{t-1} + \epsilon_t$

**AEGIS behaviour:**

| Aspect | Expectation |
|--------|-------------|
| Dominant model | LocalTrend |
| Drift estimation | Accurate via slope |
| Prediction quality | Good |

---

### 5.2 Variance Switching (Two-State)

**Signal:** $y_t \sim \mathcal{N}(0, \sigma^2_{S_t})$ where $S_t$ alternates

**Domain examples:** Market volatility regimes; turbulent/laminar flow.

**AEGIS behaviour:**

| Aspect | Expectation |
|--------|-------------|
| Point prediction | Optimal (zero mean) |
| Variance tracking | VolatilityTracker adapts |
| Interval calibration | QuantileTracker helps |
| Adaptation speed | 10-30 observations |

**Phase 2 difference:** Minimal for variance-only switching (level prediction unaffected).

---

### 5.3 Mean Switching (Two-State)

**Signal:** $y_t = \mu_{S_t} + \epsilon_t$ where $S_t \in \{A, B\}$

**Domain examples:** Economic expansion/recession; equipment states.

**AEGIS behaviour:**

| Aspect | Expectation |
|--------|-------------|
| During regime | Excellent |
| At transition | Poor initially |
| Recovery time (Phase 1) | 30-70 observations |
| Recovery time (Phase 2) | 20-50 observations |
| Break detection | Triggers if shift large |

**Phase 2 advantage:** Epistemic bonus accelerates adaptation. Models with high parameter uncertainty receive elevated weight during transition.

---

### 5.4 Threshold Autoregression

**Signal:** Different φ above and below threshold τ

**Domain examples:** Asset support/resistance; homeostatic bounds; business cycles.

**AEGIS behaviour:**

| Aspect | Expectation |
|--------|-------------|
| Dominant model | ThresholdAR |
| Regime detection | Accurate after τ learned |
| τ adaptation | Converges to empirical median |
| Prediction quality | Good |

**Phase 2 difference:** High epistemic value near threshold (observations very informative about τ and regime-specific φ).

---

### 5.5 Structural Break

**Signal:** Parameter change at unknown time T*

**AEGIS behaviour:**

| Aspect | Expectation |
|--------|-------------|
| Detection | CUSUM triggers on large breaks |
| Recovery (Phase 1) | 50-100 observations |
| Recovery (Phase 2) | 30-70 observations |
| Uncertainty | Correctly elevated during transition |

**Break detection tuning:** `break_threshold` parameter controls sensitivity. Lower values detect smaller breaks but risk false positives.

---

### 5.6 Gradual Drift

**Signal:** Parameters change slowly over time

**Domain examples:** Sensor degradation; market microstructure evolution.

**AEGIS behaviour:**

| Aspect | Expectation |
|--------|-------------|
| Tracking | Good via exponential forgetting |
| Break detection | May not trigger |
| Prediction quality | Good if drift slow |

**Key parameter:** `likelihood_forget` controls adaptation speed. Lower values (e.g., 0.95) track drift better but increase variance.

---

## 6. Heavy-Tailed Signals

### 6.1 Student-t Innovations

**Signal:** $y_t = \phi y_{t-1} + \epsilon_t$ where $\epsilon_t \sim t_\nu$

**Domain examples:** Financial returns (ν ≈ 3-5); insurance claims.

**AEGIS behaviour:**

| Aspect | ν > 4 | ν = 3-4 | ν < 3 |
|--------|-------|---------|-------|
| Point prediction | Good | Good | Moderate |
| Gaussian intervals | Undercover | Undercover badly | Fail |
| Quantile calibration | Corrects | Corrects | Partial |

**QuantileTracker benefit:** Adjusts interval widths based on empirical coverage, providing calibrated intervals even for non-Gaussian errors.

---

### 6.2 Occasional Jumps

**Signal:** $y_t = y_{t-1} + \epsilon_t + J_t$ where $J_t$ is rare large move

**Domain examples:** Currency interventions; flash crashes; supply shocks.

**AEGIS behaviour:**

| Aspect | Expectation |
|--------|-------------|
| Dominant model | JumpDiffusion |
| Jump detection | Classifies by size threshold |
| Variance estimate | Includes jump risk |
| λ estimation | Tracks jump frequency |

**Key advantage:** JumpDiffusion provides elevated variance during quiet periods, reflecting jump risk. Standard models underestimate tail risk.

**Phase 2 difference:** After jumps, epistemic value spikes (informative about λ), potentially accelerating return to normal weighting.

---

### 6.3 Power-Law Tails

**Signal:** Innovations with Pareto-like tails

**Domain examples:** Extreme weather; large insurance losses; viral content.

**AEGIS behaviour:**

| Aspect | α > 2 | α ≤ 2 |
|--------|-------|-------|
| Point prediction | Reasonable | Moderate |
| Variance estimate | Finite, trackable | Infinite, problematic |
| Interval calibration | QuantileTracker helps | Partial |

**Limitation:** For α ≤ 2 (infinite variance), no fixed-width interval achieves reliable coverage. QuantileTracker provides best-effort calibration.

---

## 7. Multi-Scale Structure

### 7.1 Fractional Brownian Motion

**Signal:** Long-memory process with Hurst exponent H ≠ 0.5

**AEGIS behaviour:**

| Aspect | H > 0.5 (persistent) | H < 0.5 (antipersistent) |
|--------|---------------------|-------------------------|
| Scale-1 models | Near random walk | Near random walk |
| Long-scale models | Detect persistence | Detect mean-reversion |
| Overall quality | Good | Good |

**Multi-scale advantage:** Different scales reveal H through autocorrelation patterns.

---

### 7.2 Multi-Timescale Mean-Reversion

**Signal:** $y_t = z^{(fast)}_t + z^{(slow)}_t$ with different reversion rates

**Domain examples:** Interest rates (fast mean-reversion to local, slow to global); asset prices.

**AEGIS behaviour:**

| Aspect | Expectation |
|--------|-------------|
| Short scales | Capture fast component |
| Long scales | Capture slow component |
| Prediction quality | Good |
| HierarchicalMR model | Naturally suited |

---

### 7.3 Trend + Momentum + Reversion

**Signal:** Short-term momentum, medium-term trend, long-term reversion

**Domain examples:** Asset prices; economic indicators.

**AEGIS behaviour:**

| Scale | Dominant Model | Structure |
|-------|----------------|-----------|
| 1-4 | AR2/MeanReversion | Momentum |
| 8-32 | LocalTrend/DampedTrend | Trend |
| 64+ | MeanReversion | Long-term reversion |

**Multi-scale architecture advantage:** Each timescale handled by appropriate models.

---

### 7.4 GARCH-like Volatility

**Signal:** $y_t = \sigma_t \epsilon_t$ with clustered volatility

**AEGIS behaviour:**

| Aspect | Expectation |
|--------|-------------|
| Level prediction | Random walk dominates |
| Volatility tracking | VolatilityTracker captures clustering |
| Interval calibration | Good |
| Leverage effects | LevelDependentVol if enabled |

---

## 8. Multiple Correlated Series

### 8.1 Perfectly Correlated

**Signal:** $y^{(1)}_t = y^{(2)}_t = x_t$

**AEGIS behaviour:**

| Aspect | Expectation |
|--------|-------------|
| Factor extraction | Identifies common factor |
| Per-stream prediction | Identical |
| Cross-stream regression | β ≈ 1 |

---

### 8.2 Contemporaneous Relationship

**Signal:** $y^{(2)}_t = \beta y^{(1)}_t + \epsilon_t$

**Domain examples:** Bid-ask prices; related sensors; simultaneous quotes.

**AEGIS behaviour (with `include_lag_zero=True`):**

| Aspect | Expectation |
|--------|-------------|
| β estimation | Accurate via lag-0 regression |
| Stream 2 prediction | Improved when stream 1 observed first |
| Prediction quality | Good |

**Requirement:** Stream 1 must be observed before stream 2 within each period.

---

### 8.3 Lead-Lag Relationship

**Signal:** $y^{(2)}_t = y^{(1)}_{t-k} + \epsilon_t$

**Domain examples:** Information propagation; supply chain effects.

**AEGIS behaviour:**

| Aspect | Expectation |
|--------|-------------|
| Lag detection | Automatic via cross-stream regression |
| β_k estimation | Accurate if k ≤ cross_stream_lags |
| Follower prediction | Substantially improved |

**Configuration:** `cross_stream_lags` must be ≥ k to capture the relationship.

---

### 8.4 Cointegration

**Signal:** Both I(1) but $y^{(1)}_t - \beta y^{(2)}_t \sim I(0)$

**Domain examples:** Pairs trading; purchasing power parity.

**AEGIS behaviour:**

| Aspect | Expectation |
|--------|-------------|
| Per-stream | RandomWalk dominates |
| Spread prediction | Cross-stream regression captures |
| Error correction | Implicit via residual regression |
| Prediction quality | Better than univariate |

---

### 8.5 Factor Structure

**Signal:** $\mathbf{y}_t = \Lambda \mathbf{f}_t + \epsilon_t$

**Domain examples:** Equity returns; macroeconomic indicators.

**AEGIS behaviour (>3 streams):**

| Aspect | Expectation |
|--------|-------------|
| Factor extraction | OnlinePCA identifies factors |
| Dimension reduction | Effective |
| Factor dynamics | Modelled separately |

---

## 9. Domain-Specific Applications

### 9.1 Financial Returns

**Characteristics:** Near-random walk; volatility clustering; fat tails; occasional jumps.

**AEGIS configuration:**
```python
config = AEGISConfig(
    use_quantile_calibration=True,
    volatility_decay=0.94,
    # Enable jump model
)
```

**Expected behaviour:**

| Aspect | Expectation |
|--------|-------------|
| Level prediction | RandomWalk dominates |
| Volatility | VolatilityTracker captures clustering |
| Jump risk | JumpDiffusion provides appropriate variance |
| Tail risk | QuantileTracker calibrates intervals |

---

### 9.2 Retail Sales

**Characteristics:** Strong day-of-week; secular trend; holiday spikes; promotions.

**AEGIS configuration:**
```python
config = AEGISConfig(
    seasonal_periods=[7, 365],
    # JumpDiffusion for promotions/holidays
)
```

**Expected behaviour:**

| Aspect | Expectation |
|--------|-------------|
| Weekly pattern | SeasonalDummy (period=7) |
| Trend | LocalTrend or DampedTrend |
| Holiday spikes | JumpDiffusion or elevated variance |
| Overall quality | Good to Excellent |

---

### 9.3 Temperature / Climate

**Characteristics:** Strong annual cycle; diurnal cycle; mean-reversion; occasional extremes.

**Expected behaviour:**

| Aspect | Expectation |
|--------|-------------|
| Annual cycle | OscillatorBank or SeasonalDummy |
| Diurnal cycle | SeasonalDummy (period=24) |
| Mean-reversion | MeanReversion for deviations |
| Extremes | QuantileTracker for calibration |

---

### 9.4 Web Traffic

**Characteristics:** Strong weekly pattern; trend; occasional viral spikes; heteroscedasticity.

**Expected behaviour:**

| Aspect | Expectation |
|--------|-------------|
| Weekly pattern | SeasonalDummy (period=7) |
| Trend | LocalTrend |
| Viral spikes | JumpDiffusion |
| Heteroscedasticity | LevelDependentVol if enabled |

---

### 9.5 Physiological Signals (ECG-like)

**Characteristics:** Quasi-periodic; variable rate; occasional arrhythmias.

**Expected behaviour:**

| Aspect | Expectation |
|--------|-------------|
| Regular rhythm | OscillatorBank or SeasonalDummy |
| Rate variability | Requires external rate input |
| Arrhythmias | JumpDiffusion, elevated uncertainty |

**Limitation:** AEGIS doesn't handle variable-rate periodicity natively.

---

### 9.6 Manufacturing / Quality Control

**Characteristics:** Stable mean; occasional drift; rare out-of-spec events.

**Expected behaviour:**

| Aspect | Expectation |
|--------|-------------|
| Stable operation | LocalLevel dominates |
| Drift detection | Break detector or LocalTrend |
| Out-of-spec | JumpDiffusion variance; QuantileTracker |

---

### 9.7 Energy Demand

**Characteristics:** Daily pattern; weekly pattern; weather-dependent; trend.

**AEGIS configuration:**
```python
config = AEGISConfig(
    seasonal_periods=[24, 168],  # Daily, weekly
    cross_stream_lags=24,
    include_lag_zero=True  # For weather contemporaneous effect
)
```

**Expected behaviour:**

| Aspect | Expectation |
|--------|-------------|
| Daily pattern | SeasonalDummy (period=24) |
| Weekly pattern | SeasonalDummy (period=168) or OscillatorBank |
| Weather effect | Cross-stream regression on weather stream |
| Prediction quality | Good to Excellent |

---

### 9.8 Insurance Claims

**Characteristics:** Count data; occasional large claims; level-dependent variance.

**AEGIS configuration:**
```python
config = AEGISConfig(
    use_level_dependent_vol=True,
    use_quantile_calibration=True
)
```

**Expected behaviour:**

| Aspect | Expectation |
|--------|-------------|
| Level prediction | Moderate (claims hard to predict) |
| Large claims | JumpDiffusion variance |
| Variance scaling | LevelDependentVol |
| Interval calibration | QuantileTracker essential |

---

## 10. Adversarial and Edge Cases

### 10.1 Impulse

**Signal:** $y_t = \delta_{t,T}$

**AEGIS behaviour:**

| Aspect | Expectation |
|--------|-------------|
| Spike handling | LocalLevel tracks then decays |
| Break detection | May trigger |
| Recovery | Returns to baseline |

---

### 10.2 Step Function

**Signal:** Piecewise constant with unknown jump times

**AEGIS behaviour:**

| Aspect | Expectation |
|--------|-------------|
| Within segments | Excellent (LocalLevel) |
| At jumps | Lag then recover |
| Variance | JumpDiffusion provides appropriate risk |

---

### 10.3 Contaminated Data

**Signal:** True process plus occasional erroneous observations

**AEGIS behaviour:**

| Aspect | Expectation |
|--------|-------------|
| Robustness | Moderate |
| Outlier handling | JumpDiffusion absorbs some |
| Interval calibration | QuantileTracker adapts |

**Recommendation:** Pre-filter obvious outliers for best results.

---

### 10.4 Missing Data

**Signal:** Irregular observation times

**AEGIS behaviour:** Not directly supported. Workarounds:
- Interpolate before feeding to AEGIS
- Use time-weighted updates (requires modification)

---

### 10.5 Very Short Series (n < 50)

**AEGIS behaviour:**

| Aspect | Expectation |
|--------|-------------|
| Model convergence | Incomplete |
| Weighting | Near uniform initially |
| Prediction quality | Moderate |
| Uncertainty | Appropriately wide |

**Phase 2 advantage:** Epistemic value provides exploration bonus, potentially improving early performance.

---

### 10.6 Very Long Series (n > 10,000)

**AEGIS behaviour:**

| Aspect | Expectation |
|--------|-------------|
| Parameter convergence | Complete |
| Weighting | Stable |
| Computational cost | Linear in n |
| Memory | Bounded (history capped) |

---

### 10.7 High-Frequency Data

**Signal:** Sub-second observations

**Considerations:**
- Microstructure noise dominates at very short timescales
- Consider aggregating before feeding to AEGIS
- Adjust `volatility_decay` for observation frequency

---

### 10.8 Chaotic Dynamics

**Signal:** Deterministic but unpredictable (Lorenz, etc.)

**AEGIS behaviour:**

| Aspect | Expectation |
|--------|-------------|
| Short horizon | May capture local dynamics |
| Long horizon | Fundamentally unpredictable |
| Dominant model | Likely RandomWalk |

**Limitation:** AEGIS cannot beat fundamental predictability limits of chaotic systems.

---

## 11. Summary Tables

### 11.1 Performance by Signal Type

| Signal | Point Prediction | Uncertainty | Dominant Model(s) |
|--------|------------------|-------------|-------------------|
| Constant | Excellent | Excellent | LocalLevel |
| Linear trend | Excellent | Excellent | LocalTrend |
| Sine wave | Excellent | Excellent | OscillatorBank |
| Sharp seasonal | Excellent | Excellent | SeasonalDummy |
| Random walk | Excellent | Excellent | RandomWalk |
| AR(1), φ=0.9 | Excellent | Excellent | MeanReversion |
| AR(1), φ=0.99 | Good | Good | Multi-scale MeanReversion |
| MA(1) | Good | Good | MA1 |
| Threshold AR | Good | Good | ThresholdAR |
| Asymmetric MR | Good | Good | AsymmetricMR |
| Jump-diffusion | Good | Good | JumpDiffusion |
| Mean switching | Lags→Good | Elevated→Good | Varies + Break detection |
| Variance switching | Good | Good | VolatilityTracker |
| Heavy tails (ν>3) | Good | Good | QuantileTracker |
| Heavy tails (ν≤3) | Moderate | Moderate | QuantileTracker |
| Cointegrated pair | Good | Good | Cross-stream regression |
| Lead-lag | Good | Good | Cross-stream regression |

### 11.2 Phase 1 vs Phase 2 Comparison

| Scenario | Phase 1 | Phase 2 | Difference |
|----------|---------|---------|------------|
| Stationary AR(1) | Excellent | Excellent | Minimal |
| Random walk | Excellent | Excellent | None |
| Regime change | Moderate→Good | Good faster | 30-50% faster adaptation |
| Near threshold | Good | Good+ | Elevated ThresholdAR weight |
| Jump event | Good | Good | Faster return to normal |
| Limited data | Moderate | Moderate+ | Exploration bonus |

### 11.3 Recommended Configuration by Domain

| Domain | Key Settings |
|--------|--------------|
| Financial | `use_quantile_calibration=True`, JumpDiffusion enabled |
| Retail | `seasonal_periods=[7, 365]`, SeasonalDummy |
| Energy | `seasonal_periods=[24, 168]`, cross-stream with weather |
| Manufacturing | `break_threshold=2.5` (sensitive detection) |
| Insurance | `use_level_dependent_vol=True`, QuantileTracker |
| General | Defaults work well for most cases |

### 11.4 Critical Success Factors

AEGIS performs well when:

1. **Signal structure matches model vocabulary** - The 15+ model types cover most common patterns
2. **Multi-scale decomposition exposes structure** - Slow dynamics become visible at long scales
3. **Cross-stream relationships are approximately linear** - Regression captures lead-lag and contemporaneous effects
4. **Tail behaviour is moderate** - ν > 2 or bounded jumps
5. **Regimes persist long enough** - At least 30-50 observations per regime
6. **Data quality is reasonable** - Limited missing data and outliers

### 11.5 Known Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| Novel dynamics outside vocabulary | Cannot capture | Extend model bank |
| Polynomial/exponential acceleration | Underestimates growth | DampedTrend for deceleration |
| Chaotic dynamics | Fundamentally limited | None |
| Very rapid regime switching | Lags behind | Lower `likelihood_forget` |
| Extreme heavy tails (α≤2) | Variance undefined | QuantileTracker helps |
| Nonlinear cross-stream | Linear approximation | Consider preprocessing |
| Variable-rate periodicity | Not supported | External rate signal |

---

*End of Appendix D*
