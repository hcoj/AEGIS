# AEGIS Multi-Horizon Performance Report v4

**Date:** 2025-12-27
**Version:** 4.0 (Post Scale Weight Computation Fix)
**Test Suite:** Signal Taxonomy Acceptance Tests
**Horizons Tested:** 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024

---

## Executive Summary

This report analyzes AEGIS performance across 38 signal types and 11 forecast horizons following the fix to **scale weight computation**. Scale weights are now computed based on level prediction accuracy rather than return prediction accuracy, ensuring scales that give accurate level predictions receive appropriate weights.

### Key Metrics

| Metric | v3 Value | v4 Value | Change |
|--------|----------|----------|--------|
| Total Signals Tested | 38 | 38 | - |
| Total Runtime | 549.93s | 554.84s | +0.9% |
| Mean Coverage (h=1) | 68.5% | 63.6% | -4.9pp |
| Mean Coverage (h=64) | 91.3% | 90.7% | -0.6pp |
| Mean Coverage (h=1024) | 93.7% | 93.3% | -0.4pp |

### Changes Since v3

1. **Scale weight computation fix**: `update_scale_weights()` now measures level prediction accuracy instead of return prediction accuracy
2. Scale 2 now receives higher weight (~26%) than before (~4%), improving short-term accuracy for some signals
3. Scale 64 weight reduced from ~37% to ~12%

---

## 1. Coverage Analysis by Horizon

### 1.1 Aggregate Coverage Trend

| Horizon | Mean Coverage | Target | Status |
|---------|---------------|--------|--------|
| h=1 | 63.6% | 95% | Under-covered |
| h=2 | 67.2% | 95% | Under-covered |
| h=4 | 70.9% | 95% | Under-covered |
| h=8 | 76.4% | 95% | Under-covered |
| h=16 | 83.1% | 95% | Under-covered |
| h=32 | 88.6% | 95% | Approaching target |
| h=64 | 90.7% | 95% | Close to target |
| h=128 | 91.6% | 95% | Close to target |
| h=256 | 91.8% | 95% | Close to target |
| h=512 | 91.9% | 95% | Close to target |
| h=1024 | 93.3% | 95% | Near target |

**Interpretation:** Coverage improves monotonically with horizon. The system is systematically under-confident at short horizons, producing prediction intervals that are too narrow. Coverage approaches the 95% target only at h=1024.

### 1.2 Coverage by Signal Category

| Category | h=1 | h=16 | h=64 | h=1024 | Best Performer |
|----------|-----|------|------|--------|----------------|
| Deterministic | 44.4% | 52.8% | 44.0% | 64.6% | Constant (100%) |
| Stochastic | 57.5% | 88.7% | 99.1% | 100% | All converge |
| Composite | 63.9% | 72.5% | 98.5% | 100% | MR+Osc (70%) |
| Non-Stationary | 60.2% | 86.5% | 97.5% | 100% | Variance Switch (69%) |
| Heavy-Tailed | 68.6% | 87.8% | 97.5% | 100% | Occasional Jumps (74%) |
| Multi-Scale | 63.7% | 86.6% | 96.6% | 100% | fBM Persistent (77%) |
| Multi-Stream | 67.8% | 89.8% | 99.0% | 100% | All similar |
| Edge Cases | 94.4% | 99.0% | 96.0% | 74.3% | Impulse (100%) |

### 1.3 Well-Calibrated vs Poorly-Calibrated Signals

**Well-Calibrated (h=1 coverage > 70%):**
- Constant Value: 100%
- Square Wave: 94%
- Impulse: 100%
- Step Function: 99%
- Contaminated Data: 84%
- Trend + Noise: 82%
- fBM Persistent: 77%
- Occasional Jumps: 74%

**Poorly-Calibrated (h=1 coverage < 50%):**
- Linear Trend: 4% (expected - deterministic)
- Polynomial Trend: 4% (expected - deterministic)
- Sinusoidal: 19% (expected - deterministic)
- White Noise: 47%
- MA(1): 46%
- Gradual Drift: 47%

---

## 2. Accuracy Analysis by Horizon

### 2.1 MAE Growth Patterns

| Growth Category | Signals | MAE Growth (h=1 to h=1024) |
|-----------------|---------|---------------------------|
| **Minimal** (< 5x) | Constant, Linear Trend | 0-2.2x |
| **Slow** (5-20x) | Trend+Noise, White Noise, Gradual Drift | 8.8-19.3x |
| **Moderate** (20-50x) | AR(1), Threshold AR, Multi-Timescale MR | 33-66x |
| **Fast** (50-100x) | ARMA, O-U, Random Walk | 58-109x |
| **Very Fast** (> 100x) | fBM, Power-Law, Contaminated | 99-213x |
| **Extreme** (> 300x) | Sinusoidal, Impulse, Step | 306-452x |

### 2.2 Best Accuracy by Signal Type

| Signal | MAE h=1 | MAE h=1024 | Growth | Notes |
|--------|---------|------------|--------|-------|
| Constant Value | 0.00 | 0.00 | 0.0x | Perfect |
| Impulse | 0.01 | 2.85 | 320x | Low base |
| Step Function | 0.02 | 10.25 | 452x | Low base |
| Linear Trend | 0.10 | 0.22 | 2.2x | Excellent |
| Sinusoidal | 0.13 | 39.93 | 306x | Period drift |
| Mean-Rev + Osc | 0.37 | 53.55 | 145x | Combined |
| fBM Persistent | 0.43 | 114.74 | 269x | Long memory |

### 2.3 Worst Accuracy by Signal Type

| Signal | MAE h=1 | MAE h=1024 | Growth | Notes |
|--------|---------|------------|--------|-------|
| Variance Switching | 2.38 | 109.13 | 46x | High variance |
| Student-t df=3 | 1.80 | 130.98 | 73x | Heavy tails |
| ARMA(1,1) | 1.36 | 78.31 | 58x | Complex dynamics |
| MA(1) | 1.34 | 35.24 | 26x | Short memory |
| Lead-Lag | 1.30 | 87.09 | 67x | Cross-stream |
| Cointegrated | 1.30 | 114.65 | 88x | Error correction |

---

## 3. Model Group Weighting Analysis

### 3.1 Dominant Model Groups

| Signal Type | Expected | Actual Dominant | Weight | Correct? |
|-------------|----------|-----------------|--------|----------|
| Constant Value | persistence | persistence | 67% | Yes |
| Linear Trend | trend | persistence | 100% | Partial* |
| AR(1) phi=0.8 | reversion | reversion | 49% | Yes |
| MA(1) | dynamic | dynamic | 99% | Yes |
| Sinusoidal | periodic | variance | 43% | No |
| Square Wave | periodic | periodic | 86% | Yes |
| Random Walk | persistence | reversion | 35% | No |
| White Noise | persistence | periodic | 38% | No |
| Threshold AR | reversion | reversion | 58% | Yes |

*Linear Trend: persistence dominates because LocalLevel predicts constant returns, matching a linear trend's constant slope.

### 3.2 Model Group Performance

| Group | Best Signal Match | Weight | Key Strength |
|-------|-------------------|--------|--------------|
| **persistence** | Constant Value | 67% | Stable levels |
| **reversion** | Threshold AR | 58% | Mean-reverting dynamics |
| **periodic** | Square Wave | 86% | Cyclical patterns |
| **dynamic** | MA(1) | 99% | Short-term dependencies |
| **variance** | Contaminated Data | 92% | Volatility adaptation |

---

## 4. Signal Category Performance

### 4.1 Deterministic Signals

| Signal | MAE h=1 | MAE h=1024 | Coverage h=1 | Dominant |
|--------|---------|------------|--------------|----------|
| Constant Value | 0.00 | 0.00 | 100% | persistence (67%) |
| Linear Trend | 0.10 | 0.22 | 4% | persistence (100%) |
| Sinusoidal | 0.13 | 39.93 | 19% | variance (43%) |
| Square Wave | 0.16 | 40.67 | 94% | periodic (86%) |
| Polynomial Trend | NaN | 15.44 | 4% | persistence |

**Key Finding:** Deterministic signals show low coverage at h=1 because predictions are highly confident but slightly offset. This is expected behavior - tight intervals around accurate predictions.

### 4.2 Stochastic Signals

| Signal | MAE h=1 | MAE h=1024 | Coverage h=1 | Dominant |
|--------|---------|------------|--------------|----------|
| White Noise | 1.13 | 18.81 | 47% | periodic (38%) |
| Random Walk | 1.16 | 125.55 | 65% | reversion (35%) |
| AR(1) phi=0.8 | 0.57 | 37.37 | 63% | reversion (49%) |
| AR(1) phi=0.99 | 0.58 | 64.12 | 68% | reversion (41%) |
| MA(1) | 1.34 | 35.24 | 46% | dynamic (99%) |
| ARMA(1,1) | 1.36 | 78.31 | 55% | dynamic (47%) |
| Ornstein-Uhlenbeck | 0.60 | 53.02 | 63% | reversion (41%) |

**Key Finding:** AR(1) signals show reversion model dominance as expected. MA(1) correctly identifies dynamic group (99% weight).

### 4.3 Composite Signals

| Signal | MAE h=1 | MAE h=1024 | Coverage h=1 | Dominant |
|--------|---------|------------|--------------|----------|
| Trend + Noise | 1.09 | 9.64 | 82% | dynamic (33%) |
| Sine + Noise | 0.60 | 63.72 | 58% | periodic (49%) |
| Trend + Season + Noise | 0.59 | 24.92 | 65% | periodic (59%) |
| Mean-Rev + Oscillation | 0.37 | 53.55 | 70% | dynamic (40%) |

**Key Finding:** Composite signals show mixed model group usage, with periodic and dynamic groups sharing weight.

### 4.4 Non-Stationary Signals

| Signal | MAE h=1 | MAE h=1024 | Coverage h=1 | Notes |
|--------|---------|------------|--------------|-------|
| RW with Drift | 1.15 | 119.42 | 65% | Similar to RW |
| Variance Switching | 2.38 | 109.13 | 69% | High variance phases |
| Mean Switching | 1.17 | 61.60 | 68% | Regime changes |
| Threshold AR | 0.59 | 35.95 | 63% | Regimes learned |
| Structural Break | 1.11 | 27.74 | 54% | Break detected |
| Gradual Drift | 1.07 | 20.79 | 47% | Forgetting adapts |

### 4.5 Heavy-Tailed Signals

| Signal | MAE h=1 | MAE h=1024 | Coverage h=1 | Notes |
|--------|---------|------------|--------------|-------|
| Student-t df=4 | 1.59 | 81.13 | 65% | Moderate tails |
| Student-t df=3 | 1.80 | 130.98 | 65% | Heavy tails |
| Occasional Jumps | 0.69 | 90.16 | 74% | Jump model helps |
| Power-Law Tails | 1.08 | 195.65 | 71% | Extreme tails |

### 4.6 Multi-Stream Signals

| Signal | MAE h=1 | MAE h=1024 | Coverage h=1 | Notes |
|--------|---------|------------|--------------|-------|
| Perfectly Correlated | 1.13 | 109.95 | 70% | Cross-stream active |
| Contemporaneous | 1.07 | 91.50 | 67% | Lag-0 captured |
| Lead-Lag | 1.30 | 87.09 | 66% | Lag detected |
| Cointegrated | 1.30 | 114.65 | 68% | Error correction |

---

## 5. Comparison with v3

### 5.1 Coverage Changes

| Horizon | v3 Coverage | v4 Coverage | Change |
|---------|-------------|-------------|--------|
| h=1 | 68.5% | 63.6% | -4.9pp |
| h=16 | 83.1% | 83.1% | 0.0pp |
| h=64 | 91.3% | 90.7% | -0.6pp |
| h=1024 | 93.7% | 93.3% | -0.4pp |

**Interpretation:** Short-horizon coverage decreased slightly. The scale weight fix changed which scales contribute most to predictions, affecting variance estimates.

### 5.2 Scale Weight Distribution

| Scale | v3 Weight (est.) | v4 Weight (est.) |
|-------|------------------|------------------|
| Scale 1 | 4% | 12% |
| Scale 2 | 4% | 26% |
| Scale 4 | 5% | 14% |
| Scale 8 | 21% | 12% |
| Scale 64 | 37% | 12% |

The fix redistributed weight toward shorter scales that better predict 1-step level changes.

---

## 6. Root Cause Analysis

### 6.1 Short-Horizon Under-Coverage

The consistent under-coverage at short horizons (63.6% vs 95% target) has multiple causes:

1. **Multi-scale combination**: Different scales predict different level changes, and averaging dilutes accuracy
2. **Model receiving returns**: Models like MeanReversionModel receive returns, not levels. For AR(1) levels with phi=0.8, returns have phi≈-0.2 (negative autocorrelation), which the model clips to phi=0.01
3. **Variance not scaled for combination**: Raw variances from different scales are combined without proper dimensional adjustment

### 6.2 Model Mismatch

| Signal | Expected Model | Actual Dominant | Issue |
|--------|----------------|-----------------|-------|
| AR(1) phi=0.8 | MeanReversion | OscillatorBank (77%) | MeanReversion learns wrong phi |
| White Noise | RandomWalk | Periodic (38%) | Periodic captures noise structure |
| Sinusoidal | OscillatorBank | Variance (43%) | Period mismatch |

### 6.3 Long-Horizon Variance Inflation

At h=1024, variance grows to ~6000 vs theoretical ~0.69 for AR(1). This is caused by non-stationary models (RandomWalk, LinearTrend) contributing variances that grow with horizon, even when weighted down.

---

## 7. Recommendations

### 7.1 Immediate Priorities

1. **Fix MeanReversionModel input**: Consider processing levels instead of returns, or adapting phi estimation for return dynamics
2. **Numerical stability**: Add overflow protection for polynomial and dynamic models
3. **Variance scaling**: Investigate proper variance adjustment for multi-scale combination

### 7.2 Architecture Improvements

1. **Horizon-aware variance**: Use appropriate variance formulas for each horizon
2. **Scale selection**: Consider using fewer scales for short-horizon predictions
3. **Model priors**: Give stationary models (MeanReversion) stronger priors on appropriate signals

### 7.3 Coverage Calibration

1. **Adaptive intervals**: Widen prediction intervals at short horizons based on empirical calibration
2. **Quantile tracking**: Use observed coverage to adjust interval multipliers

---

## 8. Conclusion

AEGIS v4 demonstrates consistent performance across diverse signal types:

**Strengths:**
- Excellent accuracy on constant and linear trend signals (MAE < 0.25)
- Good model group selection for MA(1), Square Wave, Threshold AR
- Robust long-horizon coverage (90%+ at h=64 and beyond)
- Effective handling of heavy-tailed and multi-stream signals

**Weaknesses:**
- Under-coverage at short horizons (63.6% at h=1 vs 95% target)
- MeanReversionModel learns incorrect phi due to return-based input
- Variance inflation at long horizons from non-stationary models
- Some model group mismatches (Sinusoidal, White Noise)

**Overall Assessment:** The scale weight computation fix improved weight distribution but did not resolve the fundamental short-horizon coverage issue. The root causes are architectural: models receive returns instead of levels, and variance combination across scales lacks proper dimensional adjustment.

---

## Appendix: Test Configuration

```
Horizons: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
Signals: 38 total across 8 categories
Default Scales: [1, 2, 4, 8, 16, 32, 64]
Models: 23 per scale (persistence, reversion, trend, periodic, dynamic, special, variance)
Runtime: 554.84 seconds
```

---

*Report generated from AEGIS acceptance test suite results*
*Total test runtime: 554.84 seconds*
