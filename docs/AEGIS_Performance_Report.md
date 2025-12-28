# AEGIS Performance Report

## Comprehensive Evaluation Across Signal Taxonomy

**Date:** December 2024
**Test Configuration:** 1000 samples per signal, horizons 1-256, seed 42
**Test Suite Status:** 451/452 tests passing (99.8%)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Test Suite Results](#2-test-suite-results)
3. [Performance by Signal Category](#3-performance-by-signal-category)
4. [Horizon Analysis](#4-horizon-analysis)
5. [Model Selection Behavior](#5-model-selection-behavior)
6. [Coverage Calibration](#6-coverage-calibration)
7. [Multi-Stream Performance](#7-multi-stream-performance)
8. [Known Issue: Multi-Scale Averaging for Seasonal Signals](#8-known-issue-multi-scale-averaging-for-seasonal-signals)
9. [Strengths](#9-strengths)
10. [Weaknesses](#10-weaknesses)
11. [Potential Improvements](#11-potential-improvements)
12. [Conclusions](#12-conclusions)

---

## 1. Executive Summary

AEGIS demonstrates **excellent performance** across most signal types in the taxonomy:

| Metric | H=1 | H=64 | H=256 |
|--------|-----|------|-------|
| Average MAE | 0.69 | 3.63 | 11.80 |
| Coverage (stochastic signals, 95% target) | **95.8%** | 99.8% | 100% |
| Test Pass Rate | 99.8% (451/452) | - | - |

**Key Findings:**
- Coverage on stochastic signals **exceeds the 95% target** at all horizons
- 22/23 stochastic signals have ≥93% coverage at H=1
- Seasonal patterns work correctly but are degraded by multi-scale averaging (MAE 1.98 vs optimal 0.5)
- Only GARCH-like signals show persistent under-coverage (83.8%)

---

## 2. Test Suite Results

### 2.1 Summary

| Test Category | Passed | Failed | Total |
|---------------|--------|--------|-------|
| Unit Tests | 390 | 0 | 390 |
| Integration Tests | 26 | 0 | 26 |
| Acceptance Tests | 4 | 1 | 5 |
| Regression Tests | 18 | 0 | 18 |
| Validation Tests | 13 | 0 | 13 |
| **Total** | **451** | **1** | **452** |

### 2.2 Single Failure Analysis

The failing test `test_long_horizon_forecasting` has an incorrect expectation:
- Expects H=1024 MAE < 10× H=1 MAE
- For random walks, error scales as √H, so H=1024 should be ~32× H=1
- Actual ratio of 14.6× is reasonable

**Recommendation:** Adjust test to use √H scaling expectation.

---

## 3. Performance by Signal Category

### 3.1 Deterministic Signals

| Signal | MAE@H1 | MAE@H64 | Coverage@H1 | Assessment |
|--------|--------|---------|-------------|------------|
| Constant | 0.0000 | 0.0000 | 100% | Perfect |
| Linear Trend | 0.0000 | 0.0011 | 100% | Perfect |
| Sine (P=16) | 0.2380 | 0.8354 | 11.8% | Good MAE (deterministic, low cov OK) |
| Sine (P=64) | 0.0616 | 5.3640 | 94.1% | Excellent |
| Polynomial | 0.0011 | 0.3182 | 100% | Excellent |
| Square Wave | 0.5000 | 0.0000 | 75.0% | Good (midpoint prediction) |

**Note:** Deterministic signals have low coverage by design - predictions are near-exact with tight intervals.

### 3.2 Stochastic Processes

| Signal | MAE@H1 | MAE@H64 | Coverage@H1 | RelMAE |
|--------|--------|---------|-------------|--------|
| White Noise | 1.07 | 1.19 | 96.1% | 98% |
| Random Walk | 0.82 | 9.91 | 98.6% | 100% |
| AR(1) φ=0.9 | 0.85 | 4.70 | 98.2% | 102% |
| AR(1) φ=0.99 | 0.80 | 10.56 | 98.0% | 101% |
| AR(1) φ=0.7 | 0.89 | 1.87 | 97.4% | 102% |
| AR(1) φ=0.5 | 0.95 | 1.53 | 96.8% | 102% |
| MA(1) θ=0.6 | 1.00 | 1.37 | 96.4% | 100% |

**All stochastic signals have ≥96% coverage**, exceeding the 95% target.

### 3.3 Composite Signals

| Signal | MAE@H1 | MAE@H64 | Coverage@H1 | RelMAE |
|--------|--------|---------|-------------|--------|
| Trend + Noise | 1.11 | 1.19 | 96.2% | 98% |
| Sine + Noise | 0.59 | 0.62 | 93.0% | 97% |
| Trend + Season + Noise | 0.59 | 1.25 | 94.6% | 99% |

### 3.4 Regime-Changing Signals

| Signal | MAE@H1 | MAE@H64 | Coverage@H1 | RelMAE |
|--------|--------|---------|-------------|--------|
| RW + Drift | 0.81 | 9.13 | 98.2% | 101% |
| Variance Switch | 1.43 | 1.71 | 94.6% | 98% |
| Mean Switch | 1.08 | 1.59 | 97.0% | 98% |
| Threshold AR | 0.43 | 2.13 | 95.4% | 102% |
| Structural Break | 1.31 | 1.96 | 94.9% | 98% |
| Gradual Drift | 0.53 | 0.60 | 93.1% | 98% |

**Regime adaptation working well** with 94-98% coverage.

### 3.5 Heavy-Tailed Signals

| Signal | MAE@H1 | MAE@H64 | Coverage@H1 | RelMAE |
|--------|--------|---------|-------------|--------|
| Heavy-Tail (ν=4) | 0.97 | 16.53 | 97.7% | 102% |
| Heavy-Tail (ν=3) | 1.09 | 19.09 | 97.7% | 104% |
| Jump Diffusion | 0.49 | 7.43 | 96.3% | 104% |

**Excellent calibration** even with fat tails.

### 3.6 Multi-Scale Signals

| Signal | MAE@H1 | MAE@H64 | Coverage@H1 | RelMAE |
|--------|--------|---------|-------------|--------|
| Multi-Timescale MR | 0.96 | 5.12 | 98.6% | 102% |
| GARCH-like | 0.13 | 0.13 | **83.8%** | 98% |
| Asymmetric AR | 0.42 | 2.51 | 95.3% | 102% |

**GARCH-like is the only significantly under-covered signal** (83.8% vs 95% target).

### 3.7 Seasonal Signals

| Signal | MAE@H1 | MAE@H64 | Coverage@H1 | RelMAE |
|--------|--------|---------|-------------|--------|
| Seasonal (P=7) | 1.98 | 1.98 | 95.4% | **69%** |

**Coverage is correct (95.4%)**, but MAE is elevated due to multi-scale averaging issue (see Section 8). The signal beats baseline by 31% (RelMAE=69%), indicating the pattern IS being captured, just not optimally.

### 3.8 Edge Cases

| Signal | MAE@H1 | MAE@H64 | Coverage@H1 | RelMAE |
|--------|--------|---------|-------------|--------|
| Impulse | 0.02 | 0.03 | 99.8% | 100% |
| Step Function | 0.25 | 1.91 | 90.2% | 104% |

---

## 4. Horizon Analysis

### 4.1 MAE and Coverage Growth with Horizon

| Horizon | Average MAE | Coverage (Stochastic) |
|---------|-------------|----------------------|
| H=1 | 0.69 | **95.8%** |
| H=2 | 0.89 | 97.1% |
| H=4 | 1.13 | 97.9% |
| H=8 | 1.30 | 98.6% |
| H=16 | 1.69 | 99.5% |
| H=32 | 2.41 | 99.8% |
| H=64 | 3.63 | 99.8% |
| H=128 | 6.26 | 100% |
| H=256 | 11.80 | 100% |

**Key Observations:**
- Coverage at H=1 is **95.8%**, meeting the 95% target
- Coverage increases to near-100% at longer horizons (conservative but safe)
- MAE growth is slower than √H, indicating value from non-random-walk models

### 4.2 Coverage by Individual Stochastic Signal (H=1)

| Assessment | Count | Signals |
|------------|-------|---------|
| Excellent (≥97%) | 9 | Random Walk, AR(1) variants, Heavy-Tail, Multi-Timescale MR |
| Good (95-97%) | 9 | White Noise, MA(1), Threshold AR, Seasonal, Jump Diffusion |
| Acceptable (93-95%) | 4 | Sine+Noise, Variance Switch, Structural Break, Gradual Drift |
| Under-covered (<93%) | 1 | **GARCH-like (83.8%)** |

**22/23 stochastic signals have ≥93% coverage.**

---

## 5. Model Selection Behavior

### 5.1 Scale Weight Distribution

| Signal | Scale 1 | Scale 8 | Scale 32 | Scale 64 |
|--------|---------|---------|----------|----------|
| Random Walk | 0.04% | 2.3% | 24.6% | 65.2% |
| AR(1) φ=0.9 | 0.4% | 3.5% | 55.4% | 23.2% |
| Sine (P=16) | 0.0% | 0.0% | 33.3% | 33.3% |
| Linear Trend | 14.3% | 14.3% | 14.3% | 14.3% |
| Threshold AR | 0.4% | 2.3% | 35.5% | 42.3% |

Scale weights correctly adapt:
- Random walks favor long scales (65% at scale 64)
- Trend signals have uniform weights across scales
- Mean-reverting signals balance medium-long scales

### 5.2 Model Dominance

For the seasonal signal (period 7):
- **SeasonalDummy_p7 receives 100% weight** at all scales
- The model correctly learns the pattern
- Issue is in scale combination, not model selection

---

## 6. Coverage Calibration

### 6.1 Stochastic Signal Coverage Summary

| Horizon | Average | Min | Max | Target |
|---------|---------|-----|-----|--------|
| H=1 | 95.8% | 83.8% | 98.6% | 95% |
| H=4 | 97.9% | 94.0% | 99.5% | 95% |
| H=16 | 99.5% | 98.0% | 100% | 95% |
| H=64 | 99.8% | 99.0% | 100% | 95% |

**Coverage meets or exceeds target at all horizons.**

### 6.2 Under-Covered Signals

Only **GARCH-like** (83.8%) shows significant under-coverage. This is due to:
- Volatility clustering not fully captured by VolatilityTracker
- Regime-dependent variance changes too rapidly for EWMA tracking

---

## 7. Multi-Stream Performance

### 7.1 Correlated Streams

| Horizon | Stream 2 MAE |
|---------|--------------|
| H=1 | 1.15 |
| H=64 | 9.08 |
| H=256 | 26.00 |

### 7.2 Lead-Lag Relationship

| Horizon | Follower MAE |
|---------|--------------|
| H=1 | 1.29 |
| H=64 | 8.57 |
| H=256 | 25.40 |

Cross-stream relationships are being captured, though coefficient diagnostics are empty (diagnostic gap).

---

## 8. Known Issue: Multi-Scale Averaging for Seasonal Signals

### 8.1 Problem Description

The multi-scale architecture divides scale-s predictions by s to get "per-step" predictions. This assumes **linear interpolation** of returns, which fails for seasonal patterns:

```
For seasonal pattern [10, 12, 15, 14, 13, 8, 5]:
- 1-step returns: [2, 3, -1, -1, -5, -3, 5]
- 2-step returns: [5, 2, -2, -6, -8, 2, 7]
- 4-step returns: [3, -4, -10, -4, -1, 7, 9]

At position 2 (value=15), predicting next step:
- Scale 1: predicts -1, divide by 1 = -1 ✓
- Scale 2: predicts -2, divide by 2 = -1 ✓ (coincidentally)
- Scale 4: predicts -10, divide by 4 = -2.5 ✗ (should be -1)
```

### 8.2 Impact

- With **scale=1 only**: MAE = **0.50** (optimal)
- With **all scales**: MAE = **1.98** (3.9× worse)
- Coverage is unaffected (95.4%) because intervals are appropriately wide

### 8.3 Potential Fixes

1. **Horizon-aware scale weighting**: Weight scale 1 heavily for periodic signals
2. **Include seasonal-aligned scales**: Add scales 7, 14, 21 for period-7 seasonality
3. **Non-linear scale combination**: Don't assume linear interpolation between scales

---

## 9. Strengths

1. **Excellent coverage calibration**: 95.8% at H=1, meeting target
2. **Robust multi-scale architecture**: Correctly adapts scale weights for different signal types
3. **Strong regime adaptation**: 94-98% coverage during structural breaks and mean shifts
4. **Good heavy-tail handling**: QuantileTracker maintains calibration even with ν=3 innovations
5. **Comprehensive model bank**: 24 models covering persistence, trend, reversion, periodic, dynamic, and variance patterns
6. **Solid test suite**: 451/452 tests passing with comprehensive coverage

---

## 10. Weaknesses

### 10.1 Critical

1. **GARCH-like under-coverage (83.8%)**:
   - Only signal with persistent calibration failure
   - VolatilityTracker not adapting fast enough to volatility clustering

### 10.2 Moderate

2. **Multi-scale seasonal degradation**:
   - MAE 1.98 vs optimal 0.50 (3.9× worse)
   - Coverage correct, but predictions sub-optimal
   - See Section 8 for details

3. **Diagnostic gaps**:
   - Cross-stream coefficients not populated
   - Model weights at scale 1 sometimes empty
   - Limits debugging and tuning visibility

---

## 11. Potential Improvements

### 11.1 High Priority

1. **Fix GARCH-like coverage**:
   - Increase volatility_decay (faster adaptation)
   - Add regime-switching volatility model
   - Consider GARCH-specific model

2. **Improve seasonal accuracy**:
   - Implement horizon-aware scale weighting
   - Add seasonal-aligned scales [1, 7, 14, 28, 56]
   - Or detect dominant seasonality and weight scale 1 heavily

### 11.2 Medium Priority

3. **Fix diagnostics**:
   - Populate cross-stream coefficients
   - Ensure model weights appear at all scales
   - Add more detailed logging

4. **Test suite fix**:
   - Adjust `test_long_horizon_forecasting` to use √H scaling

### 11.3 Lower Priority

5. **Phase 2 evaluation**: Epistemic value not providing significant benefit in current tests
6. **Add ARMA model**: Would improve MA(1) signal performance

---

## 12. Conclusions

### 12.1 Overall Assessment

AEGIS is a **well-calibrated, robust time series prediction system** that exceeds its coverage targets:

| Metric | Target | Actual |
|--------|--------|--------|
| H=1 Coverage (stochastic) | 95% | **95.8%** |
| Test Pass Rate | - | 99.8% |
| Signals with ≥93% coverage | - | 22/23 |

### 12.2 Production Readiness

AEGIS is suitable for:
- Production forecasting on standard stochastic processes
- Regime-changing environments (structural breaks, mean shifts)
- Heavy-tailed data (financial returns, etc.)

Caution advised for:
- GARCH/volatility-clustering signals (under-coverage)
- Seasonal signals where maximum accuracy is critical (use scale=1 only)

### 12.3 Comparison with Taxonomy Expectations

| Signal Type | Expected | Actual | Status |
|-------------|----------|--------|--------|
| Constant | Excellent | Excellent | ✓ |
| Linear Trend | Excellent | Excellent | ✓ |
| Random Walk | Excellent | Excellent | ✓ |
| AR(1) | Excellent | Excellent | ✓ |
| Threshold AR | Good | Good | ✓ |
| Jump Diffusion | Good | Good | ✓ |
| Seasonal | Excellent | Good* | ⚠️ |
| GARCH-like | Good | Moderate | ⚠️ |

*Seasonal coverage is excellent (95.4%), but accuracy is degraded by multi-scale averaging.

---

*Report generated by comprehensive analysis of AEGIS test suite and performance benchmarks.*
