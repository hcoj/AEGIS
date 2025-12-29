# AEGIS Comprehensive Performance Report

**Evaluation Date:** December 29, 2025
**Training Observations:** 10,000
**Test Observations:** 200
**Horizons Evaluated:** 1, 4, 16, 64, 256, 1024
**Total Signals Tested:** 34 (across 7 categories)
**Test Suite Status:** All 448 tests passing

---

## Executive Summary

AEGIS (Active Epistemic Generative Inference System) was evaluated against 34 signal types from the signal taxonomy, covering deterministic, stochastic, composite, non-stationary, heavy-tailed, multi-scale, and adversarial patterns.

### Key Findings

| Metric | Value | Assessment |
|--------|-------|------------|
| **Stochastic signals beating naive (h=64)** | 6/26 (23%) | Needs improvement |
| **Mean MASE @ h=64 (stochastic)** | 1.587 | Slightly worse than naive |
| **Mean MASE @ h=1024 (stochastic)** | 6.283 | Significant degradation |
| **Coverage @ 95% target (h=64)** | Achieved for most signals | Good |
| **Numerical stability** | 4 signals with overflow | Critical issue |

### Overall Assessment

AEGIS demonstrates **strong uncertainty calibration** with coverage generally meeting the 95% target at longer horizons. However, **point prediction accuracy** is below expectations for many signal types, particularly at short horizons (h=1) where MASE often exceeds 1.0.

**Critical issues identified:**
1. Numerical overflow in dynamic models for trending signals
2. Oscillator model instability at very long horizons
3. Mean-reversion models not dominating for AR(1) signals
4. Consistent underperformance at h=1 across many signal types

---

## Detailed Results by Category

### 1. Deterministic Signals

| Signal | h=1 MAE | h=64 MAE | h=1024 MAE | Assessment |
|--------|---------|----------|------------|------------|
| Constant | 0.000 | 0.000 | 0.000 | Excellent |
| Linear trend | NaN | NaN | NaN | **CRITICAL: Overflow** |
| Sinusoidal | 0.625 | 0.629 | 0.629 | MAE good, MASE broken |
| Square wave | 3.571 | 34.062 | 585.287 | Poor |
| Polynomial trend | NaN | NaN | NaN | **CRITICAL: Overflow** |

**Observations:**
- Constant signal handled perfectly with LevelAwareMeanReversionModel dominating
- Linear and polynomial trends cause numerical overflow in AR2Model and related dynamic models
- Pure sinusoidal signal: OscillatorBank correctly dominates but variance estimation fails at long horizons
- Square wave: VolatilityTracker dominates instead of SeasonalDummy (period mismatch with default config)

### 2. Simple Stochastic Signals

| Signal | h=1 MASE | h=64 MASE | h=1024 MASE | Beats Naive? |
|--------|----------|-----------|-------------|--------------|
| White noise | 0.758 | 0.972 | 0.920 | Yes (all h) |
| Random walk | 1.485 | 1.116 | 1.642 | No |
| AR(1) phi=0.9 | 1.453 | 1.056 | 1.335 | No |
| AR(1) phi=0.7 | 1.388 | 1.058 | 1.420 | No |
| AR(1) phi=0.99 | 1.475 | 1.236 | 1.264 | No |
| MA(1) | 1.395 | 1.001 | 1.067 | Borderline |
| ARMA(1,1) | 1.549 | 1.058 | 1.319 | No |
| Ornstein-Uhlenbeck | 1.454 | 1.058 | 1.333 | No |

**Observations:**
- White noise is the only simple stochastic signal where AEGIS consistently beats naive
- AR(1) signals: MA1Model at scale 2 dominates instead of MeanReversionModel
- All signals show worse MASE at h=1 than at h=4 or h=64 (short-term prediction weakness)
- Model selection appears suboptimal - ThresholdARModel often ranks high for AR(1) signals

### 3. Composite Signals

| Signal | h=1 MASE | h=64 MASE | h=1024 MASE | Beats Naive? |
|--------|----------|-----------|-------------|--------------|
| Trend + noise | NaN | NaN | NaN | **Overflow** |
| Sine + noise | 1.075 | 1.000 | 0.975 | Yes (h=64+) |
| Trend + seasonality + noise | NaN | NaN | NaN | **Overflow** |
| Reversion + oscillation | 1.078 | 1.438 | 1.423 | h=4, h=16 only |

**Observations:**
- Trend-containing signals cause numerical overflow (critical bug)
- Sine + noise: OscillatorBank correctly dominates and achieves good performance
- Reversion + oscillation: Excellent at h=4 (MASE=0.34) and h=16 (MASE=0.16), degrades at longer horizons

### 4. Non-Stationary Signals

| Signal | h=1 MASE | h=64 MASE | h=1024 MASE | Beats Naive? |
|--------|----------|-----------|-------------|--------------|
| Random walk with drift | 1.479 | 1.314 | 0.983 | h=1024 only |
| Variance switching | 0.781 | 0.974 | 0.931 | Yes (all h) |
| Mean switching | 0.913 | 1.366 | 3.481 | h=1 only |
| Threshold AR | 1.399 | 1.052 | 1.345 | No |
| Structural break | 0.792 | 0.973 | 0.918 | Yes (all h) |
| Gradual drift | 0.757 | 0.973 | 0.859 | Yes (all h) |

**Observations:**
- Variance switching, structural break, and gradual drift handled well
- Mean switching: Good at h=1 but degrades significantly at longer horizons
- Threshold AR: ThresholdARModel correctly ranks high but doesn't dominate

### 5. Heavy-Tailed Signals

| Signal | h=1 MASE | h=64 MASE | h=1024 MASE | Beats Naive? |
|--------|----------|-----------|-------------|--------------|
| Student-t (df=4) | 0.872 | 1.308 | 11.189 | h=1 only |
| Student-t (df=3) | 1.133 | 3.542 | 82.185 | No |
| Jump diffusion | 1.514 | 1.219 | 14.844 | No |
| Power-law tails | 1.581 | 9.415 | 175.671 | No |

**Observations:**
- Heavy-tailed signals show severe degradation at long horizons
- JumpDiffusionModel correctly activates for jump diffusion signal
- Power-law tails: ChangePointModel dominates (appropriate for extreme values)
- Coverage still reasonably calibrated despite poor point predictions

### 6. Multi-Scale Signals

| Signal | h=1 MASE | h=64 MASE | h=1024 MASE | Beats Naive? |
|--------|----------|-----------|-------------|--------------|
| Fractional Brownian (H=0.7) | 1.604 | 1.194 | 3.801 | No |
| Multi-timescale reversion | 1.377 | 1.036 | 1.015 | Borderline |
| Trend + momentum + reversion | 1.339 | 1.023 | 0.945 | h=1024 |
| GARCH-like volatility | 0.768 | 0.975 | 0.919 | Yes (all h) |

**Observations:**
- GARCH-like signal handled well (MA1Model dominates for level, volatility tracked)
- Multi-scale signals show AEGIS's multi-scale architecture providing some benefit
- Performance improves at longer horizons for trend+momentum+reversion

### 7. Adversarial/Edge Cases

| Signal | h=1 MASE | h=64 MASE | h=1024 MASE | Assessment |
|--------|----------|-----------|-------------|------------|
| Impulse | inf | inf | inf | Perfect MAE, MASE undefined |
| Step function | 2.697 | 1.601 | 3.718 | Poor |
| Contaminated data | 1.728 | 2.306 | 17.247 | Poor |

**Observations:**
- Impulse: Handled correctly (MAE ~0) but MASE undefined due to zero naive error
- Step function: ChangePointModel and JumpDiffusionModel activate appropriately
- Contaminated data: Poor coverage (85-89%) despite robust estimation being enabled

---

## Coverage Analysis

### Coverage by Horizon (Stochastic Signals)

| Horizon | Mean Coverage | Target (95%) | Assessment |
|---------|---------------|--------------|------------|
| h=1 | 88.8% | 95% | Under-covered |
| h=4 | 95.8% | 95% | On target |
| h=16 | 99.0% | 95% | Over-covered |
| h=64 | 97.8% | 95% | Slightly over |
| h=256 | 97.7% | 95% | Slightly over |
| h=1024 | 98.6% | 95% | Over-covered |

**Observations:**
- Short-term (h=1) intervals are too narrow
- Medium to long-term intervals are appropriately calibrated or slightly conservative
- Quantile calibration appears effective at h=4+

---

## Model Selection Analysis

### Dominant Models by Signal Category

| Category | Expected Winner | Actual Winner | Correct? |
|----------|-----------------|---------------|----------|
| Constant | LocalLevel | LevelAwareMeanReversion | Acceptable |
| AR(1) | MeanReversion | MA1Model | **Incorrect** |
| MA(1) | MA1Model | MA1Model | Correct |
| Sinusoidal | OscillatorBank | OscillatorBank | Correct |
| Jump diffusion | JumpDiffusion | MA1Model + JumpDiffusion | Correct |
| GARCH | VolatilityTracker | MA1Model | Acceptable |
| Threshold AR | ThresholdAR | MA1Model + ThresholdAR | Partial |

**Key Finding:** MA1Model dominates across many signal types where it shouldn't be optimal. This suggests the model scoring/weighting mechanism may be biased toward MA1Model's likelihood function.

---

## Strengths

1. **Excellent uncertainty calibration** at horizons h>=4
2. **Good handling of variance changes** (variance switching, GARCH-like)
3. **Appropriate model activation** for periodic signals (OscillatorBank)
4. **Robust to gradual drift** and structural breaks
5. **Multi-scale architecture** provides benefit for complex signals
6. **White noise** and similar mean-zero signals handled optimally

## Weaknesses

1. **Numerical instability** for trending signals (critical bug in dynamic models)
2. **MA1Model bias** - dominates even when suboptimal
3. **Poor h=1 predictions** - consistently worse than h=4 or h=64
4. **MeanReversion models underweighted** for AR(1) signals
5. **Heavy-tailed signal degradation** at long horizons
6. **Square wave/step function** poor performance (period mismatch)
7. **Short-term coverage** below 95% target

---

## Recommended Improvements

### Critical Priority (Bugs)

1. **Fix numerical overflow in dynamic models**
   - Location: `src/aegis/models/dynamic.py:94-132`
   - Issue: AR2Model theta/phi parameters overflow for trending signals
   - Fix: Add bounds checking and gradient clipping

2. **Fix oscillator variance at very long horizons**
   - Location: `src/aegis/models/periodic.py`
   - Issue: Variance collapses to near-zero for pure sinusoids at h>64
   - Fix: Ensure minimum variance floor is applied correctly

### High Priority (Performance)

3. **Investigate MA1Model dominance**
   - MA1Model consistently wins even for AR(1) signals
   - May indicate issue with MeanReversion model likelihood computation
   - Compare log-likelihood values between models on AR(1) data

4. **Improve h=1 predictions**
   - All signals show MASE(h=1) > MASE(h=4)
   - May indicate over-smoothing at short horizons
   - Consider scale-specific model weighting

5. **Better long-horizon heavy-tail handling**
   - MASE explodes at h=256+ for heavy-tailed signals
   - Consider capping variance growth or using robust estimators

### Medium Priority (Features)

6. **Add period detection for square waves**
   - SeasonalDummyModel with correct period should dominate
   - Current config defaults don't match test signal period (14)

7. **Improve short-term coverage**
   - h=1 coverage is 88.8% vs 95% target
   - Quantile tracker may need faster adaptation at h=1

8. **Consider model bank pruning**
   - Some models (e.g., SeasonalDummyModel at most scales) rarely win
   - Computational cost could be reduced

### Low Priority (Enhancements)

9. **Add horizon-specific model weighting**
   - Scale weights are based on h=1 accuracy only
   - Long-horizon predictions might benefit from different model selection

10. **Implement ensemble variance inflation**
    - Between-model variance could be inflated at long horizons
    - Would improve heavy-tail handling

---

## Signal-by-Signal Results

### Full Results Table (Stochastic Signals Only)

| Signal | h=1 | h=4 | h=16 | h=64 | h=256 | h=1024 |
|--------|-----|-----|------|------|-------|--------|
| white_noise | 0.76 | 0.94 | 0.98 | 0.97 | 0.99 | 0.92 |
| random_walk | 1.48 | 1.12 | 1.06 | 1.12 | 0.82 | 1.64 |
| ar1_phi09 | 1.45 | 1.09 | 1.05 | 1.06 | 1.21 | 1.33 |
| ar1_phi07 | 1.39 | 1.06 | 1.07 | 1.06 | 1.26 | 1.42 |
| ar1_near_unit | 1.47 | 1.12 | 1.11 | 1.24 | 1.77 | 1.26 |
| ma1 | 1.39 | 1.02 | 1.02 | 1.00 | 1.15 | 1.07 |
| arma11 | 1.55 | 1.06 | 1.07 | 1.06 | 1.23 | 1.32 |
| ornstein_uhlenbeck | 1.45 | 1.09 | 1.06 | 1.06 | 1.22 | 1.33 |
| sine_plus_noise | 1.07 | 0.71 | 0.58 | 1.00 | 1.02 | 0.98 |
| reversion_oscillation | 1.08 | 0.34 | 0.16 | 1.44 | 1.51 | 1.42 |
| random_walk_drift | 1.48 | 1.12 | 1.13 | 1.31 | 1.87 | 0.98 |
| variance_switching | 0.78 | 0.94 | 0.98 | 0.97 | 1.00 | 0.93 |
| mean_switching | 0.91 | 1.01 | 1.09 | 1.37 | 2.13 | 3.48 |
| threshold_ar | 1.40 | 1.07 | 1.07 | 1.05 | 1.18 | 1.34 |
| structural_break | 0.79 | 0.95 | 0.98 | 0.97 | 0.99 | 0.92 |
| gradual_drift | 0.76 | 0.94 | 0.98 | 0.97 | 0.97 | 0.86 |
| student_t_df4 | 0.87 | 0.96 | 1.04 | 1.31 | 2.98 | 11.19 |
| student_t_df3 | 1.13 | 1.21 | 1.72 | 3.54 | 15.97 | 82.19 |
| jump_diffusion | 1.51 | 1.23 | 1.54 | 1.22 | 2.24 | 14.84 |
| power_law | 1.58 | 1.93 | 3.36 | 9.41 | 41.10 | 175.67 |
| fractional_brownian | 1.60 | 1.19 | 1.12 | 1.19 | 1.33 | 3.80 |
| multi_timescale_reversion | 1.38 | 1.13 | 1.11 | 1.04 | 1.04 | 1.01 |
| trend_momentum_reversion | 1.34 | 1.14 | 1.05 | 1.02 | 1.14 | 0.95 |
| garch_like | 0.77 | 0.94 | 0.98 | 0.98 | 0.99 | 0.92 |
| step_function | 2.70 | 1.74 | 1.49 | 1.60 | 2.31 | 3.72 |
| contaminated | 1.73 | 1.65 | 1.79 | 2.31 | 3.07 | 17.25 |

*Values are MASE (Mean Absolute Scaled Error). MASE < 1.0 beats naive baseline.*

---

## Conclusion

AEGIS shows promise in uncertainty quantification but requires significant improvements in point prediction accuracy. The numerical stability issues with trending signals are critical bugs that must be addressed. The MA1Model dominance across signal types suggests the model selection mechanism needs investigation.

**Recommended next steps:**
1. Fix numerical overflow bugs (critical)
2. Investigate and fix MA1Model bias
3. Improve h=1 prediction accuracy
4. Re-evaluate after fixes with same test suite

---

*Report generated by comprehensive_evaluation.py using 10,000 training observations per signal.*
