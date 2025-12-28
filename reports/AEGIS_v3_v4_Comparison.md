# AEGIS Performance Comparison: v3 vs v4

**Date:** 2025-12-27
**Purpose:** Comprehensive comparison of AEGIS performance before and after scale weight computation fix

---

## Executive Summary

| Metric | v3 | v4 | Delta | Assessment |
|--------|----|----|-------|------------|
| Mean Coverage h=1 | 68.5% | 63.6% | **-4.9pp** | Regression |
| Mean Coverage h=64 | 91.3% | 90.7% | -0.6pp | Stable |
| Mean Coverage h=1024 | 93.7% | 93.3% | -0.4pp | Stable |
| Runtime | 549.93s | 554.84s | +0.9% | Stable |

**Bottom Line:** The scale weight fix did not improve short-horizon coverage as hoped. Coverage at h=1 actually decreased by 4.9 percentage points. The fix correctly redistributed scale weights toward shorter scales, but this exposed that the underlying models produce insufficiently calibrated variances.

---

## 1. Coverage Comparison by Horizon

### 1.1 Aggregate Coverage

| Horizon | v3 | v4 | Change | Winner |
|---------|----|----|--------|--------|
| h=1 | 68.5% | 63.6% | -4.9pp | v3 |
| h=2 | - | 67.2% | - | - |
| h=4 | - | 70.9% | - | - |
| h=8 | - | 76.4% | - | - |
| h=16 | 83.1% | 83.1% | 0.0pp | Tie |
| h=32 | - | 88.6% | - | - |
| h=64 | 91.3% | 90.7% | -0.6pp | v3 |
| h=128 | - | 91.6% | - | - |
| h=256 | - | 91.8% | - | - |
| h=512 | - | 91.9% | - | - |
| h=1024 | 93.7% | 93.3% | -0.4pp | v3 |

### 1.2 Coverage by Signal Type (h=1)

| Signal | v3 | v4 | Change | Notes |
|--------|----|----|--------|-------|
| Constant Value | 100% | 100% | 0pp | Perfect both |
| Random Walk | 76% | 65% | **-11pp** | Significant regression |
| AR(1) phi=0.8 | 66% | 63% | -3pp | Slight regression |
| AR(1) phi=0.99 | 80% | 68% | **-12pp** | Significant regression |
| Trend + Noise | 81% | 82% | +1pp | Stable |
| Mean-Rev + Osc | 84% | 70% | **-14pp** | Significant regression |
| Student-t (df=4) | 66% | 65% | -1pp | Stable |
| Occasional Jumps | 81% | 74% | -7pp | Moderate regression |
| White Noise | 49% | 47% | -2pp | Stable |
| MA(1) | 52% | 46% | -6pp | Moderate regression |

**Biggest Regressions:**
1. Mean-Rev + Oscillation: -14pp
2. AR(1) phi=0.99: -12pp
3. Random Walk: -11pp

**Improvements:**
- Trend + Noise: +1pp (only signal that improved)

---

## 2. Accuracy Comparison (MAE)

### 2.1 MAE at h=1

| Signal | v3 | v4 | Change | Winner |
|--------|----|----|--------|--------|
| Constant Value | 0.00 | 0.00 | 0.00 | Tie |
| Linear Trend | 0.10 | 0.10 | 0.00 | Tie |
| Sinusoidal | 0.23 | 0.13 | **-0.10** | v4 |
| Square Wave | 0.12 | 0.16 | +0.04 | v3 |
| White Noise | 1.13 | 1.13 | 0.00 | Tie |
| Random Walk | 1.14 | 1.16 | +0.02 | v3 |
| AR(1) phi=0.8 | 0.57 | 0.57 | 0.00 | Tie |
| MA(1) | 1.33 | 1.34 | +0.01 | v3 |
| Trend + Noise | 1.09 | 1.09 | 0.00 | Tie |

### 2.2 MAE at h=1024

| Signal | v3 | v4 | Change | Winner |
|--------|----|----|--------|--------|
| Constant Value | 0.00 | 0.00 | 0.00 | Tie |
| Linear Trend | 0.18 | 0.22 | +0.04 | v3 |
| Sinusoidal | 29.59 | 39.93 | **+10.34** | v3 |
| Random Walk | 110.37 | 125.55 | **+15.18** | v3 |
| AR(1) phi=0.8 | 28.30 | 37.37 | **+9.07** | v3 |
| MA(1) | 22.01 | 35.24 | **+13.23** | v3 |
| Trend + Noise | 13.29 | 9.64 | **-3.65** | v4 |
| O-U Process | 33.29 | 53.02 | **+19.73** | v3 |

**Biggest MAE Regressions at h=1024:**
1. O-U Process: +19.73
2. Random Walk: +15.18
3. MA(1): +13.23
4. Sinusoidal: +10.34

**MAE Improvements:**
- Trend + Noise: -3.65 at h=1024 (v4 better)
- Sinusoidal: -0.10 at h=1 (v4 better)

---

## 3. Model Group Weighting Comparison

### 3.1 Dominant Group Changes

| Signal | v3 Dominant | v4 Dominant | Weight v3 | Weight v4 | Change |
|--------|-------------|-------------|-----------|-----------|--------|
| AR(1) phi=0.8 | reversion | reversion | 63% | 49% | **-14pp** |
| Sinusoidal | periodic | variance | 86% | 43% | Changed group |
| MR + Oscillation | reversion | dynamic | 46% | 40% | Changed group |
| Trend + Noise | dynamic | dynamic | 36% | 33% | -3pp |

### 3.2 Reversion Group Weight Decline

The reversion group weight decreased on several signals:
- AR(1) phi=0.8: 63% → 49% (-14pp)
- MR + Oscillation: 46% → 40% (now dynamic dominates)

This suggests the scale weight fix penalized MeanReversionModel, likely because it produces poor level predictions due to the phi estimation issue.

---

## 4. Scale Weight Distribution

### 4.1 Before (v3) vs After (v4)

| Scale | v3 Weight | v4 Weight | Change |
|-------|-----------|-----------|--------|
| Scale 1 | ~4% | ~12% | **+8pp** |
| Scale 2 | ~4% | ~26% | **+22pp** |
| Scale 4 | ~5% | ~14% | +9pp |
| Scale 8 | ~21% | ~12% | -9pp |
| Scale 16 | ~12% | ~12% | 0pp |
| Scale 32 | ~20% | ~12% | -8pp |
| Scale 64 | ~37% | ~12% | **-25pp** |

### 4.2 Impact Analysis

The fix achieved its goal of redistributing weight toward shorter scales:
- Scale 2 gained the most (+22pp)
- Scale 64 lost the most (-25pp)

However, this redistribution did not improve coverage because:
1. **Short scales have higher variance estimates** - they see more noise
2. **MeanReversionModel at scale 1 learns phi≈0.01** instead of the true phi
3. **Combining across scales still dilutes predictions**

---

## 5. Category-Level Comparison

### 5.1 Stochastic Signals

| Metric | v3 | v4 | Change |
|--------|----|----|--------|
| Avg Coverage h=1 | 63% | 57.5% | -5.5pp |
| Avg MAE h=1 | 0.96 | ~0.95 | Stable |
| Avg MAE h=1024 | 44.50 | ~59.0 | +14.5 |

### 5.2 Composite Signals

| Metric | v3 | v4 | Change |
|--------|----|----|--------|
| Avg Coverage h=1 | 76% | 63.9% | **-12.1pp** |
| Avg MAE h=1 | 0.66 | ~0.66 | Stable |
| Avg MAE h=1024 | 37.71 | ~37.9 | Stable |

### 5.3 Non-Stationary Signals

| Metric | v3 | v4 | Change |
|--------|----|----|--------|
| Avg Coverage h=1 | 65% | 60.2% | -4.8pp |
| Avg MAE h=1 | 1.27 | ~1.25 | Stable |

---

## 6. Root Cause Analysis

### 6.1 Why Coverage Decreased

The scale weight fix correctly identified that larger scales were being over-weighted for short-horizon predictions. However:

1. **Short scales have calibration issues**
   - MeanReversionModel at scale 1 receives returns, not levels
   - For AR(1) with phi=0.8, returns have phi_returns ≈ -0.2
   - Model clips to phi=0.01, producing wrong predictions

2. **Variance estimates are too narrow at short scales**
   - Short scales see more volatility
   - But models don't account for this in variance
   - Giving short scales more weight → narrower intervals → lower coverage

3. **Between-scale disagreement**
   - Different scales predict different level changes
   - Combined prediction dilutes accuracy
   - Between_var captures disagreement but within_var dominates

### 6.2 Why Some Signals Got Worse

**Random Walk (76% → 65%):**
- Scale 64 was previously dominant (good for random walk)
- Scale 2 now dominant (sees more noise)
- Lower coverage due to noisier predictions

**AR(1) phi=0.99 (80% → 68%):**
- Near unit root benefits from longer scales
- Short scales see mean-reverting behavior
- Reduced accuracy at short horizons

**Mean-Rev + Oscillation (84% → 70%):**
- Previously reversion group dominated (46%)
- Now dynamic group dominates (40%)
- Dynamic models may have different variance characteristics

---

## 7. What the Fix Did Right

Despite the coverage regression, the fix had some positive effects:

1. **More sensible scale weights**
   - Scales are now weighted by level prediction accuracy
   - Short scales appropriately get more weight for short-horizon predictions

2. **Exposed underlying issues**
   - The regression revealed that MeanReversionModel has phi estimation problems
   - Short scales have calibration issues that were masked by long-scale dominance

3. **Accuracy stable or improved for some signals**
   - Trend + Noise improved at h=1024
   - Sinusoidal improved at h=1
   - Most signals have stable h=1 accuracy

---

## 8. Recommendations

### 8.1 Short-Term Fixes

1. **Consider reverting scale weight change** if coverage is priority
2. **Add variance inflation factor** for short scales
3. **Use horizon-dependent scale selection**

### 8.2 Architectural Fixes Needed

1. **Fix MeanReversionModel**
   - Option A: Process levels instead of returns
   - Option B: Adapt phi estimation for return dynamics
   - Option C: Use AR(1) on returns (negative phi allowed)

2. **Improve variance combination**
   - Scale variances appropriately when combining across scales
   - Consider using minimum variance weighting instead of equal weights

3. **Add empirical calibration**
   - Track observed coverage by horizon
   - Adjust interval multipliers dynamically

---

## 9. Summary

| Aspect | v3 | v4 | Verdict |
|--------|----|----|---------|
| Short-horizon coverage | 68.5% | 63.6% | v3 better |
| Long-horizon coverage | 93.7% | 93.3% | Comparable |
| Scale weight distribution | Biased to long | Balanced | v4 conceptually better |
| Accuracy at h=1 | Good | Good | Comparable |
| Accuracy at h=1024 | Good | Slightly worse | v3 better |
| Root cause identification | - | Yes | v4 revealed issues |

**Overall Assessment:** The v4 fix was conceptually correct (scales should be weighted by level prediction accuracy) but revealed that the underlying models are not properly calibrated for short-horizon predictions. The fix should be kept, but additional work is needed to address the variance calibration issues it exposed.

---

*Comparison generated 2025-12-27*
