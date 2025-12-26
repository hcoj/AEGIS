# AEGIS Signal Taxonomy Acceptance Test Report

**Generated:** 2025-12-26 19:29:53
**Total Tests:** 38
**Horizons Tested:** 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024

---

## Executive Summary

### Performance by Horizon

| Horizon | Mean MAE | Mean RMSE | Mean Coverage | MAE Ratio (vs h=1) |
|---------|----------|-----------|---------------|-------------------|
| 1 | nan | nan | 74.9% | 0.00x |
| 2 | nan | nan | 78.5% | 0.00x |
| 4 | nan | nan | 81.9% | 0.00x |
| 8 | nan | nan | 85.3% | 0.00x |
| 16 | nan | nan | 84.9% | 0.00x |
| 32 | nan | nan | 84.8% | 0.00x |
| 64 | nan | nan | 87.9% | 0.00x |
| 128 | nan | nan | 87.6% | 0.00x |
| 256 | nan | nan | 87.6% | 0.00x |
| 512 | 13.2059 | 15.1623 | 86.2% | 0.00x |
| 1024 | 23.1515 | 25.4835 | 85.3% | 0.00x |

**Total Runtime:** 431.77s

---

## Horizon Scaling Analysis

How prediction error grows with forecast horizon:

| Signal | h=1 | h=4 | h=16 | h=64 | h=256 | h=1024 |
|--------|-----|-----|------|------|-------|--------|
| Constant Value | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Linear Trend | 1.56 | 1.26 | 0.15 | 4.74 | 23.94 | 100.77 |
| Sinusoidal | 0.28 | 0.62 | 1.91 | 0.18 | 0.18 | 0.19 |
| Square Wave | 0.27 | 0.45 | 1.21 | 0.21 | 0.20 | 0.21 |
| Polynomial Trend | nan | nan | nan | nan | nan | 377.48 |
| White Noise | 1.13 | 1.17 | 1.15 | 1.13 | 1.15 | 1.15 |
| Random Walk | 4.79 | 4.88 | 5.42 | 8.36 | 16.13 | 25.18 |
| AR(1) phi=0.8 | 0.86 | 0.93 | 1.04 | 1.01 | 1.01 | 0.99 |
| AR(1) phi=0.99 | 2.21 | 2.29 | 2.54 | 3.74 | 6.38 | 5.27 |
| MA(1) | 1.59 | 1.30 | 1.32 | 1.29 | 1.34 | 1.36 |
| ARMA(1,1) | 2.04 | 2.30 | 2.26 | 2.27 | 2.30 | 2.41 |
| Ornstein-Uhlenbeck | 1.14 | 1.36 | 1.62 | 1.63 | 1.72 | 1.73 |
| Trend + Noise | 1.32 | 1.29 | 1.10 | 2.32 | 11.86 | 50.31 |
| Sine + Noise | 0.86 | 0.82 | 1.37 | 0.87 | 0.87 | 0.85 |
| Trend + Seasonality + Noi | 1.11 | 1.11 | 1.87 | 1.22 | 4.77 | 20.19 |
| Mean-Reversion + Oscillat | 1.16 | 1.23 | 1.78 | 1.07 | 1.09 | 1.49 |
| Random Walk with Drift | 4.78 | 4.97 | 5.70 | 8.21 | 12.19 | 36.17 |
| Variance Switching | 2.52 | 2.61 | 2.60 | 2.76 | 2.86 | 3.23 |
| Mean Switching | 1.75 | 1.81 | 2.13 | 3.24 | 3.91 | 5.38 |
| Threshold AR | 0.85 | 0.92 | 0.98 | 1.00 | 0.99 | 1.05 |
| Structural Break | 1.14 | 1.17 | 1.16 | 1.22 | 1.56 | 3.58 |
| Gradual Drift | 1.09 | 1.13 | 1.12 | 1.14 | 1.51 | 5.07 |
| Student-t (df=4) | 2.29 | 2.46 | 2.57 | 2.50 | 2.94 | 4.74 |
| Student-t (df=3) | 2.88 | 3.21 | 3.30 | 3.56 | 4.91 | 10.23 |
| Occasional Jumps | 3.32 | 3.45 | 4.00 | 6.53 | 10.98 | 22.54 |
| Power-Law Tails (alpha=2. | 5.35 | 5.59 | 6.54 | 9.76 | 18.62 | 33.65 |
| fBM Persistent (H=0.7) | 2.84 | 2.80 | 3.09 | 5.29 | 9.93 | 18.34 |
| fBM Antipersistent (H=0.3 | 4.32 | 4.43 | 5.03 | 7.01 | 13.54 | 40.05 |
| Multi-Timescale Mean-Reve | 0.83 | 0.81 | 0.89 | 1.08 | 1.24 | 1.33 |
| Trend + Momentum + Revers | 0.71 | 0.68 | 0.69 | 0.91 | 2.42 | 10.05 |
| GARCH-like Volatility | 1.08 | 1.12 | 1.13 | 1.14 | 1.14 | 1.17 |
| Perfectly Correlated | 4.63 | 4.81 | 5.60 | 7.89 | 12.59 | 23.92 |
| Contemporaneous Relations | 3.51 | 3.64 | 4.30 | 6.22 | 8.86 | 19.71 |
| Lead-Lag | 4.31 | 4.44 | 5.03 | 6.86 | 11.83 | 21.57 |
| Cointegrated Pair | 5.22 | 5.31 | 6.02 | 9.21 | 13.01 | 23.82 |
| Impulse | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| Step Function | 0.27 | 0.29 | 0.38 | 0.77 | 2.49 | 2.27 |
| Contaminated Data | 1.43 | 1.65 | 1.81 | 1.89 | 1.99 | 2.31 |

---

## Coverage by Horizon

95% prediction interval coverage across horizons:

| Signal | h=1 | h=4 | h=16 | h=64 | h=256 | h=1024 |
|--------|-----|-----|------|------|-------|--------|
| Constant Value | 100% | 100% | 100% | 100% | 100% | 100% |
| Linear Trend | 100% | 100% | 100% | 10% | 0% | 0% |
| Sinusoidal | 25% | 17% | 4% | 42% | 40% | 37% |
| Square Wave | 94% | 84% | 47% | 97% | 97% | 100% |
| Polynomial Trend | 89% | 89% | 2% | 0% | 0% | 0% |
| White Noise | 89% | 93% | 99% | 100% | 100% | 100% |
| Random Walk | 59% | 73% | 85% | 88% | 85% | 98% |
| AR(1) phi=0.8 | 75% | 86% | 96% | 100% | 100% | 100% |
| AR(1) phi=0.99 | 58% | 71% | 84% | 89% | 91% | 100% |
| MA(1) | 77% | 94% | 99% | 100% | 100% | 100% |
| ARMA(1,1) | 72% | 80% | 94% | 100% | 100% | 100% |
| Ornstein-Uhlenbeck | 74% | 81% | 90% | 99% | 100% | 100% |
| Trend + Noise | 93% | 96% | 100% | 100% | 96% | 37% |
| Sine + Noise | 97% | 98% | 92% | 100% | 100% | 100% |
| Trend + Seasonality + Noi | 91% | 97% | 93% | 100% | 97% | 50% |
| Mean-Reversion + Oscillat | 52% | 72% | 81% | 100% | 100% | 100% |
| Random Walk with Drift | 59% | 70% | 84% | 87% | 87% | 88% |
| Variance Switching | 85% | 88% | 92% | 93% | 97% | 100% |
| Mean Switching | 80% | 85% | 88% | 94% | 99% | 100% |
| Threshold AR | 76% | 87% | 97% | 100% | 100% | 100% |
| Structural Break | 86% | 88% | 96% | 99% | 100% | 100% |
| Gradual Drift | 90% | 94% | 99% | 100% | 100% | 100% |
| Student-t (df=4) | 69% | 77% | 87% | 97% | 100% | 100% |
| Student-t (df=3) | 64% | 73% | 85% | 93% | 99% | 100% |
| Occasional Jumps | 52% | 60% | 72% | 77% | 82% | 99% |
| Power-Law Tails (alpha=2. | 51% | 62% | 73% | 78% | 77% | 88% |
| fBM Persistent (H=0.7) | 57% | 66% | 75% | 68% | 68% | 75% |
| fBM Antipersistent (H=0.3 | 58% | 70% | 82% | 88% | 83% | 60% |
| Multi-Timescale Mean-Reve | 77% | 90% | 98% | 100% | 100% | 100% |
| Trend + Momentum + Revers | 82% | 94% | 99% | 100% | 100% | 100% |
| GARCH-like Volatility | 88% | 92% | 98% | 100% | 100% | 100% |
| Perfectly Correlated | 60% | 72% | 83% | 89% | 96% | 99% |
| Contemporaneous Relations | 63% | 75% | 85% | 89% | 94% | 92% |
| Lead-Lag | 67% | 79% | 88% | 94% | 95% | 100% |
| Cointegrated Pair | 53% | 68% | 83% | 85% | 94% | 99% |
| Impulse | 100% | 100% | 100% | 100% | 100% | 100% |
| Step Function | 97% | 98% | 97% | 88% | 49% | 22% |
| Contaminated Data | 89% | 91% | 96% | 98% | 100% | 100% |

---

## Composite

| Signal | MAE (h=1) | MAE (h=64) | MAE (h=1024) | Coverage (h=1) | Dominant |
|--------|-----------|------------|--------------|----------------|----------|
| Trend + Noise | 1.3171 | 2.3169 | 50.3094 | 92.7% | dynamic (42%) |
| Sine + Noise | 0.8629 | 0.8669 | 0.8473 | 97.0% | periodic (68%) |
| Trend + Seasonality + Noise | 1.1124 | 1.2224 | 20.1904 | 91.1% | periodic (59%) |
| Mean-Reversion + Oscillation | 1.1567 | 1.0728 | 1.4860 | 52.3% | reversion (49%) |

---

## Deterministic

| Signal | MAE (h=1) | MAE (h=64) | MAE (h=1024) | Coverage (h=1) | Dominant |
|--------|-----------|------------|--------------|----------------|----------|
| Constant Value | 0.0000 | 0.0000 | 0.0000 | 100.0% | persistence (67%) |
| Linear Trend | 1.5632 | 4.7380 | 100.7664 | 100.0% | periodic (99%) |
| Sinusoidal | 0.2798 | 0.1781 | 0.1946 | 25.0% | periodic (86%) |
| Square Wave | 0.2665 | 0.2060 | 0.2096 | 93.8% | reversion (38%) |
| Polynomial Trend | nan | nan | 377.4819 | 88.7% | persistence (nan%) |

---

## Edge Case

| Signal | MAE (h=1) | MAE (h=64) | MAE (h=1024) | Coverage (h=1) | Dominant |
|--------|-----------|------------|--------------|----------------|----------|
| Impulse | 0.0126 | 0.0129 | 0.0143 | 100.0% | persistence (100%) |
| Step Function | 0.2666 | 0.7729 | 2.2730 | 97.5% | variance (86%) |
| Contaminated Data | 1.4317 | 1.8893 | 2.3077 | 89.1% | variance (100%) |

---

## Heavy-Tailed

| Signal | MAE (h=1) | MAE (h=64) | MAE (h=1024) | Coverage (h=1) | Dominant |
|--------|-----------|------------|--------------|----------------|----------|
| Student-t (df=4) | 2.2860 | 2.4950 | 4.7351 | 68.9% | dynamic (37%) |
| Student-t (df=3) | 2.8814 | 3.5641 | 10.2330 | 64.2% | dynamic (44%) |
| Occasional Jumps | 3.3198 | 6.5272 | 22.5380 | 51.6% | reversion (38%) |
| Power-Law Tails (alpha=2.5) | 5.3518 | 9.7574 | 33.6461 | 51.3% | variance (45%) |

---

## Multi-Scale

| Signal | MAE (h=1) | MAE (h=64) | MAE (h=1024) | Coverage (h=1) | Dominant |
|--------|-----------|------------|--------------|----------------|----------|
| fBM Persistent (H=0.7) | 2.8351 | 5.2885 | 18.3379 | 56.9% | reversion (37%) |
| fBM Antipersistent (H=0.3) | 4.3203 | 7.0056 | 40.0503 | 57.9% | reversion (34%) |
| Multi-Timescale Mean-Reversion | 0.8273 | 1.0796 | 1.3289 | 77.1% | dynamic (67%) |
| Trend + Momentum + Reversion | 0.7127 | 0.9069 | 10.0502 | 81.8% | dynamic (62%) |
| GARCH-like Volatility | 1.0780 | 1.1427 | 1.1708 | 88.5% | periodic (52%) |

---

## Multi-Stream

| Signal | MAE (h=1) | MAE (h=64) | MAE (h=1024) | Coverage (h=1) | Dominant |
|--------|-----------|------------|--------------|----------------|----------|
| Perfectly Correlated | 4.6280 | 7.8861 | 23.9244 | 59.7% | reversion (33%) |
| Contemporaneous Relationship | 3.5132 | 6.2221 | 19.7102 | 63.0% | reversion (50%) |
| Lead-Lag | 4.3060 | 6.8595 | 21.5728 | 66.9% | reversion (45%) |
| Cointegrated Pair | 5.2208 | 9.2139 | 23.8170 | 53.3% | reversion (37%) |

---

## Non-Stationary

| Signal | MAE (h=1) | MAE (h=64) | MAE (h=1024) | Coverage (h=1) | Dominant |
|--------|-----------|------------|--------------|----------------|----------|
| Random Walk with Drift | 4.7752 | 8.2110 | 36.1673 | 58.8% | reversion (33%) |
| Variance Switching | 2.5168 | 2.7552 | 3.2283 | 85.0% | periodic (52%) |
| Mean Switching | 1.7541 | 3.2433 | 5.3838 | 79.6% | periodic (26%) |
| Threshold AR | 0.8501 | 1.0047 | 1.0499 | 76.3% | dynamic (57%) |
| Structural Break | 1.1384 | 1.2185 | 3.5757 | 85.7% | periodic (57%) |
| Gradual Drift | 1.0862 | 1.1378 | 5.0743 | 90.1% | periodic (58%) |

---

## Stochastic

| Signal | MAE (h=1) | MAE (h=64) | MAE (h=1024) | Coverage (h=1) | Dominant |
|--------|-----------|------------|--------------|----------------|----------|
| White Noise | 1.1323 | 1.1275 | 1.1466 | 88.8% | periodic (46%) |
| Random Walk | 4.7872 | 8.3586 | 25.1782 | 58.7% | reversion (30%) |
| AR(1) phi=0.8 | 0.8562 | 1.0077 | 0.9916 | 74.7% | dynamic (48%) |
| AR(1) phi=0.99 | 2.2090 | 3.7383 | 5.2696 | 58.3% | reversion (44%) |
| MA(1) | 1.5886 | 1.2947 | 1.3596 | 76.9% | dynamic (99%) |
| ARMA(1,1) | 2.0424 | 2.2748 | 2.4132 | 71.8% | reversion (49%) |
| Ornstein-Uhlenbeck | 1.1411 | 1.6328 | 1.7256 | 73.9% | reversion (44%) |

---

## Detailed Notes

| Signal | Notes |
|--------|-------|
| Constant Value | LocalLevel should converge to constant |
| Linear Trend | LocalTrend or DampedTrend should dominate |
| Sinusoidal | OscillatorBank should dominate with matching period |
| Square Wave | SeasonalDummy should capture sharp transitions |
| Polynomial Trend | LocalTrend tracks slope but underestimates curvature |
| White Noise | RandomWalk should dominate, predicting 0 |
| Random Walk | RandomWalk is optimal for this signal |
| AR(1) phi=0.8 | MeanReversion should dominate |
| AR(1) phi=0.99 | Near unit root - hard to distinguish from RW at short scales |
| MA(1) | MA1 model should capture structure |
| ARMA(1,1) | Combination of AR and MA models |
| Ornstein-Uhlenbeck | Discretized OU is AR(1) toward mean |
| Trend + Noise | LocalTrend should dominate |
| Sine + Noise | OscillatorBank captures periodic component |
| Trend + Seasonality + Noise | Mix of LocalTrend and SeasonalDummy |
| Mean-Reversion + Oscillation | Split weight between reversion and periodic |
| Random Walk with Drift | LocalTrend captures drift |
| Variance Switching | VolatilityTracker should adapt |
| Mean Switching | Break detection should trigger on large shifts |
| Threshold AR | ThresholdAR should learn regimes |
| Structural Break | CUSUM should detect break |
| Gradual Drift | Exponential forgetting tracks drift |
| Student-t (df=4) | QuantileTracker should calibrate intervals |
| Student-t (df=3) | Heavier tails, harder calibration |
| Occasional Jumps | JumpDiffusion should provide jump risk variance |
| Power-Law Tails (alpha=2.5) | Finite variance, trackable |
| fBM Persistent (H=0.7) | Long scales should detect persistence |
| fBM Antipersistent (H=0.3) | Long scales should detect mean-reversion |
| Multi-Timescale Mean-Reversion | Different scales capture different components |
| Trend + Momentum + Reversion | Multi-scale architecture advantage |
| GARCH-like Volatility | VolatilityTracker captures clustering |
| Perfectly Correlated | Cross-stream should identify common factor |
| Contemporaneous Relationship | Lag-0 regression captures beta |
| Lead-Lag | Cross-stream regression learns lag |
| Cointegrated Pair | Cross-stream captures error correction |
| Impulse | LocalLevel tracks then decays |
| Step Function | Break detection may trigger at jumps |
| Contaminated Data | JumpDiffusion absorbs some outliers |

---

## Key Findings

### Error Growth Patterns

**Slowest Error Growth (best long-horizon performance):**
- Constant Value (Deterministic): 0.0x increase from h=1 to h=1024
- Sinusoidal (Deterministic): 0.7x increase from h=1 to h=1024
- Square Wave (Deterministic): 0.8x increase from h=1 to h=1024
- MA(1) (Stochastic): 0.9x increase from h=1 to h=1024
- Sine + Noise (Composite): 1.0x increase from h=1 to h=1024

**Fastest Error Growth (challenging for long-horizon):**
- Trend + Momentum + Reversion (Multi-Scale): 14.1x increase from h=1 to h=1024
- Trend + Seasonality + Noise (Composite): 18.2x increase from h=1 to h=1024
- Trend + Noise (Composite): 38.2x increase from h=1 to h=1024
- Linear Trend (Deterministic): 64.5x increase from h=1 to h=1024
- Polynomial Trend (Deterministic): 377481.9x increase from h=1 to h=1024

---

*Report generated by AEGIS acceptance test suite*