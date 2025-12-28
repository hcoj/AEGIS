# AEGIS Signal Taxonomy Acceptance Test Report

**Generated:** 2025-12-27 00:55:58
**Total Tests:** 38
**Horizons Tested:** 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024

---

## Executive Summary

### Performance by Horizon

| Horizon | Mean MAE | Mean RMSE | Mean Coverage | MAE Ratio (vs h=1) |
|---------|----------|-----------|---------------|-------------------|
| 1 | nan | nan | 83.8% | 0.00x |
| 2 | nan | nan | 85.2% | 0.00x |
| 4 | nan | nan | 85.7% | 0.00x |
| 8 | nan | nan | 86.5% | 0.00x |
| 16 | nan | nan | 87.0% | 0.00x |
| 32 | nan | nan | 86.5% | 0.00x |
| 64 | nan | nan | 89.8% | 0.00x |
| 128 | nan | nan | 89.0% | 0.00x |
| 256 | nan | nan | 85.0% | 0.00x |
| 512 | nan | nan | 77.8% | 0.00x |
| 1024 | 107.2795 | 316.5551 | 65.5% | 0.00x |

**Total Runtime:** 450.00s

---

## Horizon Scaling Analysis

How prediction error grows with forecast horizon:

| Signal | h=1 | h=4 | h=16 | h=64 | h=256 | h=1024 |
|--------|-----|-----|------|------|-------|--------|
| Constant Value | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Linear Trend | 0.10 | 0.10 | 0.10 | 0.10 | 0.11 | 0.18 |
| Sinusoidal | 0.23 | 0.55 | 1.86 | 1.35 | 5.22 | 28.52 |
| Square Wave | 0.12 | 0.31 | 1.07 | 0.09 | 0.13 | 0.31 |
| Polynomial Trend | nan | nan | nan | nan | nan | 101.58 |
| White Noise | 1.13 | 1.18 | 1.20 | 1.48 | 4.19 | 26.82 |
| Random Walk | 1.14 | 1.80 | 3.60 | 9.95 | 30.57 | 130.31 |
| AR(1) phi=0.8 | 0.57 | 0.81 | 1.14 | 1.93 | 6.50 | 32.87 |
| AR(1) phi=0.99 | 0.58 | 0.92 | 1.73 | 4.43 | 15.96 | 57.55 |
| MA(1) | 1.33 | 1.33 | 1.38 | 1.63 | 3.72 | 29.95 |
| ARMA(1,1) | 1.35 | 1.94 | 2.44 | 4.26 | 14.26 | 64.03 |
| Ornstein-Uhlenbeck | 0.59 | 0.94 | 1.53 | 2.76 | 8.72 | 40.37 |
| Trend + Noise | 1.09 | 1.15 | 1.12 | 1.31 | 2.81 | 17.89 |
| Sine + Noise | 0.60 | 0.75 | 1.63 | 3.98 | 16.20 | 73.70 |
| Trend + Seasonality + Noi | 0.60 | 0.81 | 2.16 | 3.32 | 14.02 | 78.47 |
| Mean-Reversion + Oscillat | 0.37 | 0.57 | 1.41 | 2.41 | 9.40 | 52.72 |
| Random Walk with Drift | 1.15 | 1.85 | 3.80 | 9.89 | 29.42 | 121.57 |
| Variance Switching | 2.38 | 2.62 | 3.19 | 6.32 | 23.19 | 220.88 |
| Mean Switching | 1.17 | 1.35 | 1.93 | 4.33 | 16.02 | 163.77 |
| Threshold AR | 0.58 | 0.81 | 1.06 | 1.85 | 6.69 | 41.15 |
| Structural Break | 1.11 | 1.17 | 1.22 | 1.69 | 5.34 | 51.82 |
| Gradual Drift | 1.08 | 1.14 | 1.15 | 1.35 | 3.02 | 25.16 |
| Student-t (df=4) | 1.57 | 2.10 | 2.72 | 4.82 | 23.17 | 275.39 |
| Student-t (df=3) | 1.77 | 2.51 | 3.25 | 7.01 | 45.19 | 489.64 |
| Occasional Jumps | 0.69 | 1.19 | 2.62 | 7.83 | 35.94 | 388.54 |
| Power-Law Tails (alpha=2. | 1.07 | 2.04 | 4.76 | 13.27 | 59.85 | 464.57 |
| fBM Persistent (H=0.7) | 0.44 | 0.83 | 2.11 | 6.69 | 22.08 | 85.34 |
| fBM Antipersistent (H=0.3 | 0.95 | 1.56 | 3.37 | 8.52 | 25.65 | 98.83 |
| Multi-Timescale Mean-Reve | 0.60 | 0.72 | 0.89 | 1.50 | 4.69 | 25.90 |
| Trend + Momentum + Revers | 0.58 | 0.64 | 0.70 | 1.15 | 3.99 | 35.64 |
| GARCH-like Volatility | 1.07 | 1.12 | 1.18 | 1.59 | 4.81 | 38.96 |
| Perfectly Correlated | 1.13 | 1.84 | 3.85 | 9.30 | 26.85 | 106.21 |
| Contemporaneous Relations | 1.06 | 1.59 | 3.02 | 7.33 | 22.37 | 120.08 |
| Lead-Lag | 1.28 | 1.91 | 3.65 | 8.34 | 26.43 | 119.06 |
| Cointegrated Pair | 1.29 | 1.99 | 3.97 | 10.64 | 32.23 | 124.66 |
| Impulse | 0.01 | 0.01 | 0.02 | 0.06 | 0.24 | 1.35 |
| Step Function | 0.02 | 0.07 | 0.27 | 1.07 | 4.57 | 10.72 |
| Contaminated Data | 1.02 | 1.55 | 3.08 | 9.67 | 40.99 | 332.15 |

---

## Coverage by Horizon

95% prediction interval coverage across horizons:

| Signal | h=1 | h=4 | h=16 | h=64 | h=256 | h=1024 |
|--------|-----|-----|------|------|-------|--------|
| Constant Value | 100% | 100% | 100% | 100% | 100% | 100% |
| Linear Trend | 4% | 5% | 6% | 7% | 8% | 13% |
| Sinusoidal | 31% | 27% | 16% | 28% | 15% | 5% |
| Square Wave | 94% | 84% | 47% | 97% | 98% | 100% |
| Polynomial Trend | 4% | 5% | 5% | 7% | 2% | 0% |
| White Noise | 62% | 66% | 81% | 96% | 98% | 89% |
| Random Walk | 97% | 99% | 98% | 95% | 89% | 66% |
| AR(1) phi=0.8 | 85% | 89% | 92% | 97% | 95% | 78% |
| AR(1) phi=0.99 | 98% | 99% | 99% | 98% | 93% | 68% |
| MA(1) | 57% | 69% | 83% | 96% | 99% | 92% |
| ARMA(1,1) | 85% | 89% | 95% | 98% | 96% | 79% |
| Ornstein-Uhlenbeck | 91% | 94% | 95% | 97% | 94% | 74% |
| Trend + Noise | 86% | 90% | 96% | 99% | 100% | 99% |
| Sine + Noise | 100% | 100% | 100% | 95% | 74% | 53% |
| Trend + Seasonality + Noi | 99% | 100% | 100% | 96% | 88% | 69% |
| Mean-Reversion + Oscillat | 98% | 99% | 99% | 99% | 95% | 65% |
| Random Walk with Drift | 98% | 99% | 99% | 97% | 92% | 65% |
| Variance Switching | 88% | 89% | 93% | 93% | 93% | 77% |
| Mean Switching | 90% | 92% | 92% | 94% | 96% | 72% |
| Threshold AR | 87% | 91% | 94% | 97% | 95% | 75% |
| Structural Break | 75% | 79% | 90% | 97% | 98% | 87% |
| Gradual Drift | 65% | 70% | 84% | 96% | 98% | 94% |
| Student-t (df=4) | 90% | 94% | 97% | 98% | 96% | 59% |
| Student-t (df=3) | 94% | 97% | 98% | 99% | 94% | 49% |
| Occasional Jumps | 98% | 99% | 98% | 97% | 85% | 27% |
| Power-Law Tails (alpha=2. | 96% | 97% | 97% | 94% | 80% | 35% |
| fBM Persistent (H=0.7) | 99% | 99% | 99% | 97% | 86% | 43% |
| fBM Antipersistent (H=0.3 | 99% | 99% | 99% | 98% | 94% | 60% |
| Multi-Timescale Mean-Reve | 81% | 87% | 91% | 94% | 92% | 75% |
| Trend + Momentum + Revers | 78% | 86% | 93% | 97% | 95% | 81% |
| GARCH-like Volatility | 72% | 75% | 87% | 97% | 97% | 87% |
| Perfectly Correlated | 98% | 99% | 98% | 98% | 94% | 70% |
| Contemporaneous Relations | 95% | 97% | 97% | 94% | 88% | 72% |
| Lead-Lag | 97% | 98% | 98% | 98% | 95% | 71% |
| Cointegrated Pair | 98% | 99% | 99% | 96% | 94% | 68% |
| Impulse | 100% | 100% | 100% | 100% | 100% | 100% |
| Step Function | 100% | 99% | 97% | 87% | 43% | 15% |
| Contaminated Data | 95% | 95% | 95% | 90% | 79% | 57% |

---

## Composite

| Signal | MAE (h=1) | MAE (h=64) | MAE (h=1024) | Coverage (h=1) | Dominant |
|--------|-----------|------------|--------------|----------------|----------|
| Trend + Noise | 1.0897 | 1.3144 | 17.8902 | 86.1% | dynamic (35%) |
| Sine + Noise | 0.6031 | 3.9767 | 73.7000 | 99.7% | periodic (64%) |
| Trend + Seasonality + Noise | 0.6024 | 3.3235 | 78.4664 | 99.1% | periodic (45%) |
| Mean-Reversion + Oscillation | 0.3662 | 2.4068 | 52.7163 | 97.5% | reversion (48%) |

---

## Deterministic

| Signal | MAE (h=1) | MAE (h=64) | MAE (h=1024) | Coverage (h=1) | Dominant |
|--------|-----------|------------|--------------|----------------|----------|
| Constant Value | 0.0000 | 0.0000 | 0.0000 | 100.0% | persistence (67%) |
| Linear Trend | 0.1000 | 0.1033 | 0.1844 | 4.1% | persistence (100%) |
| Sinusoidal | 0.2281 | 1.3548 | 28.5157 | 30.5% | variance (86%) |
| Square Wave | 0.1248 | 0.0911 | 0.3068 | 93.8% | periodic (41%) |
| Polynomial Trend | nan | nan | 101.5838 | 3.8% | persistence (nan%) |

---

## Edge Case

| Signal | MAE (h=1) | MAE (h=64) | MAE (h=1024) | Coverage (h=1) | Dominant |
|--------|-----------|------------|--------------|----------------|----------|
| Impulse | 0.0080 | 0.0616 | 1.3452 | 100.0% | persistence (100%) |
| Step Function | 0.0243 | 1.0667 | 10.7178 | 99.6% | variance (86%) |
| Contaminated Data | 1.0215 | 9.6666 | 332.1526 | 95.2% | variance (100%) |

---

## Heavy-Tailed

| Signal | MAE (h=1) | MAE (h=64) | MAE (h=1024) | Coverage (h=1) | Dominant |
|--------|-----------|------------|--------------|----------------|----------|
| Student-t (df=4) | 1.5748 | 4.8160 | 275.3883 | 90.2% | dynamic (37%) |
| Student-t (df=3) | 1.7704 | 7.0110 | 489.6369 | 94.4% | reversion (53%) |
| Occasional Jumps | 0.6859 | 7.8264 | 388.5371 | 98.1% | reversion (34%) |
| Power-Law Tails (alpha=2.5) | 1.0696 | 13.2689 | 464.5678 | 96.4% | special (48%) |

---

## Multi-Scale

| Signal | MAE (h=1) | MAE (h=64) | MAE (h=1024) | Coverage (h=1) | Dominant |
|--------|-----------|------------|--------------|----------------|----------|
| fBM Persistent (H=0.7) | 0.4357 | 6.6919 | 85.3425 | 99.5% | reversion (35%) |
| fBM Antipersistent (H=0.3) | 0.9468 | 8.5248 | 98.8265 | 98.7% | reversion (34%) |
| Multi-Timescale Mean-Reversion | 0.6046 | 1.4991 | 25.9012 | 80.9% | dynamic (48%) |
| Trend + Momentum + Reversion | 0.5824 | 1.1550 | 35.6369 | 78.5% | reversion (52%) |
| GARCH-like Volatility | 1.0745 | 1.5878 | 38.9564 | 71.6% | periodic (50%) |

---

## Multi-Stream

| Signal | MAE (h=1) | MAE (h=64) | MAE (h=1024) | Coverage (h=1) | Dominant |
|--------|-----------|------------|--------------|----------------|----------|
| Perfectly Correlated | 1.1252 | 9.2998 | 106.2050 | 97.9% | reversion (33%) |
| Contemporaneous Relationship | 1.0554 | 7.3297 | 120.0841 | 94.9% | reversion (46%) |
| Lead-Lag | 1.2775 | 8.3358 | 119.0614 | 96.9% | reversion (41%) |
| Cointegrated Pair | 1.2860 | 10.6378 | 124.6596 | 97.8% | reversion (35%) |

---

## Non-Stationary

| Signal | MAE (h=1) | MAE (h=64) | MAE (h=1024) | Coverage (h=1) | Dominant |
|--------|-----------|------------|--------------|----------------|----------|
| Random Walk with Drift | 1.1457 | 9.8866 | 121.5710 | 97.8% | reversion (30%) |
| Variance Switching | 2.3821 | 6.3178 | 220.8791 | 87.7% | periodic (44%) |
| Mean Switching | 1.1703 | 4.3320 | 163.7703 | 90.3% | trend (33%) |
| Threshold AR | 0.5847 | 1.8462 | 41.1475 | 87.1% | reversion (59%) |
| Structural Break | 1.1120 | 1.6882 | 51.8193 | 75.0% | periodic (42%) |
| Gradual Drift | 1.0773 | 1.3497 | 25.1552 | 65.2% | periodic (50%) |

---

## Stochastic

| Signal | MAE (h=1) | MAE (h=64) | MAE (h=1024) | Coverage (h=1) | Dominant |
|--------|-----------|------------|--------------|----------------|----------|
| White Noise | 1.1266 | 1.4763 | 26.8237 | 62.3% | periodic (38%) |
| Random Walk | 1.1444 | 9.9548 | 130.3081 | 97.2% | reversion (37%) |
| AR(1) phi=0.8 | 0.5680 | 1.9273 | 32.8713 | 84.7% | reversion (62%) |
| AR(1) phi=0.99 | 0.5796 | 4.4348 | 57.5537 | 98.4% | reversion (37%) |
| MA(1) | 1.3319 | 1.6304 | 29.9453 | 57.3% | dynamic (99%) |
| ARMA(1,1) | 1.3493 | 4.2628 | 64.0257 | 85.0% | dynamic (42%) |
| Ornstein-Uhlenbeck | 0.5935 | 2.7615 | 40.3687 | 91.4% | reversion (39%) |

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
- Linear Trend (Deterministic): 1.8x increase from h=1 to h=1024
- Square Wave (Deterministic): 2.5x increase from h=1 to h=1024
- Trend + Noise (Composite): 16.4x increase from h=1 to h=1024
- MA(1) (Stochastic): 22.5x increase from h=1 to h=1024

**Fastest Error Growth (challenging for long-horizon):**
- Contaminated Data (Edge Case): 325.2x increase from h=1 to h=1024
- Power-Law Tails (alpha=2.5) (Heavy-Tailed): 434.3x increase from h=1 to h=1024
- Step Function (Edge Case): 441.1x increase from h=1 to h=1024
- Occasional Jumps (Heavy-Tailed): 566.4x increase from h=1 to h=1024
- Polynomial Trend (Deterministic): 101583.8x increase from h=1 to h=1024

---

*Report generated by AEGIS acceptance test suite*