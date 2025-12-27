# AEGIS Signal Taxonomy Acceptance Test Report

**Generated:** 2025-12-27 01:45:29
**Total Tests:** 38
**Horizons Tested:** 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024

---

## Executive Summary

### Performance by Horizon

| Horizon | Mean MAE | Mean RMSE | Mean Coverage | MAE Ratio (vs h=1) |
|---------|----------|-----------|---------------|-------------------|
| 1 | nan | nan | 68.5% | 0.00x |
| 2 | nan | nan | 71.8% | 0.00x |
| 4 | nan | nan | 74.8% | 0.00x |
| 8 | nan | nan | 78.7% | 0.00x |
| 16 | nan | nan | 83.1% | 0.00x |
| 32 | nan | nan | 85.9% | 0.00x |
| 64 | nan | nan | 91.3% | 0.00x |
| 128 | nan | nan | 91.9% | 0.00x |
| 256 | nan | nan | 91.9% | 0.00x |
| 512 | nan | nan | 91.8% | 0.00x |
| 1024 | 53.4496 | 118.8241 | 93.7% | 0.00x |

**Total Runtime:** 549.93s

---

## Horizon Scaling Analysis

How prediction error grows with forecast horizon:

| Signal | h=1 | h=4 | h=16 | h=64 | h=256 | h=1024 |
|--------|-----|-----|------|------|-------|--------|
| Constant Value | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Linear Trend | 0.10 | 0.10 | 0.10 | 0.10 | 0.11 | 0.18 |
| Sinusoidal | 0.23 | 0.56 | 1.87 | 1.33 | 5.41 | 29.59 |
| Square Wave | 0.12 | 0.31 | 1.07 | 0.10 | 0.13 | 0.25 |
| Polynomial Trend | nan | nan | nan | nan | nan | 15.73 |
| White Noise | 1.13 | 1.13 | 1.15 | 1.29 | 2.77 | 20.05 |
| Random Walk | 1.14 | 1.80 | 3.61 | 10.05 | 30.60 | 110.37 |
| AR(1) phi=0.8 | 0.57 | 0.82 | 1.18 | 1.91 | 5.78 | 28.30 |
| AR(1) phi=0.99 | 0.58 | 0.92 | 1.73 | 4.43 | 14.82 | 41.67 |
| MA(1) | 1.33 | 1.31 | 1.40 | 1.71 | 3.89 | 22.01 |
| ARMA(1,1) | 1.35 | 1.96 | 2.54 | 4.30 | 13.03 | 55.51 |
| Ornstein-Uhlenbeck | 0.59 | 0.95 | 1.56 | 2.79 | 8.36 | 33.29 |
| Trend + Noise | 1.09 | 1.09 | 1.09 | 1.28 | 2.63 | 13.29 |
| Sine + Noise | 0.61 | 0.77 | 1.93 | 2.14 | 8.03 | 51.79 |
| Trend + Seasonality + Noi | 0.60 | 0.78 | 2.03 | 2.43 | 9.24 | 52.12 |
| Mean-Reversion + Oscillat | 0.37 | 0.57 | 1.43 | 2.37 | 8.69 | 33.64 |
| Random Walk with Drift | 1.15 | 1.85 | 3.81 | 9.86 | 28.78 | 97.39 |
| Variance Switching | 2.38 | 2.54 | 3.11 | 5.84 | 18.50 | 104.32 |
| Mean Switching | 1.17 | 1.35 | 1.97 | 4.33 | 10.89 | 47.21 |
| Threshold AR | 0.58 | 0.82 | 1.10 | 1.75 | 5.25 | 28.61 |
| Structural Break | 1.11 | 1.13 | 1.18 | 1.51 | 3.76 | 27.71 |
| Gradual Drift | 1.07 | 1.09 | 1.11 | 1.25 | 2.35 | 14.78 |
| Student-t (df=4) | 1.58 | 2.14 | 2.84 | 4.92 | 15.91 | 58.73 |
| Student-t (df=3) | 1.78 | 2.59 | 3.54 | 7.48 | 26.92 | 85.04 |
| Occasional Jumps | 0.69 | 1.19 | 2.61 | 7.26 | 20.81 | 80.46 |
| Power-Law Tails (alpha=2. | 1.07 | 2.04 | 4.73 | 12.62 | 45.86 | 158.40 |
| fBM Persistent (H=0.7) | 0.44 | 0.83 | 2.12 | 6.71 | 22.00 | 79.99 |
| fBM Antipersistent (H=0.3 | 0.95 | 1.56 | 3.38 | 8.59 | 25.87 | 92.57 |
| Multi-Timescale Mean-Reve | 0.60 | 0.73 | 0.91 | 1.44 | 4.06 | 22.35 |
| Trend + Momentum + Revers | 0.58 | 0.64 | 0.71 | 1.11 | 3.19 | 19.40 |
| GARCH-like Volatility | 1.08 | 1.08 | 1.15 | 1.38 | 3.22 | 24.72 |
| Perfectly Correlated | 1.13 | 1.83 | 3.87 | 9.43 | 27.18 | 106.63 |
| Contemporaneous Relations | 1.06 | 1.60 | 3.07 | 7.35 | 20.94 | 73.76 |
| Lead-Lag | 1.28 | 1.90 | 3.66 | 8.37 | 24.78 | 87.47 |
| Cointegrated Pair | 1.29 | 1.99 | 4.01 | 10.89 | 33.44 | 120.43 |
| Impulse | 0.01 | 0.01 | 0.02 | 0.06 | 0.24 | 1.35 |
| Step Function | 0.02 | 0.07 | 0.28 | 1.09 | 4.66 | 11.17 |
| Contaminated Data | 1.03 | 1.59 | 3.27 | 10.03 | 36.78 | 180.81 |

---

## Coverage by Horizon

95% prediction interval coverage across horizons:

| Signal | h=1 | h=4 | h=16 | h=64 | h=256 | h=1024 |
|--------|-----|-----|------|------|-------|--------|
| Constant Value | 100% | 100% | 100% | 100% | 100% | 100% |
| Linear Trend | 4% | 5% | 6% | 7% | 9% | 15% |
| Sinusoidal | 23% | 21% | 15% | 22% | 24% | 35% |
| Square Wave | 94% | 84% | 47% | 97% | 99% | 100% |
| Polynomial Trend | 4% | 5% | 6% | 8% | 18% | 90% |
| White Noise | 49% | 59% | 81% | 99% | 100% | 100% |
| Random Walk | 76% | 83% | 91% | 98% | 100% | 100% |
| AR(1) phi=0.8 | 66% | 75% | 94% | 100% | 100% | 100% |
| AR(1) phi=0.99 | 80% | 88% | 94% | 100% | 100% | 100% |
| MA(1) | 52% | 75% | 94% | 100% | 100% | 100% |
| ARMA(1,1) | 66% | 77% | 94% | 100% | 100% | 100% |
| Ornstein-Uhlenbeck | 69% | 77% | 96% | 100% | 100% | 100% |
| Trend + Noise | 81% | 90% | 98% | 100% | 100% | 100% |
| Sine + Noise | 66% | 71% | 70% | 95% | 100% | 100% |
| Trend + Seasonality + Noi | 73% | 78% | 79% | 97% | 100% | 100% |
| Mean-Reversion + Oscillat | 84% | 89% | 95% | 100% | 100% | 100% |
| Random Walk with Drift | 78% | 84% | 95% | 99% | 100% | 100% |
| Variance Switching | 69% | 76% | 88% | 97% | 100% | 100% |
| Mean Switching | 72% | 82% | 89% | 95% | 100% | 100% |
| Threshold AR | 64% | 72% | 93% | 100% | 100% | 100% |
| Structural Break | 55% | 64% | 82% | 96% | 100% | 100% |
| Gradual Drift | 51% | 61% | 82% | 99% | 100% | 100% |
| Student-t (df=4) | 66% | 76% | 93% | 100% | 100% | 100% |
| Student-t (df=3) | 70% | 76% | 94% | 100% | 100% | 100% |
| Occasional Jumps | 81% | 85% | 90% | 97% | 99% | 100% |
| Power-Law Tails (alpha=2. | 76% | 78% | 86% | 97% | 100% | 100% |
| fBM Persistent (H=0.7) | 81% | 83% | 81% | 87% | 97% | 100% |
| fBM Antipersistent (H=0.3 | 80% | 87% | 91% | 98% | 100% | 100% |
| Multi-Timescale Mean-Reve | 56% | 69% | 89% | 100% | 100% | 100% |
| Trend + Momentum + Revers | 62% | 73% | 92% | 99% | 100% | 100% |
| GARCH-like Volatility | 56% | 65% | 84% | 99% | 100% | 100% |
| Perfectly Correlated | 78% | 85% | 93% | 100% | 100% | 100% |
| Contemporaneous Relations | 82% | 88% | 96% | 100% | 100% | 100% |
| Lead-Lag | 81% | 88% | 96% | 100% | 100% | 100% |
| Cointegrated Pair | 76% | 85% | 96% | 100% | 100% | 100% |
| Impulse | 100% | 100% | 100% | 100% | 100% | 100% |
| Step Function | 100% | 99% | 97% | 88% | 49% | 20% |
| Contaminated Data | 83% | 85% | 93% | 97% | 98% | 100% |

---

## Composite

| Signal | MAE (h=1) | MAE (h=64) | MAE (h=1024) | Coverage (h=1) | Dominant |
|--------|-----------|------------|--------------|----------------|----------|
| Trend + Noise | 1.0899 | 1.2796 | 13.2852 | 81.0% | dynamic (36%) |
| Sine + Noise | 0.6060 | 2.1416 | 51.7886 | 66.1% | periodic (49%) |
| Trend + Seasonality + Noise | 0.5975 | 2.4302 | 52.1232 | 72.7% | periodic (59%) |
| Mean-Reversion + Oscillation | 0.3666 | 2.3658 | 33.6393 | 83.7% | reversion (46%) |

---

## Deterministic

| Signal | MAE (h=1) | MAE (h=64) | MAE (h=1024) | Coverage (h=1) | Dominant |
|--------|-----------|------------|--------------|----------------|----------|
| Constant Value | 0.0000 | 0.0000 | 0.0000 | 100.0% | persistence (67%) |
| Linear Trend | 0.1000 | 0.1029 | 0.1830 | 4.2% | persistence (100%) |
| Sinusoidal | 0.2310 | 1.3319 | 29.5907 | 23.4% | periodic (86%) |
| Square Wave | 0.1249 | 0.0970 | 0.2518 | 93.8% | reversion (37%) |
| Polynomial Trend | nan | nan | 15.7280 | 3.9% | persistence (nan%) |

---

## Edge Case

| Signal | MAE (h=1) | MAE (h=64) | MAE (h=1024) | Coverage (h=1) | Dominant |
|--------|-----------|------------|--------------|----------------|----------|
| Impulse | 0.0080 | 0.0616 | 1.3452 | 100.0% | persistence (100%) |
| Step Function | 0.0246 | 1.0915 | 11.1723 | 99.6% | variance (86%) |
| Contaminated Data | 1.0327 | 10.0255 | 180.8144 | 83.5% | variance (90%) |

---

## Heavy-Tailed

| Signal | MAE (h=1) | MAE (h=64) | MAE (h=1024) | Coverage (h=1) | Dominant |
|--------|-----------|------------|--------------|----------------|----------|
| Student-t (df=4) | 1.5789 | 4.9220 | 58.7342 | 66.1% | dynamic (53%) |
| Student-t (df=3) | 1.7828 | 7.4833 | 85.0423 | 70.5% | reversion (46%) |
| Occasional Jumps | 0.6851 | 7.2552 | 80.4611 | 81.1% | special (30%) |
| Power-Law Tails (alpha=2.5) | 1.0694 | 12.6197 | 158.3963 | 75.6% | special (38%) |

---

## Multi-Scale

| Signal | MAE (h=1) | MAE (h=64) | MAE (h=1024) | Coverage (h=1) | Dominant |
|--------|-----------|------------|--------------|----------------|----------|
| fBM Persistent (H=0.7) | 0.4362 | 6.7095 | 79.9853 | 80.9% | reversion (36%) |
| fBM Antipersistent (H=0.3) | 0.9462 | 8.5935 | 92.5669 | 80.4% | reversion (34%) |
| Multi-Timescale Mean-Reversion | 0.6047 | 1.4419 | 22.3473 | 56.1% | dynamic (48%) |
| Trend + Momentum + Reversion | 0.5824 | 1.1080 | 19.4033 | 62.2% | reversion (58%) |
| GARCH-like Volatility | 1.0758 | 1.3797 | 24.7214 | 56.4% | periodic (50%) |

---

## Multi-Stream

| Signal | MAE (h=1) | MAE (h=64) | MAE (h=1024) | Coverage (h=1) | Dominant |
|--------|-----------|------------|--------------|----------------|----------|
| Perfectly Correlated | 1.1255 | 9.4329 | 106.6327 | 77.9% | reversion (30%) |
| Contemporaneous Relationship | 1.0558 | 7.3535 | 73.7559 | 81.7% | reversion (46%) |
| Lead-Lag | 1.2774 | 8.3728 | 87.4747 | 80.7% | reversion (41%) |
| Cointegrated Pair | 1.2859 | 10.8892 | 120.4259 | 76.0% | reversion (36%) |

---

## Non-Stationary

| Signal | MAE (h=1) | MAE (h=64) | MAE (h=1024) | Coverage (h=1) | Dominant |
|--------|-----------|------------|--------------|----------------|----------|
| Random Walk with Drift | 1.1464 | 9.8624 | 97.3916 | 77.5% | reversion (30%) |
| Variance Switching | 2.3750 | 5.8421 | 104.3210 | 69.0% | periodic (49%) |
| Mean Switching | 1.1731 | 4.3310 | 47.2075 | 72.0% | periodic (26%) |
| Threshold AR | 0.5844 | 1.7464 | 28.6110 | 64.3% | reversion (58%) |
| Structural Break | 1.1132 | 1.5125 | 27.7068 | 54.8% | periodic (47%) |
| Gradual Drift | 1.0734 | 1.2466 | 14.7830 | 50.6% | periodic (53%) |

---

## Stochastic

| Signal | MAE (h=1) | MAE (h=64) | MAE (h=1024) | Coverage (h=1) | Dominant |
|--------|-----------|------------|--------------|----------------|----------|
| White Noise | 1.1259 | 1.2931 | 20.0472 | 49.1% | periodic (38%) |
| Random Walk | 1.1438 | 10.0502 | 110.3715 | 75.8% | reversion (36%) |
| AR(1) phi=0.8 | 0.5675 | 1.9065 | 28.2952 | 65.7% | reversion (63%) |
| AR(1) phi=0.99 | 0.5795 | 4.4341 | 41.6676 | 80.0% | reversion (37%) |
| MA(1) | 1.3332 | 1.7081 | 22.0142 | 52.0% | dynamic (99%) |
| ARMA(1,1) | 1.3485 | 4.2963 | 55.5093 | 66.3% | dynamic (43%) |
| Ornstein-Uhlenbeck | 0.5933 | 2.7882 | 33.2881 | 69.2% | reversion (40%) |

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
- Square Wave (Deterministic): 2.0x increase from h=1 to h=1024
- Trend + Noise (Composite): 12.2x increase from h=1 to h=1024
- Gradual Drift (Non-Stationary): 13.8x increase from h=1 to h=1024

**Fastest Error Growth (challenging for long-horizon):**
- Impulse (Edge Case): 168.6x increase from h=1 to h=1024
- Contaminated Data (Edge Case): 175.1x increase from h=1 to h=1024
- fBM Persistent (H=0.7) (Multi-Scale): 183.4x increase from h=1 to h=1024
- Step Function (Edge Case): 453.6x increase from h=1 to h=1024
- Polynomial Trend (Deterministic): 15728.0x increase from h=1 to h=1024

---

*Report generated by AEGIS acceptance test suite*