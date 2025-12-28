# AEGIS Signal Taxonomy Acceptance Test Report

**Generated:** 2025-12-28 01:00:42
**Total Tests:** 38
**Horizons Tested:** 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024

---

## Executive Summary

### Performance by Horizon

| Horizon | Mean MAE | Mean RMSE | Mean Coverage | MAE Ratio (vs h=1) |
|---------|----------|-----------|---------------|-------------------|
| 1 | nan | nan | 85.5% | 0.00x |
| 2 | nan | nan | 88.4% | 0.00x |
| 4 | nan | nan | 90.0% | 0.00x |
| 8 | nan | nan | 90.5% | 0.00x |
| 16 | nan | nan | 90.3% | 0.00x |
| 32 | nan | nan | 89.4% | 0.00x |
| 64 | nan | nan | 92.5% | 0.00x |
| 128 | nan | nan | 92.3% | 0.00x |
| 256 | nan | nan | 91.9% | 0.00x |
| 512 | 24.4413 | 56.3142 | 91.4% | 0.00x |
| 1024 | 47.1379 | 103.5351 | 92.6% | 0.00x |

**Total Runtime:** 660.58s

---

## Horizon Scaling Analysis

How prediction error grows with forecast horizon:

| Signal | h=1 | h=4 | h=16 | h=64 | h=256 | h=1024 |
|--------|-----|-----|------|------|-------|--------|
| Constant Value | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Linear Trend | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 | 0.12 |
| Sinusoidal | 0.19 | 0.44 | 1.95 | 4.59 | 18.21 | 72.12 |
| Square Wave | 0.15 | 0.42 | 1.50 | 1.81 | 6.97 | 25.83 |
| Polynomial Trend | nan | nan | nan | nan | nan | 3.05 |
| White Noise | 1.12 | 1.11 | 1.12 | 1.11 | 1.23 | 2.04 |
| Random Walk | 1.14 | 1.80 | 3.63 | 10.14 | 30.87 | 107.34 |
| AR(1) phi=0.8 | 0.57 | 0.81 | 1.07 | 1.23 | 1.99 | 5.19 |
| AR(1) phi=0.99 | 0.58 | 0.92 | 1.72 | 4.43 | 15.04 | 41.73 |
| MA(1) | 1.33 | 1.29 | 1.34 | 1.36 | 1.65 | 3.10 |
| ARMA(1,1) | 1.34 | 1.92 | 2.28 | 2.84 | 5.27 | 17.41 |
| Ornstein-Uhlenbeck | 0.59 | 0.93 | 1.46 | 2.11 | 4.71 | 14.28 |
| Trend + Noise | 1.09 | 1.08 | 1.07 | 1.18 | 2.05 | 7.68 |
| Sine + Noise | 0.59 | 0.69 | 1.32 | 0.70 | 1.12 | 3.82 |
| Trend + Seasonality + Noi | 0.58 | 0.70 | 1.36 | 0.91 | 2.51 | 11.64 |
| Mean-Reversion + Oscillat | 0.37 | 0.57 | 1.38 | 1.98 | 6.82 | 25.59 |
| Random Walk with Drift | 1.14 | 1.84 | 3.77 | 9.92 | 27.81 | 93.43 |
| Variance Switching | 2.36 | 2.52 | 3.11 | 6.04 | 22.17 | 97.08 |
| Mean Switching | 1.17 | 1.34 | 2.00 | 4.78 | 15.34 | 60.52 |
| Threshold AR | 0.58 | 0.81 | 1.00 | 1.20 | 1.95 | 6.23 |
| Structural Break | 1.10 | 1.10 | 1.12 | 1.29 | 2.29 | 11.18 |
| Gradual Drift | 1.07 | 1.07 | 1.09 | 1.12 | 1.33 | 3.16 |
| Student-t (df=4) | 1.58 | 2.14 | 2.83 | 4.78 | 14.88 | 52.88 |
| Student-t (df=3) | 1.78 | 2.60 | 3.63 | 7.82 | 26.01 | 86.00 |
| Occasional Jumps | 0.68 | 1.19 | 2.59 | 7.25 | 20.81 | 82.19 |
| Power-Law Tails (alpha=2. | 1.08 | 2.11 | 5.12 | 15.12 | 52.22 | 190.64 |
| fBM Persistent (H=0.7) | 0.44 | 0.84 | 2.13 | 6.78 | 22.22 | 77.41 |
| fBM Antipersistent (H=0.3 | 0.94 | 1.55 | 3.32 | 8.57 | 25.12 | 89.04 |
| Multi-Timescale Mean-Reve | 0.60 | 0.72 | 0.88 | 1.22 | 2.27 | 6.34 |
| Trend + Momentum + Revers | 0.58 | 0.63 | 0.68 | 0.93 | 1.72 | 5.90 |
| GARCH-like Volatility | 1.07 | 1.06 | 1.09 | 1.13 | 1.24 | 2.05 |
| Perfectly Correlated | 1.13 | 1.84 | 3.87 | 9.37 | 26.70 | 99.56 |
| Contemporaneous Relations | 1.06 | 1.60 | 3.01 | 6.79 | 16.94 | 54.31 |
| Lead-Lag | 1.28 | 1.90 | 3.60 | 7.91 | 21.24 | 69.20 |
| Cointegrated Pair | 1.28 | 1.98 | 3.92 | 10.72 | 31.13 | 105.05 |
| Impulse | 0.01 | 0.01 | 0.03 | 0.11 | 0.45 | 2.56 |
| Step Function | 0.02 | 0.07 | 0.28 | 1.09 | 4.65 | 9.88 |
| Contaminated Data | 1.06 | 1.74 | 3.97 | 13.13 | 51.55 | 245.72 |

---

## Coverage by Horizon

95% prediction interval coverage across horizons:

| Signal | h=1 | h=4 | h=16 | h=64 | h=256 | h=1024 |
|--------|-----|-----|------|------|-------|--------|
| Constant Value | 100% | 100% | 100% | 100% | 100% | 100% |
| Linear Trend | 0% | 0% | 0% | 1% | 2% | 4% |
| Sinusoidal | 29% | 32% | 23% | 40% | 45% | 93% |
| Square Wave | 94% | 84% | 47% | 97% | 98% | 99% |
| Polynomial Trend | 0% | 0% | 0% | 1% | 2% | 4% |
| White Noise | 95% | 99% | 100% | 100% | 100% | 100% |
| Random Walk | 90% | 97% | 100% | 100% | 100% | 100% |
| AR(1) phi=0.8 | 88% | 97% | 100% | 100% | 100% | 100% |
| AR(1) phi=0.99 | 87% | 95% | 100% | 100% | 100% | 100% |
| MA(1) | 87% | 100% | 100% | 100% | 100% | 100% |
| ARMA(1,1) | 86% | 97% | 100% | 100% | 100% | 100% |
| Ornstein-Uhlenbeck | 87% | 96% | 100% | 100% | 100% | 100% |
| Trend + Noise | 95% | 99% | 100% | 100% | 100% | 100% |
| Sine + Noise | 93% | 96% | 99% | 100% | 100% | 100% |
| Trend + Seasonality + Noi | 94% | 96% | 99% | 100% | 100% | 100% |
| Mean-Reversion + Oscillat | 87% | 95% | 100% | 100% | 100% | 100% |
| Random Walk with Drift | 90% | 97% | 99% | 100% | 100% | 100% |
| Variance Switching | 93% | 96% | 97% | 98% | 100% | 100% |
| Mean Switching | 95% | 97% | 99% | 100% | 100% | 100% |
| Threshold AR | 90% | 97% | 100% | 100% | 100% | 100% |
| Structural Break | 95% | 99% | 100% | 100% | 100% | 100% |
| Gradual Drift | 95% | 99% | 100% | 100% | 100% | 100% |
| Student-t (df=4) | 92% | 97% | 100% | 100% | 100% | 100% |
| Student-t (df=3) | 93% | 97% | 100% | 100% | 100% | 100% |
| Occasional Jumps | 92% | 94% | 97% | 99% | 100% | 100% |
| Power-Law Tails (alpha=2. | 90% | 93% | 96% | 98% | 98% | 98% |
| fBM Persistent (H=0.7) | 82% | 85% | 83% | 95% | 99% | 100% |
| fBM Antipersistent (H=0.3 | 90% | 96% | 99% | 100% | 100% | 100% |
| Multi-Timescale Mean-Reve | 91% | 98% | 100% | 100% | 100% | 100% |
| Trend + Momentum + Revers | 92% | 99% | 100% | 100% | 100% | 100% |
| GARCH-like Volatility | 95% | 99% | 100% | 100% | 100% | 100% |
| Perfectly Correlated | 90% | 98% | 99% | 100% | 100% | 100% |
| Contemporaneous Relations | 93% | 99% | 100% | 100% | 100% | 100% |
| Lead-Lag | 93% | 99% | 100% | 100% | 100% | 100% |
| Cointegrated Pair | 92% | 99% | 100% | 100% | 100% | 100% |
| Impulse | 100% | 100% | 100% | 100% | 100% | 100% |
| Step Function | 100% | 99% | 97% | 88% | 49% | 20% |
| Contaminated Data | 92% | 96% | 97% | 97% | 97% | 100% |

---

## Composite

| Signal | MAE (h=1) | MAE (h=64) | MAE (h=1024) | Coverage (h=1) | Dominant |
|--------|-----------|------------|--------------|----------------|----------|
| Trend + Noise | 1.0861 | 1.1839 | 7.6755 | 95.2% | dynamic (35%) |
| Sine + Noise | 0.5928 | 0.6951 | 3.8174 | 93.1% | periodic (65%) |
| Trend + Seasonality + Noise | 0.5847 | 0.9128 | 11.6353 | 93.9% | periodic (59%) |
| Mean-Reversion + Oscillation | 0.3664 | 1.9771 | 25.5948 | 86.9% | dynamic (45%) |

---

## Deterministic

| Signal | MAE (h=1) | MAE (h=64) | MAE (h=1024) | Coverage (h=1) | Dominant |
|--------|-----------|------------|--------------|----------------|----------|
| Constant Value | 0.0000 | 0.0000 | 0.0000 | 100.0% | variance (100%) |
| Linear Trend | 0.1000 | 0.1001 | 0.1165 | N/A | periodic (100%) |
| Sinusoidal | 0.1892 | 4.5927 | 72.1182 | 28.5% | variance (73%) |
| Square Wave | 0.1518 | 1.8094 | 25.8290 | 93.8% | reversion (39%) |
| Polynomial Trend | nan | nan | 3.0476 | N/A | persistence (nan%) |

---

## Edge Case

| Signal | MAE (h=1) | MAE (h=64) | MAE (h=1024) | Coverage (h=1) | Dominant |
|--------|-----------|------------|--------------|----------------|----------|
| Impulse | 0.0088 | 0.1113 | 2.5587 | 100.0% | variance (100%) |
| Step Function | 0.0246 | 1.0947 | 9.8804 | 99.6% | variance (100%) |
| Contaminated Data | 1.0554 | 13.1298 | 245.7151 | 91.7% | variance (99%) |

---

## Heavy-Tailed

| Signal | MAE (h=1) | MAE (h=64) | MAE (h=1024) | Coverage (h=1) | Dominant |
|--------|-----------|------------|--------------|----------------|----------|
| Student-t (df=4) | 1.5806 | 4.7795 | 52.8800 | 91.8% | dynamic (37%) |
| Student-t (df=3) | 1.7819 | 7.8224 | 86.0043 | 92.6% | reversion (53%) |
| Occasional Jumps | 0.6849 | 7.2460 | 82.1940 | 91.6% | special (44%) |
| Power-Law Tails (alpha=2.5) | 1.0832 | 15.1165 | 190.6354 | 90.0% | special (58%) |

---

## Multi-Scale

| Signal | MAE (h=1) | MAE (h=64) | MAE (h=1024) | Coverage (h=1) | Dominant |
|--------|-----------|------------|--------------|----------------|----------|
| fBM Persistent (H=0.7) | 0.4378 | 6.7825 | 77.4136 | 82.1% | reversion (34%) |
| fBM Antipersistent (H=0.3) | 0.9448 | 8.5691 | 89.0364 | 89.8% | dynamic (29%) |
| Multi-Timescale Mean-Reversion | 0.6029 | 1.2218 | 6.3447 | 91.0% | dynamic (65%) |
| Trend + Momentum + Reversion | 0.5797 | 0.9285 | 5.8989 | 91.7% | dynamic (64%) |
| GARCH-like Volatility | 1.0655 | 1.1251 | 2.0458 | 95.1% | periodic (50%) |

---

## Multi-Stream

| Signal | MAE (h=1) | MAE (h=64) | MAE (h=1024) | Coverage (h=1) | Dominant |
|--------|-----------|------------|--------------|----------------|----------|
| Perfectly Correlated | 1.1261 | 9.3747 | 99.5565 | 90.5% | dynamic (36%) |
| Contemporaneous Relationship | 1.0578 | 6.7869 | 54.3061 | 93.5% | dynamic (63%) |
| Lead-Lag | 1.2784 | 7.9129 | 69.1996 | 93.1% | reversion (41%) |
| Cointegrated Pair | 1.2844 | 10.7219 | 105.0475 | 92.0% | dynamic (31%) |

---

## Non-Stationary

| Signal | MAE (h=1) | MAE (h=64) | MAE (h=1024) | Coverage (h=1) | Dominant |
|--------|-----------|------------|--------------|----------------|----------|
| Random Walk with Drift | 1.1442 | 9.9188 | 93.4315 | 90.2% | reversion (35%) |
| Variance Switching | 2.3634 | 6.0412 | 97.0753 | 92.7% | periodic (50%) |
| Mean Switching | 1.1718 | 4.7850 | 60.5168 | 95.4% | special (27%) |
| Threshold AR | 0.5844 | 1.2035 | 6.2273 | 89.8% | dynamic (56%) |
| Structural Break | 1.0996 | 1.2862 | 11.1834 | 94.9% | periodic (48%) |
| Gradual Drift | 1.0696 | 1.1157 | 3.1563 | 95.2% | periodic (53%) |

---

## Stochastic

| Signal | MAE (h=1) | MAE (h=64) | MAE (h=1024) | Coverage (h=1) | Dominant |
|--------|-----------|------------|--------------|----------------|----------|
| White Noise | 1.1172 | 1.1114 | 2.0355 | 94.7% | periodic (37%) |
| Random Walk | 1.1447 | 10.1375 | 107.3433 | 90.1% | reversion (36%) |
| AR(1) phi=0.8 | 0.5670 | 1.2320 | 5.1906 | 88.5% | dynamic (47%) |
| AR(1) phi=0.99 | 0.5791 | 4.4288 | 41.7337 | 87.3% | reversion (35%) |
| MA(1) | 1.3282 | 1.3567 | 3.0979 | 87.5% | dynamic (99%) |
| ARMA(1,1) | 1.3435 | 2.8383 | 17.4148 | 86.5% | reversion (49%) |
| Ornstein-Uhlenbeck | 0.5915 | 2.1146 | 14.2814 | 87.3% | reversion (41%) |

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
- Linear Trend (Deterministic): 1.2x increase from h=1 to h=1024
- White Noise (Stochastic): 1.8x increase from h=1 to h=1024
- GARCH-like Volatility (Multi-Scale): 1.9x increase from h=1 to h=1024
- MA(1) (Stochastic): 2.3x increase from h=1 to h=1024

**Fastest Error Growth (challenging for long-horizon):**
- Contaminated Data (Edge Case): 232.8x increase from h=1 to h=1024
- Impulse (Edge Case): 291.8x increase from h=1 to h=1024
- Sinusoidal (Deterministic): 381.2x increase from h=1 to h=1024
- Step Function (Edge Case): 401.7x increase from h=1 to h=1024
- Polynomial Trend (Deterministic): 3047.6x increase from h=1 to h=1024

---

*Report generated by AEGIS acceptance test suite*