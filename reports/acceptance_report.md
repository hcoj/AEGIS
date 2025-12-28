# AEGIS Signal Taxonomy Acceptance Test Report

**Generated:** 2025-12-28 01:39:21
**Total Tests:** 38
**Horizons Tested:** 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024

---

## Executive Summary

### Performance by Horizon

| Horizon | Mean MAE | Mean RMSE | Mean Coverage | MAE Ratio (vs h=1) |
|---------|----------|-----------|---------------|-------------------|
| 1 | 0.8471 | 1.1596 | 86.0% | 1.00x |
| 2 | 0.9628 | 1.3172 | 88.8% | 1.14x |
| 4 | 1.1582 | 1.5978 | 90.2% | 1.37x |
| 8 | 1.4695 | 2.0680 | 90.6% | 1.73x |
| 16 | 1.9875 | 2.9001 | 90.3% | 2.35x |
| 32 | 2.9375 | 4.5171 | 89.3% | 3.47x |
| 64 | 4.3502 | 7.3894 | 92.5% | 5.14x |
| 128 | 7.4422 | 13.9892 | 92.3% | 8.79x |
| 256 | 13.1876 | 27.8210 | 91.9% | 15.57x |
| 512 | 25.2010 | 59.6751 | 91.4% | 29.75x |
| 1024 | 48.6077 | 111.7162 | 92.6% | 57.38x |

**Total Runtime:** 655.51s

---

## Horizon Scaling Analysis

How prediction error grows with forecast horizon:

| Signal | h=1 | h=4 | h=16 | h=64 | h=256 | h=1024 |
|--------|-----|-----|------|------|-------|--------|
| Constant Value | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Linear Trend | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 | 0.12 |
| Sinusoidal | 0.19 | 0.44 | 1.95 | 4.59 | 18.21 | 72.12 |
| Square Wave | 0.15 | 0.42 | 1.50 | 1.81 | 6.97 | 25.83 |
| Polynomial Trend | 0.37 | 0.37 | 0.38 | 0.40 | 0.56 | 3.05 |
| White Noise | 1.12 | 1.11 | 1.12 | 1.11 | 1.25 | 2.05 |
| Random Walk | 1.14 | 1.80 | 3.63 | 10.14 | 30.87 | 107.35 |
| AR(1) phi=0.8 | 0.57 | 0.81 | 1.07 | 1.23 | 1.99 | 5.18 |
| AR(1) phi=0.99 | 0.58 | 0.92 | 1.72 | 4.43 | 15.04 | 41.73 |
| MA(1) | 1.33 | 1.29 | 1.34 | 1.36 | 1.65 | 3.12 |
| ARMA(1,1) | 1.34 | 1.92 | 2.27 | 2.84 | 5.25 | 17.40 |
| Ornstein-Uhlenbeck | 0.59 | 0.93 | 1.46 | 2.12 | 4.71 | 14.29 |
| Trend + Noise | 1.09 | 1.08 | 1.07 | 1.19 | 2.13 | 7.94 |
| Sine + Noise | 0.59 | 0.69 | 1.32 | 0.69 | 1.12 | 3.81 |
| Trend + Seasonality + Noi | 0.58 | 0.70 | 1.36 | 0.91 | 2.50 | 11.61 |
| Mean-Reversion + Oscillat | 0.37 | 0.57 | 1.38 | 1.98 | 6.82 | 25.59 |
| Random Walk with Drift | 1.14 | 1.84 | 3.77 | 9.92 | 27.81 | 93.44 |
| Variance Switching | 2.36 | 2.50 | 3.02 | 5.67 | 20.67 | 87.08 |
| Mean Switching | 1.17 | 1.34 | 2.00 | 4.79 | 15.34 | 60.17 |
| Threshold AR | 0.58 | 0.81 | 1.00 | 1.20 | 1.95 | 6.21 |
| Structural Break | 1.10 | 1.10 | 1.12 | 1.29 | 2.30 | 11.54 |
| Gradual Drift | 1.07 | 1.07 | 1.09 | 1.11 | 1.31 | 2.98 |
| Student-t (df=4) | 1.58 | 2.14 | 2.84 | 4.80 | 14.97 | 53.61 |
| Student-t (df=3) | 1.78 | 2.59 | 3.60 | 7.69 | 25.42 | 83.80 |
| Occasional Jumps | 0.68 | 1.19 | 2.60 | 7.25 | 20.83 | 82.22 |
| Power-Law Tails (alpha=2. | 1.08 | 2.11 | 5.10 | 15.02 | 51.81 | 187.81 |
| fBM Persistent (H=0.7) | 0.44 | 0.84 | 2.13 | 6.78 | 22.22 | 77.41 |
| fBM Antipersistent (H=0.3 | 0.94 | 1.55 | 3.32 | 8.57 | 25.11 | 89.03 |
| Multi-Timescale Mean-Reve | 0.60 | 0.72 | 0.88 | 1.22 | 2.26 | 6.34 |
| Trend + Momentum + Revers | 0.58 | 0.63 | 0.68 | 0.93 | 1.72 | 5.89 |
| GARCH-like Volatility | 1.07 | 1.06 | 1.09 | 1.13 | 1.24 | 2.05 |
| Perfectly Correlated | 1.13 | 1.84 | 3.87 | 9.37 | 26.70 | 99.55 |
| Contemporaneous Relations | 1.06 | 1.60 | 3.01 | 6.79 | 16.94 | 54.29 |
| Lead-Lag | 1.28 | 1.90 | 3.60 | 7.91 | 21.24 | 69.20 |
| Cointegrated Pair | 1.28 | 1.98 | 3.92 | 10.72 | 31.13 | 105.05 |
| Impulse | 0.01 | 0.01 | 0.03 | 0.11 | 0.45 | 2.56 |
| Step Function | 0.02 | 0.07 | 0.28 | 1.09 | 4.65 | 9.88 |
| Contaminated Data | 1.11 | 1.96 | 4.92 | 17.04 | 65.90 | 315.77 |

---

## Coverage by Horizon

95% prediction interval coverage across horizons:

| Signal | h=1 | h=4 | h=16 | h=64 | h=256 | h=1024 |
|--------|-----|-----|------|------|-------|--------|
| Constant Value | 100% | 100% | 100% | 100% | 100% | 100% |
| Linear Trend | 0% | 0% | 0% | 1% | 2% | 4% |
| Sinusoidal | 38% | 36% | 23% | 40% | 45% | 93% |
| Square Wave | 94% | 84% | 47% | 97% | 98% | 99% |
| Polynomial Trend | 0% | 0% | 0% | 1% | 2% | 4% |
| White Noise | 95% | 99% | 100% | 100% | 100% | 100% |
| Random Walk | 89% | 97% | 100% | 100% | 100% | 100% |
| AR(1) phi=0.8 | 89% | 97% | 100% | 100% | 100% | 100% |
| AR(1) phi=0.99 | 88% | 95% | 100% | 100% | 100% | 100% |
| MA(1) | 88% | 100% | 100% | 100% | 100% | 100% |
| ARMA(1,1) | 86% | 98% | 100% | 100% | 100% | 100% |
| Ornstein-Uhlenbeck | 88% | 96% | 100% | 100% | 100% | 100% |
| Trend + Noise | 96% | 99% | 100% | 100% | 100% | 100% |
| Sine + Noise | 94% | 97% | 99% | 100% | 100% | 100% |
| Trend + Seasonality + Noi | 95% | 97% | 99% | 100% | 100% | 100% |
| Mean-Reversion + Oscillat | 88% | 95% | 100% | 100% | 100% | 100% |
| Random Walk with Drift | 89% | 97% | 99% | 100% | 100% | 100% |
| Variance Switching | 94% | 96% | 97% | 98% | 100% | 100% |
| Mean Switching | 96% | 97% | 99% | 100% | 100% | 100% |
| Threshold AR | 91% | 98% | 100% | 100% | 100% | 100% |
| Structural Break | 95% | 99% | 100% | 100% | 100% | 100% |
| Gradual Drift | 95% | 99% | 100% | 100% | 100% | 100% |
| Student-t (df=4) | 92% | 97% | 100% | 100% | 100% | 100% |
| Student-t (df=3) | 93% | 98% | 100% | 100% | 100% | 100% |
| Occasional Jumps | 92% | 94% | 97% | 99% | 100% | 100% |
| Power-Law Tails (alpha=2. | 91% | 94% | 96% | 98% | 99% | 98% |
| fBM Persistent (H=0.7) | 83% | 85% | 83% | 95% | 99% | 100% |
| fBM Antipersistent (H=0.3 | 90% | 96% | 99% | 100% | 100% | 100% |
| Multi-Timescale Mean-Reve | 92% | 98% | 100% | 100% | 100% | 100% |
| Trend + Momentum + Revers | 93% | 99% | 100% | 100% | 100% | 100% |
| GARCH-like Volatility | 96% | 99% | 100% | 100% | 100% | 100% |
| Perfectly Correlated | 90% | 98% | 99% | 100% | 100% | 100% |
| Contemporaneous Relations | 93% | 99% | 100% | 100% | 100% | 100% |
| Lead-Lag | 92% | 99% | 100% | 100% | 100% | 100% |
| Cointegrated Pair | 91% | 99% | 100% | 100% | 100% | 100% |
| Impulse | 100% | 100% | 100% | 100% | 100% | 100% |
| Step Function | 100% | 99% | 97% | 88% | 49% | 20% |
| Contaminated Data | 93% | 95% | 97% | 96% | 97% | 99% |

---

## Composite

| Signal | MAE (h=1) | MAE (h=64) | MAE (h=1024) | Coverage (h=1) | Dominant |
|--------|-----------|------------|--------------|----------------|----------|
| Trend + Noise | 1.0861 | 1.1910 | 7.9351 | 95.7% | dynamic (36%) |
| Sine + Noise | 0.5928 | 0.6949 | 3.8130 | 94.1% | periodic (65%) |
| Trend + Seasonality + Noise | 0.5847 | 0.9123 | 11.6136 | 94.8% | periodic (60%) |
| Mean-Reversion + Oscillation | 0.3664 | 1.9770 | 25.5903 | 88.0% | dynamic (45%) |

---

## Deterministic

| Signal | MAE (h=1) | MAE (h=64) | MAE (h=1024) | Coverage (h=1) | Dominant |
|--------|-----------|------------|--------------|----------------|----------|
| Constant Value | 0.0000 | 0.0000 | 0.0000 | 100.0% | variance (100%) |
| Linear Trend | 0.1000 | 0.1001 | 0.1165 | N/A | periodic (100%) |
| Sinusoidal | 0.1892 | 4.5927 | 72.1182 | 37.6% | variance (73%) |
| Square Wave | 0.1518 | 1.8094 | 25.8289 | 93.8% | reversion (39%) |
| Polynomial Trend | 0.3703 | 0.3986 | 3.0476 | N/A | trend (100%) |

---

## Edge Case

| Signal | MAE (h=1) | MAE (h=64) | MAE (h=1024) | Coverage (h=1) | Dominant |
|--------|-----------|------------|--------------|----------------|----------|
| Impulse | 0.0088 | 0.1113 | 2.5587 | 99.9% | variance (100%) |
| Step Function | 0.0246 | 1.0947 | 9.8804 | 99.6% | variance (100%) |
| Contaminated Data | 1.1089 | 17.0440 | 315.7723 | 92.6% | variance (98%) |

---

## Heavy-Tailed

| Signal | MAE (h=1) | MAE (h=64) | MAE (h=1024) | Coverage (h=1) | Dominant |
|--------|-----------|------------|--------------|----------------|----------|
| Student-t (df=4) | 1.5808 | 4.7998 | 53.6063 | 91.8% | dynamic (37%) |
| Student-t (df=3) | 1.7806 | 7.6923 | 83.8014 | 92.8% | reversion (53%) |
| Occasional Jumps | 0.6849 | 7.2513 | 82.2240 | 92.0% | special (44%) |
| Power-Law Tails (alpha=2.5) | 1.0825 | 15.0190 | 187.8075 | 90.6% | special (58%) |

---

## Multi-Scale

| Signal | MAE (h=1) | MAE (h=64) | MAE (h=1024) | Coverage (h=1) | Dominant |
|--------|-----------|------------|--------------|----------------|----------|
| fBM Persistent (H=0.7) | 0.4378 | 6.7825 | 77.4135 | 83.4% | reversion (34%) |
| fBM Antipersistent (H=0.3) | 0.9448 | 8.5691 | 89.0341 | 89.7% | dynamic (29%) |
| Multi-Timescale Mean-Reversion | 0.6029 | 1.2216 | 6.3395 | 91.7% | dynamic (65%) |
| Trend + Momentum + Reversion | 0.5797 | 0.9282 | 5.8935 | 93.1% | dynamic (64%) |
| GARCH-like Volatility | 1.0655 | 1.1252 | 2.0528 | 95.7% | periodic (50%) |

---

## Multi-Stream

| Signal | MAE (h=1) | MAE (h=64) | MAE (h=1024) | Coverage (h=1) | Dominant |
|--------|-----------|------------|--------------|----------------|----------|
| Perfectly Correlated | 1.1261 | 9.3746 | 99.5546 | 90.2% | dynamic (36%) |
| Contemporaneous Relationship | 1.0578 | 6.7865 | 54.2943 | 92.8% | dynamic (63%) |
| Lead-Lag | 1.2783 | 7.9131 | 69.2035 | 92.3% | reversion (41%) |
| Cointegrated Pair | 1.2844 | 10.7209 | 105.0457 | 91.1% | dynamic (31%) |

---

## Non-Stationary

| Signal | MAE (h=1) | MAE (h=64) | MAE (h=1024) | Coverage (h=1) | Dominant |
|--------|-----------|------------|--------------|----------------|----------|
| Random Walk with Drift | 1.1442 | 9.9189 | 93.4365 | 89.2% | reversion (35%) |
| Variance Switching | 2.3616 | 5.6720 | 87.0762 | 94.5% | periodic (51%) |
| Mean Switching | 1.1716 | 4.7892 | 60.1727 | 95.7% | trend (29%) |
| Threshold AR | 0.5844 | 1.2024 | 6.2073 | 91.1% | dynamic (56%) |
| Structural Break | 1.0996 | 1.2856 | 11.5420 | 95.4% | periodic (49%) |
| Gradual Drift | 1.0694 | 1.1106 | 2.9847 | 95.2% | periodic (51%) |

---

## Stochastic

| Signal | MAE (h=1) | MAE (h=64) | MAE (h=1024) | Coverage (h=1) | Dominant |
|--------|-----------|------------|--------------|----------------|----------|
| White Noise | 1.1172 | 1.1123 | 2.0472 | 95.1% | periodic (38%) |
| Random Walk | 1.1447 | 10.1378 | 107.3529 | 89.5% | reversion (36%) |
| AR(1) phi=0.8 | 0.5670 | 1.2314 | 5.1849 | 89.3% | dynamic (47%) |
| AR(1) phi=0.99 | 0.5791 | 4.4288 | 41.7343 | 87.7% | reversion (35%) |
| MA(1) | 1.3282 | 1.3572 | 3.1185 | 87.8% | dynamic (99%) |
| ARMA(1,1) | 1.3434 | 2.8357 | 17.4036 | 85.8% | reversion (49%) |
| Ornstein-Uhlenbeck | 0.5915 | 2.1153 | 14.2850 | 88.2% | reversion (41%) |

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
- fBM Persistent (H=0.7) (Multi-Scale): 176.8x increase from h=1 to h=1024
- Contaminated Data (Edge Case): 284.8x increase from h=1 to h=1024
- Impulse (Edge Case): 291.8x increase from h=1 to h=1024
- Sinusoidal (Deterministic): 381.2x increase from h=1 to h=1024
- Step Function (Edge Case): 401.7x increase from h=1 to h=1024

---

*Report generated by AEGIS acceptance test suite*