# AEGIS Signal Taxonomy Acceptance Test Report

**Generated:** 2025-12-28 16:59:51
**Total Tests:** 38
**Horizons Tested:** 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024

---

## Executive Summary

### Performance by Horizon

| Horizon | Mean MAE | Mean RMSE | Mean Coverage | MAE Ratio (vs h=1) |
|---------|----------|-----------|---------------|-------------------|
| 1 | 0.7933 | 1.1071 | 85.9% | 1.00x |
| 2 | 0.9436 | 1.3201 | 89.2% | 1.19x |
| 4 | 1.1693 | 1.6437 | 91.5% | 1.47x |
| 8 | 1.5098 | 2.1466 | 92.3% | 1.90x |
| 16 | 2.0264 | 2.9846 | 92.0% | 2.55x |
| 32 | 2.9040 | 4.5049 | 92.9% | 3.66x |
| 64 | 4.1111 | 6.9930 | 91.6% | 5.18x |
| 128 | 6.9817 | 13.4854 | 91.4% | 8.80x |
| 256 | 12.2310 | 27.3807 | 90.9% | 15.42x |
| 512 | 22.9719 | 56.6284 | 90.2% | 28.96x |
| 1024 | 43.4407 | 106.8099 | 90.8% | 54.76x |

**Total Runtime:** 819.14s

---

## Horizon Scaling Analysis

How prediction error grows with forecast horizon:

| Signal | h=1 | h=4 | h=16 | h=64 | h=256 | h=1024 |
|--------|-----|-----|------|------|-------|--------|
| Constant Value | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Linear Trend | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 | 0.12 |
| Sinusoidal | 0.13 | 0.14 | 0.42 | 0.91 | 3.53 | 18.51 |
| Square Wave | 0.06 | 0.15 | 0.39 | 0.06 | 0.06 | 0.06 |
| Polynomial Trend | 0.37 | 0.37 | 0.38 | 0.40 | 0.56 | 3.05 |
| White Noise | 0.84 | 1.05 | 1.10 | 1.11 | 1.24 | 2.00 |
| Random Walk | 1.14 | 1.83 | 3.99 | 10.14 | 30.87 | 107.35 |
| AR(1) phi=0.8 | 0.55 | 0.81 | 1.12 | 1.23 | 1.99 | 5.21 |
| AR(1) phi=0.99 | 0.58 | 0.94 | 1.93 | 4.44 | 15.07 | 41.88 |
| MA(1) | 1.22 | 1.27 | 1.34 | 1.35 | 1.61 | 2.83 |
| ARMA(1,1) | 1.32 | 1.88 | 2.39 | 2.81 | 5.12 | 16.66 |
| Ornstein-Uhlenbeck | 0.58 | 0.93 | 1.60 | 2.11 | 4.71 | 14.27 |
| Trend + Noise | 0.88 | 1.03 | 1.08 | 1.26 | 2.80 | 7.97 |
| Sine + Noise | 0.57 | 0.64 | 1.04 | 0.70 | 1.12 | 3.81 |
| Trend + Seasonality + Noi | 0.56 | 0.64 | 0.99 | 0.72 | 1.70 | 7.53 |
| Mean-Reversion + Oscillat | 0.36 | 0.54 | 1.24 | 1.98 | 6.72 | 24.77 |
| Random Walk with Drift | 1.14 | 1.84 | 4.20 | 9.92 | 27.81 | 93.43 |
| Variance Switching | 2.01 | 2.55 | 3.15 | 5.68 | 20.83 | 79.76 |
| Mean Switching | 1.06 | 1.37 | 2.19 | 5.22 | 17.18 | 72.84 |
| Threshold AR | 0.57 | 0.81 | 1.04 | 1.18 | 1.83 | 5.49 |
| Structural Break | 0.90 | 1.07 | 1.14 | 1.39 | 3.10 | 14.77 |
| Gradual Drift | 0.84 | 1.02 | 1.08 | 1.16 | 1.77 | 3.27 |
| Student-t (df=4) | 1.54 | 2.34 | 3.17 | 4.77 | 14.86 | 55.84 |
| Student-t (df=3) | 1.78 | 3.00 | 4.43 | 7.82 | 26.03 | 78.84 |
| Occasional Jumps | 0.68 | 1.28 | 2.94 | 7.26 | 20.86 | 82.33 |
| Power-Law Tails (alpha=2. | 1.11 | 2.42 | 5.68 | 13.89 | 49.06 | 168.57 |
| fBM Persistent (H=0.7) | 0.43 | 0.84 | 2.31 | 6.84 | 22.25 | 77.41 |
| fBM Antipersistent (H=0.3 | 0.94 | 1.56 | 3.75 | 8.57 | 25.11 | 89.02 |
| Multi-Timescale Mean-Reve | 0.57 | 0.72 | 0.89 | 1.24 | 2.37 | 6.35 |
| Trend + Momentum + Revers | 0.52 | 0.62 | 0.68 | 0.94 | 1.78 | 6.35 |
| GARCH-like Volatility | 0.81 | 1.00 | 1.08 | 1.13 | 1.27 | 2.41 |
| Perfectly Correlated | 1.12 | 1.85 | 4.20 | 9.32 | 26.41 | 99.55 |
| Contemporaneous Relations | 1.04 | 1.61 | 3.08 | 6.92 | 17.73 | 57.57 |
| Lead-Lag | 1.26 | 1.94 | 3.83 | 8.01 | 21.47 | 69.16 |
| Cointegrated Pair | 1.28 | 2.02 | 4.22 | 10.72 | 31.12 | 105.04 |
| Impulse | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| Step Function | 0.02 | 0.05 | 0.18 | 0.94 | 3.98 | 7.66 |
| Contaminated Data | 1.26 | 2.18 | 4.65 | 13.98 | 50.73 | 219.03 |

---

## Coverage by Horizon

95% prediction interval coverage across horizons:

| Signal | h=1 | h=4 | h=16 | h=64 | h=256 | h=1024 |
|--------|-----|-----|------|------|-------|--------|
| Constant Value | 100% | 100% | 100% | 100% | 100% | 100% |
| Linear Trend | 0% | 0% | 0% | 1% | 2% | 4% |
| Sinusoidal | 7% | 84% | 83% | 4% | 5% | 22% |
| Square Wave | 97% | 85% | 48% | 97% | 97% | 97% |
| Polynomial Trend | 0% | 0% | 0% | 1% | 2% | 4% |
| White Noise | 98% | 99% | 100% | 100% | 100% | 100% |
| Random Walk | 90% | 97% | 100% | 100% | 100% | 100% |
| AR(1) phi=0.8 | 90% | 97% | 100% | 100% | 100% | 100% |
| AR(1) phi=0.99 | 87% | 95% | 100% | 100% | 100% | 100% |
| MA(1) | 91% | 100% | 100% | 100% | 100% | 100% |
| ARMA(1,1) | 87% | 98% | 100% | 100% | 100% | 100% |
| Ornstein-Uhlenbeck | 89% | 96% | 100% | 100% | 100% | 100% |
| Trend + Noise | 97% | 99% | 100% | 100% | 100% | 100% |
| Sine + Noise | 94% | 97% | 100% | 100% | 100% | 100% |
| Trend + Seasonality + Noi | 95% | 96% | 100% | 100% | 100% | 100% |
| Mean-Reversion + Oscillat | 88% | 96% | 100% | 100% | 100% | 100% |
| Random Walk with Drift | 90% | 98% | 99% | 100% | 100% | 100% |
| Variance Switching | 95% | 97% | 98% | 99% | 100% | 100% |
| Mean Switching | 96% | 96% | 99% | 100% | 100% | 100% |
| Threshold AR | 92% | 97% | 100% | 100% | 100% | 100% |
| Structural Break | 97% | 99% | 100% | 100% | 100% | 100% |
| Gradual Drift | 97% | 99% | 100% | 100% | 100% | 100% |
| Student-t (df=4) | 93% | 97% | 100% | 100% | 100% | 100% |
| Student-t (df=3) | 93% | 97% | 100% | 100% | 100% | 100% |
| Occasional Jumps | 92% | 94% | 97% | 99% | 100% | 100% |
| Power-Law Tails (alpha=2. | 91% | 94% | 97% | 99% | 100% | 100% |
| fBM Persistent (H=0.7) | 85% | 88% | 84% | 96% | 100% | 100% |
| fBM Antipersistent (H=0.3 | 90% | 96% | 100% | 100% | 100% | 100% |
| Multi-Timescale Mean-Reve | 92% | 98% | 100% | 100% | 100% | 100% |
| Trend + Momentum + Revers | 94% | 99% | 100% | 100% | 100% | 100% |
| GARCH-like Volatility | 98% | 99% | 100% | 100% | 100% | 100% |
| Perfectly Correlated | 90% | 98% | 100% | 100% | 100% | 100% |
| Contemporaneous Relations | 93% | 99% | 100% | 100% | 100% | 100% |
| Lead-Lag | 93% | 99% | 100% | 100% | 100% | 100% |
| Cointegrated Pair | 91% | 99% | 100% | 100% | 100% | 100% |
| Impulse | 100% | 100% | 100% | 100% | 100% | 100% |
| Step Function | 100% | 99% | 97% | 88% | 49% | 22% |
| Contaminated Data | 93% | 95% | 97% | 98% | 98% | 100% |

---

## Composite

| Signal | MAE (h=1) | MAE (h=64) | MAE (h=1024) | Coverage (h=1) | Dominant |
|--------|-----------|------------|--------------|----------------|----------|
| Trend + Noise | 0.8779 | 1.2619 | 7.9683 | 97.4% | dynamic (35%) |
| Sine + Noise | 0.5727 | 0.6951 | 3.8122 | 94.2% | periodic (64%) |
| Trend + Seasonality + Noise | 0.5649 | 0.7166 | 7.5320 | 94.7% | periodic (60%) |
| Mean-Reversion + Oscillation | 0.3562 | 1.9759 | 24.7726 | 88.3% | reversion (39%) |

---

## Deterministic

| Signal | MAE (h=1) | MAE (h=64) | MAE (h=1024) | Coverage (h=1) | Dominant |
|--------|-----------|------------|--------------|----------------|----------|
| Constant Value | 0.0000 | 0.0000 | 0.0000 | 100.0% | reversion (96%) |
| Linear Trend | 0.1000 | 0.1001 | 0.1165 | N/A | periodic (100%) |
| Sinusoidal | 0.1255 | 0.9123 | 18.5135 | 6.7% | periodic (88%) |
| Square Wave | 0.0622 | 0.0622 | 0.0620 | 96.9% | periodic (86%) |
| Polynomial Trend | 0.3700 | 0.3986 | 3.0481 | N/A | trend (100%) |

---

## Edge Case

| Signal | MAE (h=1) | MAE (h=64) | MAE (h=1024) | Coverage (h=1) | Dominant |
|--------|-----------|------------|--------------|----------------|----------|
| Impulse | 0.0069 | 0.0078 | 0.0086 | 99.9% | reversion (100%) |
| Step Function | 0.0181 | 0.9369 | 7.6618 | 99.6% | variance (86%) |
| Contaminated Data | 1.2619 | 13.9841 | 219.0300 | 92.8% | variance (98%) |

---

## Heavy-Tailed

| Signal | MAE (h=1) | MAE (h=64) | MAE (h=1024) | Coverage (h=1) | Dominant |
|--------|-----------|------------|--------------|----------------|----------|
| Student-t (df=4) | 1.5429 | 4.7688 | 55.8363 | 92.7% | dynamic (36%) |
| Student-t (df=3) | 1.7839 | 7.8239 | 78.8371 | 92.7% | reversion (54%) |
| Occasional Jumps | 0.6836 | 7.2618 | 82.3329 | 92.3% | special (45%) |
| Power-Law Tails (alpha=2.5) | 1.1063 | 13.8889 | 168.5707 | 90.8% | special (51%) |

---

## Multi-Scale

| Signal | MAE (h=1) | MAE (h=64) | MAE (h=1024) | Coverage (h=1) | Dominant |
|--------|-----------|------------|--------------|----------------|----------|
| fBM Persistent (H=0.7) | 0.4293 | 6.8407 | 77.4137 | 84.6% | reversion (39%) |
| fBM Antipersistent (H=0.3) | 0.9402 | 8.5686 | 89.0232 | 89.7% | dynamic (29%) |
| Multi-Timescale Mean-Reversion | 0.5684 | 1.2386 | 6.3501 | 92.0% | dynamic (65%) |
| Trend + Momentum + Reversion | 0.5200 | 0.9396 | 6.3490 | 94.3% | dynamic (64%) |
| GARCH-like Volatility | 0.8130 | 1.1310 | 2.4076 | 97.6% | periodic (46%) |

---

## Multi-Stream

| Signal | MAE (h=1) | MAE (h=64) | MAE (h=1024) | Coverage (h=1) | Dominant |
|--------|-----------|------------|--------------|----------------|----------|
| Perfectly Correlated | 1.1189 | 9.3234 | 99.5525 | 90.3% | dynamic (36%) |
| Contemporaneous Relationship | 1.0381 | 6.9160 | 57.5745 | 93.4% | dynamic (73%) |
| Lead-Lag | 1.2625 | 8.0054 | 69.1631 | 92.9% | reversion (41%) |
| Cointegrated Pair | 1.2750 | 10.7207 | 105.0436 | 91.0% | reversion (32%) |

---

## Non-Stationary

| Signal | MAE (h=1) | MAE (h=64) | MAE (h=1024) | Coverage (h=1) | Dominant |
|--------|-----------|------------|--------------|----------------|----------|
| Random Walk with Drift | 1.1380 | 9.9189 | 93.4296 | 89.6% | reversion (36%) |
| Variance Switching | 2.0118 | 5.6790 | 79.7649 | 95.4% | periodic (44%) |
| Mean Switching | 1.0619 | 5.2233 | 72.8422 | 96.1% | trend (30%) |
| Threshold AR | 0.5701 | 1.1816 | 5.4865 | 91.6% | dynamic (56%) |
| Structural Break | 0.8977 | 1.3898 | 14.7661 | 96.7% | periodic (47%) |
| Gradual Drift | 0.8350 | 1.1575 | 3.2688 | 97.4% | periodic (53%) |

---

## Stochastic

| Signal | MAE (h=1) | MAE (h=64) | MAE (h=1024) | Coverage (h=1) | Dominant |
|--------|-----------|------------|--------------|----------------|----------|
| White Noise | 0.8441 | 1.1116 | 2.0029 | 98.1% | periodic (37%) |
| Random Walk | 1.1401 | 10.1376 | 107.3488 | 89.7% | reversion (36%) |
| AR(1) phi=0.8 | 0.5546 | 1.2325 | 5.2099 | 89.7% | dynamic (47%) |
| AR(1) phi=0.99 | 0.5779 | 4.4404 | 41.8840 | 87.4% | reversion (36%) |
| MA(1) | 1.2151 | 1.3492 | 2.8273 | 91.3% | dynamic (99%) |
| ARMA(1,1) | 1.3162 | 2.8062 | 16.6615 | 86.6% | reversion (49%) |
| Ornstein-Uhlenbeck | 0.5830 | 2.1147 | 14.2728 | 88.6% | reversion (42%) |

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
- Square Wave (Deterministic): 1.0x increase from h=1 to h=1024
- Linear Trend (Deterministic): 1.2x increase from h=1 to h=1024
- Impulse (Edge Case): 1.3x increase from h=1 to h=1024
- MA(1) (Stochastic): 2.3x increase from h=1 to h=1024

**Fastest Error Growth (challenging for long-horizon):**
- Sinusoidal (Deterministic): 147.5x increase from h=1 to h=1024
- Power-Law Tails (alpha=2.5) (Heavy-Tailed): 152.4x increase from h=1 to h=1024
- Contaminated Data (Edge Case): 173.6x increase from h=1 to h=1024
- fBM Persistent (H=0.7) (Multi-Scale): 180.3x increase from h=1 to h=1024
- Step Function (Edge Case): 424.2x increase from h=1 to h=1024

---

*Report generated by AEGIS acceptance test suite*