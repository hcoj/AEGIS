# AEGIS Signal Taxonomy Acceptance Test Report

**Generated:** 2025-12-26 16:15:22
**Total Tests:** 40

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| Mean MAE | 1.7679 |
| Mean RMSE | 2.3011 |
| Mean Coverage | 96.57% |
| Dominant Group Match Rate | 9/40 (22.5%) |
| Total Runtime | 10.95s |

---

## Composite

| Signal | MAE | RMSE | Coverage | Dominant | Expected | Match | Time |
|--------|-----|------|----------|----------|----------|-------|------|
| Trend + Noise | 4.5582 | 4.8331 | 100.0% | dynamic (74.9%) | trend | **No** | 0.14s |
| Sine + Noise | 1.8961 | 2.2010 | 100.0% | reversion (34.0%) | periodic | **No** | 0.11s |
| Trend + Seasonality + Noise | 1.9562 | 2.1657 | 100.0% | periodic (100.0%) | trend | **No** | 0.22s |
| Mean-Reversion + Oscillation | 0.7317 | 0.8945 | 94.4% | reversion (51.5%) | periodic | **No** | 0.17s |

**Category Summary:** MAE=2.2856, RMSE=2.5236, Coverage=98.6%, Matches=0/4

---

## Deterministic

| Signal | MAE | RMSE | Coverage | Dominant | Expected | Match | Time |
|--------|-----|------|----------|----------|----------|-------|------|
| Constant Value | 0.0000 | 0.0000 | 100.0% | reversion (37.5%) | persistence | **No** | 0.13s |
| Linear Trend | 1.6998 | 1.6999 | 100.0% | reversion (25.2%) | trend | **No** | 0.14s |
| Sinusoidal | 0.6410 | 0.7353 | 99.3% | reversion (49.6%) | periodic | **No** | 0.10s |
| Square Wave | 1.1357 | 1.3615 | 100.0% | periodic (100.0%) | periodic | Yes | 0.15s |
| Polynomial Trend | 1.1867 | 1.2050 | 100.0% | trend (100.0%) | trend | Yes | 0.14s |

**Category Summary:** MAE=0.9327, RMSE=1.0003, Coverage=99.9%, Matches=2/5

---

## Edge Case

| Signal | MAE | RMSE | Coverage | Dominant | Expected | Match | Time |
|--------|-----|------|----------|----------|----------|-------|------|
| Impulse | 0.3156 | 1.8196 | 98.7% | periodic (49.4%) | persistence | **No** | 0.06s |
| Step Function | 1.0441 | 1.6840 | 97.6% | variance (100.0%) | persistence | **No** | 0.22s |
| Contaminated Data | 1.8081 | 4.0758 | 96.4% | special (40.9%) | reversion | **No** | 0.23s |
| Very Short Series (n=30) | 0.9045 | 1.1098 | 100.0% | reversion (27.9%) | persistence | **No** | 0.01s |
| Very Long Series (n=2000) | 0.7776 | 0.9803 | 94.4% | reversion (54.7%) | reversion | Yes | 1.61s |

**Category Summary:** MAE=0.9700, RMSE=1.9339, Coverage=97.4%, Matches=1/5

---

## Heavy-Tailed

| Signal | MAE | RMSE | Coverage | Dominant | Expected | Match | Time |
|--------|-----|------|----------|----------|----------|-------|------|
| Student-t (df=4) | 1.8009 | 2.5441 | 96.8% | reversion (66.4%) | reversion | Yes | 0.23s |
| Student-t (df=3) | 1.7574 | 2.3118 | 97.6% | dynamic (45.5%) | reversion | **No** | 0.23s |
| Occasional Jumps | 2.3227 | 3.4271 | 91.6% | special (30.6%) | persistence | **No** | 0.23s |
| Power-Law Tails (alpha=2.5) | 3.4490 | 4.2659 | 98.0% | reversion (38.4%) | persistence | **No** | 0.23s |

**Category Summary:** MAE=2.3325, RMSE=3.1372, Coverage=96.0%, Matches=1/4

---

## Multi-Scale

| Signal | MAE | RMSE | Coverage | Dominant | Expected | Match | Time |
|--------|-----|------|----------|----------|----------|-------|------|
| fBM Persistent (H=0.7) | 1.2725 | 1.5726 | 89.1% | special (34.6%) | persistence | **No** | 0.39s |
| fBM Antipersistent (H=0.3) | 4.7231 | 6.1016 | 90.0% | reversion (39.5%) | persistence | **No** | 0.40s |
| Multi-Timescale Mean-Reversion | 0.6746 | 0.8616 | 93.5% | dynamic (63.2%) | reversion | **No** | 0.40s |
| Trend + Momentum + Reversion | 1.0988 | 1.2989 | 98.9% | dynamic (54.0%) | trend | **No** | 0.41s |
| GARCH-like Volatility | 0.9840 | 1.2499 | 96.0% | periodic (58.0%) | persistence | **No** | 0.40s |

**Category Summary:** MAE=1.7506, RMSE=2.2169, Coverage=93.5%, Matches=0/5

---

## Multi-Stream

| Signal | MAE | RMSE | Coverage | Dominant | Expected | Match | Time |
|--------|-----|------|----------|----------|----------|-------|------|
| Perfectly Correlated | 3.2470 | 4.1878 | 99.3% | reversion (28.0%) | persistence | **No** | 0.33s |
| Contemporaneous Relationship | 2.7290 | 3.4072 | 98.7% | reversion (39.7%) | persistence | **No** | 0.32s |
| Lead-Lag | 2.5224 | 3.1353 | 99.2% | reversion (30.1%) | persistence | **No** | 0.51s |
| Cointegrated Pair | 4.3900 | 5.6888 | 93.2% | reversion (37.8%) | persistence | **No** | 0.51s |

**Category Summary:** MAE=3.2221, RMSE=4.1048, Coverage=97.6%, Matches=0/4

---

## Non-Stationary

| Signal | MAE | RMSE | Coverage | Dominant | Expected | Match | Time |
|--------|-----|------|----------|----------|----------|-------|------|
| Random Walk with Drift | 3.8302 | 4.6046 | 98.7% | reversion (27.2%) | trend | **No** | 0.14s |
| Variance Switching | 2.5964 | 3.8481 | 96.0% | periodic (75.3%) | persistence | **No** | 0.23s |
| Mean Switching | 1.5769 | 2.1089 | 96.8% | reversion (43.8%) | persistence | **No** | 0.22s |
| Threshold AR | 0.7593 | 1.0189 | 94.4% | reversion (49.2%) | reversion | Yes | 0.23s |
| Structural Break | 1.3017 | 1.6944 | 96.8% | dynamic (33.5%) | persistence | **No** | 0.22s |
| Gradual Drift | 1.1160 | 1.3999 | 96.9% | dynamic (45.3%) | trend | **No** | 0.39s |

**Category Summary:** MAE=1.8634, RMSE=2.4458, Coverage=96.6%, Matches=1/6

---

## Stochastic

| Signal | MAE | RMSE | Coverage | Dominant | Expected | Match | Time |
|--------|-----|------|----------|----------|----------|-------|------|
| White Noise | 0.9913 | 1.2231 | 95.3% | periodic (55.0%) | persistence | **No** | 0.14s |
| Random Walk | 2.4602 | 3.2412 | 98.0% | reversion (44.9%) | persistence | **No** | 0.14s |
| AR(1) phi=0.8 | 0.7996 | 1.0212 | 91.6% | reversion (38.7%) | reversion | Yes | 0.22s |
| AR(1) phi=0.99 | 1.7120 | 2.1405 | 87.1% | reversion (39.2%) | persistence | **No** | 0.39s |
| MA(1) | 1.3369 | 1.6638 | 97.3% | dynamic (96.7%) | dynamic | Yes | 0.14s |
| ARMA(1,1) | 1.7004 | 2.1098 | 98.0% | dynamic (45.7%) | dynamic | Yes | 0.22s |
| Ornstein-Uhlenbeck | 0.9075 | 1.1515 | 93.6% | reversion (55.8%) | reversion | Yes | 0.22s |

**Category Summary:** MAE=1.4154, RMSE=1.7930, Coverage=94.4%, Matches=4/7

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
| Very Short Series (n=30) | Incomplete convergence expected |
| Very Long Series (n=2000) | Parameters should fully converge |

---

## Coverage Analysis

The target coverage is 95%. Tests with coverage significantly below this may indicate calibration issues.

| Signal | Coverage | Status |
|--------|----------|--------|
| Constant Value | 100.0% | Good |
| Linear Trend | 100.0% | Good |
| Sinusoidal | 99.3% | Good |
| Square Wave | 100.0% | Good |
| Polynomial Trend | 100.0% | Good |
| White Noise | 95.3% | Good |
| Random Walk | 98.0% | Good |
| AR(1) phi=0.8 | 91.6% | Good |
| AR(1) phi=0.99 | 87.1% | Acceptable |
| MA(1) | 97.3% | Good |
| ARMA(1,1) | 98.0% | Good |
| Ornstein-Uhlenbeck | 93.6% | Good |
| Trend + Noise | 100.0% | Good |
| Sine + Noise | 100.0% | Good |
| Trend + Seasonality + Noise | 100.0% | Good |
| Mean-Reversion + Oscillation | 94.4% | Good |
| Random Walk with Drift | 98.7% | Good |
| Variance Switching | 96.0% | Good |
| Mean Switching | 96.8% | Good |
| Threshold AR | 94.4% | Good |
| Structural Break | 96.8% | Good |
| Gradual Drift | 96.9% | Good |
| Student-t (df=4) | 96.8% | Good |
| Student-t (df=3) | 97.6% | Good |
| Occasional Jumps | 91.6% | Good |
| Power-Law Tails (alpha=2.5) | 98.0% | Good |
| fBM Persistent (H=0.7) | 89.1% | Acceptable |
| fBM Antipersistent (H=0.3) | 90.0% | Acceptable |
| Multi-Timescale Mean-Reversion | 93.5% | Good |
| Trend + Momentum + Reversion | 98.9% | Good |
| GARCH-like Volatility | 96.0% | Good |
| Perfectly Correlated | 99.3% | Good |
| Contemporaneous Relationship | 98.7% | Good |
| Lead-Lag | 99.2% | Good |
| Cointegrated Pair | 93.2% | Good |
| Impulse | 98.7% | Good |
| Step Function | 97.6% | Good |
| Contaminated Data | 96.4% | Good |
| Very Short Series (n=30) | 100.0% | Good |
| Very Long Series (n=2000) | 94.4% | Good |

---

## Performance Rating Summary

Based on Appendix D rating scale:
- **Excellent**: MAE < 0.5 and Coverage > 90%
- **Good**: MAE < 1.0 and Coverage > 80%
- **Moderate**: MAE < 2.0 and Coverage > 70%
- **Poor**: Otherwise

| Rating | Count | Signals |
|--------|-------|---------|
| Excellent | 2 | Constant Value, Impulse |
| Good | 10 | Sinusoidal, White Noise, AR(1) phi=0.8, Ornstein-Uhlenbeck, Mean-Reversion + Oscillation (+5 more) |
| Moderate | 17 | Linear Trend, Square Wave, Polynomial Trend, AR(1) phi=0.99, MA(1) (+12 more) |
| Poor | 11 | Random Walk, Trend + Noise, Random Walk with Drift, Variance Switching, Occasional Jumps (+6 more) |

---

*Report generated by AEGIS acceptance test suite*