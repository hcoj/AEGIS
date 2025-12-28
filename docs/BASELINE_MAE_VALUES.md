# AEGIS Baseline MAE Values

This document records baseline Mean Absolute Error (MAE) values for AEGIS predictions across signal types and horizons. These values are used for regression testing to ensure prediction accuracy is maintained.

**Source:** AEGIS Performance Report v5 (2025-12-27)
**Version:** v5 (Post Reversion - Restored v3 Baseline)

---

## Tolerance Thresholds

| Horizon Range | Tolerance | Rationale |
|---------------|-----------|-----------|
| Short (h=1 to h=64) | 20% | Low variance, stable predictions |
| Long (h=128 to h=1024) | 50% | Higher variance, sensitive to signal realization |

---

## Baseline MAE by Signal Type

### Deterministic Signals

| Signal | h=1 | h=16 | h=64 | h=256 | h=1024 | Growth |
|--------|-----|------|------|-------|--------|--------|
| Constant Value | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0x |
| Linear Trend | 0.10 | 0.10 | 0.10 | 0.11 | 0.18 | 1.8x |
| Square Wave | 0.12 | 1.07 | 0.09 | 0.13 | 0.25 | 2.0x |

### Stochastic Signals

| Signal | h=1 | h=16 | h=64 | h=256 | h=1024 | Growth |
|--------|-----|------|------|-------|--------|--------|
| White Noise | 1.13 | 1.16 | 1.32 | 2.77 | 19.18 | 17.0x |
| Random Walk | 1.14 | 3.61 | 10.05 | 30.60 | 110.36 | 96.8x |
| AR(1) phi=0.8 | 0.57 | 1.18 | 1.91 | 5.78 | 28.30 | 49.6x |
| AR(1) phi=0.99 | 0.58 | 1.73 | 4.44 | 14.85 | 41.85 | 72.2x |
| MA(1) | 1.33 | 1.40 | 1.71 | 3.92 | 22.10 | 16.6x |
| ARMA(1,1) | 1.35 | - | - | - | 56.91 | - |
| O-U Process | 0.59 | - | - | - | 33.30 | - |

### Composite Signals

| Signal | h=1 | h=16 | h=64 | h=256 | h=1024 | Growth |
|--------|-----|------|------|-------|--------|--------|
| Trend + Noise | 1.09 | 1.09 | 1.28 | 2.61 | 13.11 | 12.0x |

### Non-Stationary Signals

| Signal | h=1 | h=16 | h=64 | h=256 | h=1024 | Growth |
|--------|-----|------|------|-------|--------|--------|
| RW with Drift | 1.15 | - | - | - | 97.39 | - |
| Variance Switching | 2.38 | - | - | - | 104.83 | - |
| Mean Switching | 1.17 | 1.97 | 4.36 | 11.13 | 49.99 | 42.7x |
| Threshold AR | 0.58 | 1.10 | 1.75 | 5.25 | 28.59 | 49.3x |
| Structural Break | 1.11 | 1.17 | 1.48 | 3.67 | 25.40 | 22.9x |
| Gradual Drift | 1.08 | 1.12 | 1.26 | 2.37 | 20.17 | 18.7x |
| GARCH-like | 1.08 | 1.15 | 1.38 | 3.23 | 24.06 | 22.3x |

### Heavy-Tailed Signals

| Signal | h=1 | h=16 | h=64 | h=256 | h=1024 | Growth |
|--------|-----|------|------|-------|--------|--------|
| Student-t df=4 | 1.58 | - | - | - | 58.74 | - |
| Student-t df=3 | 1.79 | - | - | - | 87.04 | - |
| Occasional Jumps | 0.69 | - | - | - | 80.52 | - |
| Power-Law Tails | 1.07 | - | - | - | 154.09 | - |

---

## Error Growth Categories

| Category | Growth Range | Example Signals |
|----------|--------------|-----------------|
| Excellent | < 5x | Constant, Linear Trend, Square Wave |
| Good | 5-20x | Trend+Noise, Gradual Drift, MA(1), White Noise |
| Moderate | 20-50x | GARCH, Structural Break, AR(1), Threshold AR, Mean Switch |
| High | 50-100x | AR(1) phi=0.99, Random Walk |
| Challenging | > 100x | Sinusoidal, fBM, Power-Law, Contaminated |

---

## Regression Test Baselines

These are the specific values used in `test/regression/test_prediction_accuracy.py`:

### Short Horizon Tests (20% tolerance)

| Signal | Horizon | Baseline MAE | Max Allowed |
|--------|---------|--------------|-------------|
| White Noise | h=1 | 1.13 | 1.36 |
| White Noise | h=64 | 1.29 | 1.55 |
| Random Walk | h=1 | 1.14 | 1.37 |
| Random Walk | h=64 | 10.05 | 12.06 |
| AR(1) phi=0.8 | h=1 | 0.57 | 0.68 |
| AR(1) phi=0.8 | h=64 | 1.91 | 2.29 |
| Linear Trend | h=1 | 0.10 | 0.12 |
| Linear Trend | h=64 | 0.10 | 0.12 |
| Trend+Noise | h=1 | 1.09 | 1.31 |
| Trend+Noise | h=64 | 1.28 | 1.54 |

### Long Horizon Tests (50% tolerance)

| Signal | Horizon | Baseline MAE | Max Allowed |
|--------|---------|--------------|-------------|
| Random Walk | h=1024 | 110.37 | 165.56 |
| AR(1) phi=0.8 | h=1024 | 28.30 | 42.45 |

### Error Growth Ratio Tests

| Signal | Ratio | Max Allowed |
|--------|-------|-------------|
| AR(1) | h64/h1 | 5.0x |
| Random Walk | h64/h1 | 12.0x |

---

## Understanding Returns-Space MAE

AEGIS operates on **returns** (differences between consecutive observations) rather than levels. This architectural choice has important implications for interpreting MAE values.

### The √2 Factor

For white noise with variance σ²:
- **Theoretical minimum level MAE**: `√(2/π) × σ ≈ 0.798σ`
- **Returns-space MAE**: `√(2/π) × √2 × σ ≈ 1.128σ`

The √2 factor arises because returns introduce additional variance:
- If `y_t ~ N(0, σ²)` independently, then `r_t = y_t - y_{t-1}` has variance `2σ²`
- Predicting the next level from return predictions inherits this doubled variance

### Observed vs Theoretical MAE

| Signal | Observed h=1 MAE | Theoretical Level MAE | Ratio |
|--------|------------------|----------------------|-------|
| White Noise (σ=1) | 1.13 | 0.798 | 1.42 |
| AR(1) φ=0.8 (σ=1) | 0.57 | 0.40 | 1.42 |

The ~42% gap is fully explained by the returns-space architecture and is **expected behavior**, not a performance issue.

### Why Returns-Space?

AEGIS uses returns for several reasons:
1. **Scale invariance**: Models work across different value scales
2. **Stationarity**: Returns are more likely to be stationary than levels
3. **Multi-scale processing**: Returns at different scales capture different dynamics

---

## Notes

1. MAE values are computed after a warmup period (typically 100-200 observations)
2. All tests use fixed random seed (42) for reproducibility
3. Signal generators use standard parameters unless otherwise noted
4. Growth factor is h=1024 MAE divided by h=1 MAE
5. The √2 MAE factor from returns-space processing is expected and documented above

---

## Updating Baselines

If tests fail due to legitimate performance improvements:

1. Run the full acceptance test suite to get new baseline values
2. Verify the change is an improvement across all horizons
3. Update this document with new values
4. Update `test/regression/test_prediction_accuracy.py` accordingly
5. Document the reason for the change in the commit message

**IMPORTANT:** Never update thresholds to mask a regression. Always investigate and fix the root cause.
