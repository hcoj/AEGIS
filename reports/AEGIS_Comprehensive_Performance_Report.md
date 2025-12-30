# AEGIS Comprehensive Performance Report

*Generated: 2025-12-30 21:51*

## Executive Summary

- **Signals evaluated**: 34
- **Stochastic signals**: 28
- **Deterministic signals**: 6

### MASE by Horizon (Stochastic Signals)

| Horizon | Mean MASE | Beats Naive | Min | Max |
|---------|-----------|-------------|-----|-----|
| h=1 | 1.276 | 8/28 | 0.757 | 2.696 |
| h=4 | 1.083 | 10/28 | 0.340 | 1.924 |
| h=16 | 1.134 | 10/28 | 0.155 | 3.475 |
| h=64 | 1.464 | 9/28 | 0.400 | 9.259 |
| h=256 | 3.071 | 7/28 | 0.300 | 39.027 |
| h=1024 | 10.790 | 9/28 | 0.577 | 163.737 |

### Coverage by Horizon (95% target)

| Horizon | Mean Coverage | Under 90% |
|---------|---------------|-----------|
| h=1 | 89.5% | 14/28 |
| h=4 | 96.2% | 0/28 |
| h=16 | 99.2% | 0/28 |
| h=64 | 98.6% | 1/28 |
| h=256 | 98.5% | 1/28 |
| h=1024 | 98.5% | 1/28 |

## Results by Category

### Deterministic

| Signal | MAE@h1 | MAE@h64 | Coverage@h64 |
|--------|--------|---------|--------------|
| constant | 0.0000 | 0.0000 | 100% |
| linear_trend | 0.1000 | 0.1000 | 0% |
| sinusoidal | 0.6250 | 0.6294 | 0% |
| square_wave | 3.7039 | 24.2029 | 100% |
| polynomial_trend | 0.0001 | 0.0001 | 0% |

### Simple Stochastic

| Signal | MASE@h1 | MASE@h64 | Coverage@h64 | Dominant Model |
|--------|---------|----------|--------------|----------------|
| white_noise | 0.76 | 0.97 | 100% | MA1@s1 |
| random_walk | 1.48 | 1.12 | 100% | MA1@s2 |
| ar1_phi09 | 1.45 | 1.06 | 100% | MA1@s2 |
| ar1_phi07 | 1.39 | 1.04 | 100% | MA1@s2 |
| ar1_near_unit | 1.48 | 1.23 | 100% | MA1@s2 |
| ma1 | 1.39 | 1.00 | 100% | MA1@s2 |
| arma11 | 1.55 | 1.06 | 100% | MA1@s2 |
| ornstein_uhlenbeck | 1.45 | 1.05 | 100% | MA1@s2 |

### Composite

| Signal | MASE@h1 | MASE@h64 | Coverage@h64 | Dominant Model |
|--------|---------|----------|--------------|----------------|
| trend_plus_noise | 0.81 | 0.40 | 100% | MA1@s1 |
| sine_plus_noise | 1.07 | 1.00 | 100% | OscillatorBank_p32@s16 |
| trend_seasonality_noise | 1.02 | 0.93 | 100% | OscillatorBank_p64@s8 |
| reversion_oscillation | 1.08 | 1.44 | 100% | OscillatorBank_p32@s8 |

### Non-Stationary

| Signal | MASE@h1 | MASE@h64 | Coverage@h64 | Dominant Model |
|--------|---------|----------|--------------|----------------|
| random_walk_drift | 1.48 | 1.31 | 100% | MA1@s2 |
| variance_switching | 0.78 | 0.97 | 100% | MA1@s1 |
| mean_switching | 0.91 | 1.36 | 100% | MA1@s1 |
| threshold_ar | 1.40 | 1.06 | 100% | MA1@s2 |
| structural_break | 0.79 | 0.97 | 100% | MA1@s1 |
| gradual_drift | 0.76 | 0.97 | 100% | MA1@s1 |

### Heavy-Tailed

| Signal | MASE@h1 | MASE@h64 | Coverage@h64 | Dominant Model |
|--------|---------|----------|--------------|----------------|
| student_t_df4 | 0.87 | 1.08 | 100% | MA1@s1 |
| student_t_df3 | 1.13 | 3.01 | 100% | MA1@s1 |
| jump_diffusion | 1.51 | 1.22 | 100% | MA1@s2 |
| power_law | 1.60 | 9.26 | 99% | MA1@s1 |

### Multi-Scale

| Signal | MASE@h1 | MASE@h64 | Coverage@h64 | Dominant Model |
|--------|---------|----------|--------------|----------------|
| fractional_brownian | 1.60 | 1.19 | 100% | MA1@s2 |
| multi_timescale_reversion | 1.38 | 1.03 | 100% | MA1@s2 |
| trend_momentum_reversion | 1.34 | 1.02 | 100% | MA1@s2 |
| garch_like | 0.77 | 0.97 | 100% | MA1@s1 |

### Adversarial

| Signal | MASE@h1 | MASE@h64 | Coverage@h64 | Dominant Model |
|--------|---------|----------|--------------|----------------|
| impulse | inf | inf | 100% | LevelAwareMeanReversion@s |
| step_function | 2.70 | 1.57 | 60% | JumpDiffusion@s64 |
| contaminated | 1.77 | 1.71 | 100% | JumpDiffusion@s32 |

## Model Selection Analysis

### Most Frequently Dominant Models

| Model | Times Dominant |
|-------|----------------|
| MA1Model | 23 |
| OscillatorBankModel_p32 | 3 |
| LevelAwareMeanReversionModel | 2 |
| VolatilityTrackerModel | 2 |
| JumpDiffusionModel | 2 |
| SeasonalDummyModel_p12 | 1 |
| OscillatorBankModel_p64 | 1 |

## Evaluation Timing

- **Total time**: 25.4 minutes
- **Average per signal**: 44.9 seconds
