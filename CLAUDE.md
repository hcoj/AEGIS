# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AEGIS (Active Epistemic Generative Inference System) is a multi-stream time series prediction system that uses structured model ensembles with Expected Free Energy (EFE) weighting. Rather than using black-box approaches, AEGIS enumerates known time series behaviours (persistence, trend, mean-reversion, oscillation, etc.) and weights them by both predictive performance and epistemic value.

## TDD Workflow

This project strictly follows Test-Driven Development:

1. **Write a failing test** - implement the test for expected behaviour
2. **Run the test** - confirm it fails (red)
3. **Implement minimum code** - write just enough to pass the test
4. **Run the test** - confirm it passes (green)
5. **Format and lint** - `uv run ruff format . && uv run ruff check --fix .`
6. **Run all tests** - ensure no regressions (`uv run pytest test/`)
7. **Commit and push** - use conventional commits, push after each passing test

## Code Style

- **Strict type hints** - all functions, methods, and variables must have type annotations
- **Minimal comments** - only comment where logic isn't self-evident; never comment changes/fixes (that's what git is for)
- **Docstrings required** - every method must have a sensible docstring describing its purpose, parameters, and return value
- **Comments reflect current state** - describe what the code does now, not its history
- **Ad-hoc scripts** - place in `src/tmp/` for execution without permission prompts

```bash
# Format and lint
uv run ruff format .
uv run ruff check --fix .

# Conventional commit examples
git commit -m "test: add test for softmax numerical stability"
git commit -m "feat: implement softmax with overflow protection"
git commit -m "fix: correct eigenvalue bounds in stabilise_dynamics"
git commit -m "refactor: extract precision weighting to helper function"
```

## Development Commands

```bash
# Add dependencies
uv add <package>

# Run all unit tests
uv run pytest test/unit/

# Run with coverage
uv run pytest test/unit/ --cov=src --cov-report=html

# Run integration tests
uv run pytest test/integration/ -m integration

# Run performance tests
uv run pytest test/performance/ -m performance

# Run single test file
uv run pytest test/unit/test_belief_dynamics.py

# Run tests matching pattern
uv run pytest -k "test_softmax"
```



## Architecture

### Two-Phase Implementation

**Phase 1**: Accuracy-based model weighting using log-likelihood (`use_epistemic_value=False`)

**Phase 2**: Expected Free Energy weighting that adds epistemic value for faster regime adaptation (`use_epistemic_value=True`)

### Core Data Flow

1. Raw observation arrives
2. Multi-scale layer computes returns at each lookback period (default scales: 1, 2, 4, 8, 16, 32, 64)
3. Each scale's model bank updates parameters and makes predictions
4. Combiner weights predictions via softmax over cumulative scores
5. Cross-stream layer adjusts predictions using other streams (if multiple streams)
6. Uncertainty layer produces calibrated prediction intervals

### Directory Structure (Target)

```
aegis/
├── aegis/
│   ├── config.py           # AEGISConfig dataclass
│   ├── core/
│   │   ├── prediction.py   # Prediction dataclass
│   │   ├── combiner.py     # EFEModelCombiner
│   │   ├── scale_manager.py
│   │   ├── stream_manager.py
│   │   ├── cross_stream.py
│   │   ├── break_detector.py
│   │   └── quantile_tracker.py
│   ├── models/
│   │   ├── base.py         # TemporalModel ABC
│   │   ├── persistence.py  # RandomWalk, LocalLevel
│   │   ├── trend.py        # LocalTrend, DampedTrend
│   │   ├── reversion.py    # MeanReversion, AsymmetricMR, ThresholdAR
│   │   ├── periodic.py     # OscillatorBank, SeasonalDummy
│   │   ├── dynamic.py      # AR2, MA1, ARMA
│   │   ├── special.py      # JumpDiffusion, ChangePoint
│   │   ├── variance.py     # VolatilityTracker, LevelDependentVol
│   │   └── fep_native.py   # Phase 2 FEP-native models
│   └── system.py           # Main AEGIS class
└── tests/
    ├── conftest.py         # Signal generators (fixtures)
    ├── unit/models/
    └── integration/
```

### Key Interfaces

**TemporalModel** (all models must implement):
- `update(y, t)`: Update model state with observation
- `predict(horizon)`: Return `Prediction(mean, variance)`
- `log_likelihood(y)`: Score observation under predictive distribution
- `reset(partial)`: Reset parameters toward priors (for regime breaks)
- `epistemic_value()`: Return expected information gain (default 0.0, override in Phase 2)

**Model Groups**: persistence, trend, reversion, periodic, dynamic, special, variance

### Model Combination

Weights computed via: `w_m = softmax((cumulative_scores) / temperature)`

Cumulative scores updated each step: `S_m ← λ·S_m + log_likelihood + α·epistemic_value`

Where `λ` is forgetting factor (default 0.99) and `α` is epistemic weight (0 for Phase 1).

## Key Configuration Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `use_epistemic_value` | False | Enable Phase 2 EFE weighting |
| `epistemic_weight` | 1.0 | Weight on epistemic value (α) |
| `likelihood_forget` | 0.99 | Forgetting factor for cumulative scores |
| `temperature` | 1.0 | Softmax temperature for model weights |
| `break_threshold` | 3.0 | CUSUM threshold for regime break detection |
| `scales` | [1,2,4,8,16,32,64] | Multi-scale lookback periods |

## Documentation Reference

Detailed specifications are in `docs/`:
- `AEGIS_Technical_Specification.md`: Full system design and theory
- `AEGIS_Appendix_A_Core_Implementation.md`: Component implementations
- `AEGIS_Appendix_B_Model_Specifications.md`: All model implementations
- `AEGIS_Appendix_C_Implementation_Plan.md`: TDD plan and test fixtures
- `AEGIS_Appendix_D_Signal_Taxonomy.md`: Test signals for validation
