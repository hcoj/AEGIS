# AEGIS Appendix C: Implementation Plan

## Test-Driven Development in Two Phases

---

## Contents

1. [Overview](#1-overview)
2. [Project Setup](#2-project-setup)
3. [Phase 1: Core System](#3-phase-1-core-system)
4. [Phase 1 Integration Tests](#4-phase-1-integration-tests)
5. [Phase 2: Expected Free Energy](#5-phase-2-expected-free-energy)
6. [Phase 2 Integration Tests](#6-phase-2-integration-tests)
7. [Validation Suite](#7-validation-suite)
8. [Timeline and Milestones](#8-timeline-and-milestones)

---

## 1. Overview

### 1.1 Development Philosophy

- **Test-Driven Development**: Every component has tests written before implementation
- **Incremental Integration**: Each model validated independently, then integrated
- **Synthetic Data Validation**: All tests use known data-generating processes
- **Continuous Benchmarking**: Performance tracked throughout development

### 1.2 Phase Structure

**Phase 1** delivers a complete, functional system with accuracy-based model weighting. At the end of Phase 1, AEGIS can be used for production forecasting.

**Phase 2** extends Phase 1 with expected free energy weighting. Models that provide epistemic value contribute to improved regime adaptation.

### 1.3 Success Criteria

**Phase 1 Complete When:**
- All core models pass unit tests
- Dominant model matches expectation for each signal type
- Prediction intervals achieve target coverage (within 5%)
- System handles multi-stream data with cross-stream effects

**Phase 2 Complete When:**
- Epistemic value computed for applicable models
- Demonstrable faster adaptation on regime-change signals
- No degradation on stationary signals
- Comparative evaluation documented

---

## 2. Project Setup

### 2.1 Directory Structure

```
aegis/
├── aegis/
│   ├── __init__.py
│   ├── config.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── prediction.py
│   │   ├── combiner.py
│   │   ├── scale_manager.py
│   │   ├── stream_manager.py
│   │   ├── cross_stream.py
│   │   ├── break_detector.py
│   │   └── quantile_tracker.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── persistence.py
│   │   ├── trend.py
│   │   ├── reversion.py
│   │   ├── periodic.py
│   │   ├── dynamic.py
│   │   ├── special.py
│   │   ├── variance.py
│   │   └── fep_native.py
│   └── system.py
├── tests/
│   ├── conftest.py
│   ├── unit/
│   │   ├── models/
│   │   │   ├── test_persistence.py
│   │   │   ├── test_trend.py
│   │   │   ├── test_reversion.py
│   │   │   ├── test_periodic.py
│   │   │   ├── test_dynamic.py
│   │   │   ├── test_special.py
│   │   │   └── test_variance.py
│   │   ├── test_combiner.py
│   │   ├── test_scale_manager.py
│   │   └── test_stream_manager.py
│   ├── integration/
│   │   ├── test_single_stream.py
│   │   ├── test_multi_stream.py
│   │   ├── test_regime_adaptation.py
│   │   └── test_phase2_efe.py
│   └── validation/
│       ├── test_signal_taxonomy.py
│       └── benchmarks.py
├── pyproject.toml
└── README.md
```

### 2.2 Test Fixtures

```python
# tests/conftest.py

import numpy as np
import pytest
from typing import Callable, Generator

# ============================================================================
# Random State
# ============================================================================

@pytest.fixture
def rng() -> np.random.Generator:
    """Reproducible random number generator."""
    return np.random.default_rng(42)


# ============================================================================
# Signal Generators
# ============================================================================

@pytest.fixture
def constant_signal() -> Callable:
    def generate(n: int, c: float = 5.0) -> np.ndarray:
        return np.full(n, c)
    return generate


@pytest.fixture
def white_noise_signal(rng) -> Callable:
    def generate(n: int, sigma: float = 1.0) -> np.ndarray:
        return rng.normal(0, sigma, n)
    return generate


@pytest.fixture
def random_walk_signal(rng) -> Callable:
    def generate(n: int, sigma: float = 1.0) -> np.ndarray:
        innovations = rng.normal(0, sigma, n)
        return np.cumsum(innovations)
    return generate


@pytest.fixture
def ar1_signal(rng) -> Callable:
    def generate(n: int, phi: float = 0.9, sigma: float = 1.0) -> np.ndarray:
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = phi * y[t-1] + rng.normal(0, sigma)
        return y
    return generate


@pytest.fixture
def linear_trend_signal() -> Callable:
    def generate(n: int, slope: float = 0.1, intercept: float = 0.0) -> np.ndarray:
        return intercept + slope * np.arange(n)
    return generate


@pytest.fixture
def sine_wave_signal() -> Callable:
    def generate(n: int, period: int = 16, amplitude: float = 1.0) -> np.ndarray:
        return amplitude * np.sin(2 * np.pi * np.arange(n) / period)
    return generate


@pytest.fixture
def threshold_ar_signal(rng) -> Callable:
    def generate(
        n: int, 
        tau: float = 0.0,
        phi_low: float = 0.5,
        phi_high: float = 0.95,
        sigma: float = 0.5
    ) -> np.ndarray:
        y = np.zeros(n)
        for t in range(1, n):
            if y[t-1] < tau:
                y[t] = phi_low * y[t-1] + rng.normal(0, sigma)
            else:
                y[t] = phi_high * y[t-1] + rng.normal(0, sigma)
        return y
    return generate


@pytest.fixture
def regime_switching_signal(rng) -> Callable:
    def generate(
        n: int,
        break_point: int = None,
        mean1: float = 0.0,
        mean2: float = 5.0,
        sigma: float = 1.0
    ) -> np.ndarray:
        if break_point is None:
            break_point = n // 2
        
        y = np.zeros(n)
        y[:break_point] = rng.normal(mean1, sigma, break_point)
        y[break_point:] = rng.normal(mean2, sigma, n - break_point)
        return y
    return generate


@pytest.fixture
def jump_diffusion_signal(rng) -> Callable:
    def generate(
        n: int,
        sigma_diff: float = 0.5,
        jump_prob: float = 0.02,
        jump_size: float = 5.0
    ) -> np.ndarray:
        y = np.zeros(n)
        for t in range(1, n):
            diffusion = rng.normal(0, sigma_diff)
            jump = jump_size * (rng.random() < jump_prob) * rng.choice([-1, 1])
            y[t] = y[t-1] + diffusion + jump
        return y
    return generate


@pytest.fixture
def seasonal_signal(rng) -> Callable:
    def generate(
        n: int,
        period: int = 7,
        pattern: list = None,
        noise_sigma: float = 0.5
    ) -> np.ndarray:
        if pattern is None:
            pattern = [10, 12, 15, 14, 13, 8, 5]  # Example weekly pattern
        
        y = np.zeros(n)
        for t in range(n):
            y[t] = pattern[t % period] + rng.normal(0, noise_sigma)
        return y
    return generate


@pytest.fixture
def asymmetric_ar1_signal(rng) -> Callable:
    def generate(
        n: int,
        phi_up: float = 0.7,
        phi_down: float = 0.95,
        sigma: float = 0.5
    ) -> np.ndarray:
        y = np.zeros(n)
        for t in range(1, n):
            if y[t-1] > 0:
                y[t] = phi_up * y[t-1] + rng.normal(0, sigma)
            else:
                y[t] = phi_down * y[t-1] + rng.normal(0, sigma)
        return y
    return generate


@pytest.fixture
def heavy_tailed_signal(rng) -> Callable:
    def generate(n: int, df: float = 4.0) -> np.ndarray:
        from scipy.stats import t
        innovations = t.rvs(df, size=n, random_state=rng)
        return np.cumsum(innovations)
    return generate


# ============================================================================
# Multi-Stream Generators
# ============================================================================

@pytest.fixture
def correlated_streams_signal(rng) -> Callable:
    def generate(n: int, correlation: float = 0.8) -> tuple:
        z1 = rng.normal(0, 1, n)
        z2 = correlation * z1 + np.sqrt(1 - correlation**2) * rng.normal(0, 1, n)
        
        y1 = np.cumsum(z1)
        y2 = np.cumsum(z2)
        return y1, y2
    return generate


@pytest.fixture
def lead_lag_signal(rng) -> Callable:
    def generate(n: int, lag: int = 3, sigma: float = 0.5) -> tuple:
        leader = np.cumsum(rng.normal(0, 1, n))
        follower = np.zeros(n)
        for t in range(lag, n):
            follower[t] = leader[t - lag] + rng.normal(0, sigma)
        return leader, follower
    return generate
```

---

## 3. Phase 1: Core System

### 3.1 Step 1: Base Infrastructure (Days 1-3)

#### 3.1.1 Prediction Dataclass

```python
# Test first
def test_prediction_std():
    pred = Prediction(mean=0.0, variance=4.0)
    assert pred.std == 2.0

def test_prediction_interval():
    pred = Prediction(mean=0.0, variance=1.0)
    lower, upper = pred.interval(0.95)
    assert lower == pytest.approx(-1.96, abs=0.01)
    assert upper == pytest.approx(1.96, abs=0.01)
```

#### 3.1.2 Configuration

```python
def test_config_defaults():
    config = AEGISConfig()
    assert config.use_epistemic_value == False
    assert config.likelihood_forget == 0.99
    assert len(config.scales) == 7

def test_config_validation():
    with pytest.raises(AssertionError):
        config = AEGISConfig(likelihood_forget=1.5)
        config.validate()
```

### 3.2 Step 2: Persistence Models (Days 4-6)

#### 3.2.1 Random Walk

```python
def test_random_walk_predicts_last_value():
    model = RandomWalkModel()
    model.update(5.0, t=0)
    pred = model.predict(horizon=1)
    assert pred.mean == 5.0

def test_random_walk_variance_scales_with_horizon():
    model = RandomWalkModel()
    for i, y in enumerate([1.0, 2.0, 1.5, 2.5, 2.0]):
        model.update(y, t=i)
    
    pred_1 = model.predict(horizon=1)
    pred_5 = model.predict(horizon=5)
    
    assert pred_5.variance == pytest.approx(5 * pred_1.variance, rel=0.01)

def test_random_walk_log_likelihood():
    model = RandomWalkModel()
    model.update(0.0, t=0)
    ll_exact = model.log_likelihood(0.0)
    ll_far = model.log_likelihood(10.0)
    assert ll_exact > ll_far
```

#### 3.2.2 Local Level

```python
def test_local_level_smooths():
    model = LocalLevelModel(alpha=0.3)
    
    # Step from 0 to 10
    for t in range(50):
        model.update(0.0, t)
    for t in range(50, 100):
        model.update(10.0, t)
    
    pred = model.predict(horizon=1)
    assert 9.0 < pred.mean < 10.0

def test_local_level_converges_to_constant():
    model = LocalLevelModel(alpha=0.1)
    
    for t in range(200):
        model.update(5.0, t)
    
    pred = model.predict(horizon=1)
    assert pred.mean == pytest.approx(5.0, abs=0.01)
```

### 3.3 Step 3: Trend Models (Days 7-9)

```python
def test_local_trend_captures_linear():
    model = LocalTrendModel()
    
    for t in range(100):
        model.update(2.0 * t, t)
    
    pred = model.predict(horizon=10)
    expected = 2.0 * 100 + 2.0 * 10
    assert pred.mean == pytest.approx(expected, rel=0.1)

def test_damped_trend_limits_extrapolation():
    model_damped = DampedTrendModel(phi=0.9)
    model_undamped = LocalTrendModel()
    
    for t in range(100):
        y = float(t)
        model_damped.update(y, t)
        model_undamped.update(y, t)
    
    pred_damped = model_damped.predict(horizon=50)
    pred_undamped = model_undamped.predict(horizon=50)
    
    assert pred_damped.mean < pred_undamped.mean
```

### 3.4 Step 4: Mean Reversion Models (Days 10-14)

```python
def test_mean_reversion_predicts_toward_mean(ar1_signal):
    signal = ar1_signal(n=500, phi=0.9)
    model = MeanReversionModel()
    
    for t, y in enumerate(signal):
        model.update(y, t)
    
    # After training, prediction should revert toward mean
    model.last_y = 5.0  # Artificially far from mean
    pred = model.predict(horizon=1)
    
    assert pred.mean < 5.0
    assert pred.mean > model.mu

def test_threshold_ar_regime_dependent():
    model = ThresholdARModel(tau=0.0)
    
    # Manually set different regime behaviours
    model.phi_low = 0.5
    model.phi_high = 0.95
    
    model.last_y = -2.0
    pred_low = model.predict(horizon=1)
    
    model.last_y = 2.0
    pred_high = model.predict(horizon=1)
    
    # High regime should revert slower (closer to current value)
    assert abs(pred_high.mean) > abs(pred_low.mean)

def test_asymmetric_mr_learns_different_speeds(asymmetric_ar1_signal):
    signal = asymmetric_ar1_signal(n=1000, phi_up=0.7, phi_down=0.95)
    model = AsymmetricMeanReversionModel()
    
    for t, y in enumerate(signal):
        model.update(y, t)
    
    # Should learn different reversion speeds
    assert model.phi_up < model.phi_down
```

### 3.5 Step 5: Periodic Models (Days 15-17)

```python
def test_oscillator_captures_sine(sine_wave_signal):
    signal = sine_wave_signal(n=200, period=16)
    model = OscillatorBankModel(periods=[16])
    
    for t, y in enumerate(signal):
        model.update(y, t)
    
    # Predict next value
    pred = model.predict(horizon=1)
    expected = np.sin(2 * np.pi * 201 / 16)
    
    assert pred.mean == pytest.approx(expected, abs=0.1)

def test_seasonal_dummy_learns_pattern(seasonal_signal):
    pattern = [10, 12, 15, 14, 13, 8, 5]
    signal = seasonal_signal(n=350, period=7, pattern=pattern, noise_sigma=0.5)
    model = SeasonalDummyModel(period=7)
    
    for t, y in enumerate(signal):
        model.update(y, t)
    
    # Check learned means approximate pattern
    for i, expected in enumerate(pattern):
        assert model.means[i] == pytest.approx(expected, abs=1.5)
```

### 3.6 Step 6: Dynamic Models (Days 18-20)

```python
def test_ar2_captures_oscillation():
    # AR(2) with complex roots creates oscillation
    model = AR2Model()
    
    # Generate oscillatory AR(2)
    rng = np.random.default_rng(42)
    y = np.zeros(200)
    for t in range(2, 200):
        y[t] = 1.5 * y[t-1] - 0.8 * y[t-2] + rng.normal(0, 0.1)
    
    for t, val in enumerate(y):
        model.update(val, t)
    
    # Phi1 and phi2 should approach true values
    assert model.phi1 == pytest.approx(1.5, abs=0.2)
    assert model.phi2 == pytest.approx(-0.8, abs=0.2)

def test_ma1_captures_shock():
    model = MA1Model(theta=0.6)
    
    # Observe shock
    for t in range(100):
        if t == 50:
            model.update(5.0, t)
        else:
            model.update(0.0, t)
    
    # After shock, should predict theta effect at t=51
    # (Note: model learns from prediction errors, so this tests the structure)
```

### 3.7 Step 7: Special Models (Days 21-24)

```python
def test_jump_diffusion_detects_jumps(jump_diffusion_signal):
    signal = jump_diffusion_signal(n=500, jump_prob=0.05, jump_size=5.0)
    model = JumpDiffusionModel()
    
    for t, y in enumerate(signal):
        model.update(y, t)
    
    # Should have detected some jumps
    assert model.lambda_mean() > 0.01
    assert len(model.recent_jumps) > 0

def test_jump_diffusion_variance_includes_jump_risk():
    model = JumpDiffusionModel()
    model.lambda_a = 5.0
    model.lambda_b = 95.0  # 5% jump rate
    model.sigma_sq_diff = 1.0
    model.sigma_sq_jump = 25.0
    
    pred = model.predict(horizon=1)
    
    # Variance should exceed pure diffusion
    assert pred.variance > 1.0

def test_change_point_detects_mean_shift(regime_switching_signal):
    signal = regime_switching_signal(n=200, break_point=100, mean1=0.0, mean2=5.0)
    model = ChangePointModel()
    
    for t, y in enumerate(signal):
        model.update(y, t)
    
    # After break, should track new mean
    assert model.mu == pytest.approx(5.0, abs=1.0)
```

### 3.8 Step 8: Model Combiner (Days 25-27)

```python
def test_combiner_weights_sum_to_one():
    combiner = EFEModelCombiner(n_models=5, config=AEGISConfig())
    assert combiner.get_weights().sum() == pytest.approx(1.0)

def test_combiner_higher_likelihood_higher_weight():
    config = AEGISConfig(use_epistemic_value=False)
    combiner = EFEModelCombiner(n_models=3, config=config)
    
    # Simulate model 1 being more accurate
    class MockModel:
        def __init__(self, ll):
            self._ll = ll
        def log_likelihood(self, y): return self._ll
        def epistemic_value(self): return 0.0
        def update(self, y, t): pass
    
    models = [MockModel(-2.0), MockModel(-0.5), MockModel(-2.0)]
    
    for _ in range(50):
        combiner.update(models, y_observed=0.0)
    
    weights = combiner.get_weights()
    assert weights[1] > weights[0]
    assert weights[1] > weights[2]

def test_combiner_forgetting():
    config = AEGISConfig(likelihood_forget=0.9)
    combiner = EFEModelCombiner(n_models=2, config=config)
    
    class MockModel:
        def __init__(self):
            self.good = True
        def log_likelihood(self, y):
            return 0.0 if self.good else -2.0
        def epistemic_value(self): return 0.0
        def update(self, y, t): pass
    
    models = [MockModel(), MockModel()]
    
    # Model 0 good early
    models[0].good = True
    models[1].good = False
    for _ in range(50):
        combiner.update(models, 0.0)
    
    # Model 1 good later
    models[0].good = False
    models[1].good = True
    for _ in range(50):
        combiner.update(models, 0.0)
    
    weights = combiner.get_weights()
    # With forgetting, model 1 should dominate
    assert weights[1] > weights[0]

def test_combine_predictions_law_of_total_variance():
    config = AEGISConfig()
    combiner = EFEModelCombiner(n_models=2, config=config)
    combiner.last_weights = np.array([0.5, 0.5])
    
    # Predictions with disagreement
    predictions = [
        Prediction(mean=0.0, variance=1.0),
        Prediction(mean=4.0, variance=1.0)
    ]
    
    combined = combiner.combine_predictions(predictions)
    
    # Mean should be average
    assert combined.mean == 2.0
    
    # Variance should include disagreement
    within_var = 0.5 * 1.0 + 0.5 * 1.0  # = 1.0
    between_var = 0.5 * (0.0 - 2.0)**2 + 0.5 * (4.0 - 2.0)**2  # = 4.0
    expected_var = within_var + between_var
    assert combined.variance == pytest.approx(expected_var)
```

### 3.9 Step 9: Scale Manager (Days 28-30)

```python
def test_scale_manager_computes_returns():
    config = AEGISConfig(scales=[1, 4])
    
    def mock_factory():
        return [RandomWalkModel()]
    
    manager = ScaleManager(config, mock_factory)
    
    # Feed prices
    prices = [100, 101, 102, 101, 103]
    for p in prices:
        manager.observe(p)
    
    # Check scale-4 return
    # history[-1] - history[-1-4] = 103 - 100 = 3
    assert manager.history[-1] - manager.history[-5] == 3.0

def test_scale_manager_multi_scale_prediction():
    config = AEGISConfig(scales=[1, 4, 16])
    
    def mock_factory():
        return [MeanReversionModel()]
    
    manager = ScaleManager(config, mock_factory)
    
    # Feed data
    rng = np.random.default_rng(42)
    for t in range(100):
        manager.observe(rng.normal(0, 1))
    
    pred = manager.predict(horizon=5)
    assert isinstance(pred, Prediction)
    assert pred.variance > 0
```

### 3.10 Step 10: Stream Manager & Integration (Days 31-35)

```python
def test_stream_manager_full_pipeline():
    config = AEGISConfig()
    
    def model_factory():
        return [RandomWalkModel(), MeanReversionModel()]
    
    stream = StreamManager("test", config, model_factory)
    
    rng = np.random.default_rng(42)
    for t in range(100):
        stream.observe(rng.normal(0, 1), t)
        pred = stream.predict(horizon=1)
        assert isinstance(pred, Prediction)

def test_stream_manager_break_detection(regime_switching_signal):
    config = AEGISConfig(break_threshold=2.5)
    
    def model_factory():
        return [LocalLevelModel()]
    
    stream = StreamManager("test", config, model_factory)
    
    signal = regime_switching_signal(n=200, break_point=100, mean1=0.0, mean2=10.0)
    
    breaks_detected = []
    for t, y in enumerate(signal):
        if stream.in_break_adaptation:
            breaks_detected.append(t)
        stream.observe(y, t)
    
    # Should have detected break near t=100
    assert len(breaks_detected) > 0
    assert min(breaks_detected) < 150
```

---

## 4. Phase 1 Integration Tests

### 4.1 Single Stream Tests

```python
class TestSingleStreamIntegration:
    """Integration tests for single-stream processing."""
    
    def test_dominant_model_random_walk(self, random_walk_signal):
        """RandomWalk model should dominate on random walk data."""
        signal = random_walk_signal(n=500)
        system = AEGIS(AEGISConfig())
        system.add_stream("test")
        
        for t, y in enumerate(signal):
            system.observe("test", y, t)
        
        diag = system.get_diagnostics("test")
        
        # RandomWalk should have highest weight in persistence group
        assert diag["group_weights"]["persistence"] > 0.3
    
    def test_dominant_model_ar1(self, ar1_signal):
        """MeanReversion should dominate on AR(1) data."""
        signal = ar1_signal(n=500, phi=0.8)
        system = AEGIS(AEGISConfig())
        system.add_stream("test")
        
        for t, y in enumerate(signal):
            system.observe("test", y, t)
        
        diag = system.get_diagnostics("test")
        assert diag["group_weights"]["reversion"] > 0.2
    
    def test_dominant_model_seasonal(self, seasonal_signal):
        """SeasonalDummy should dominate on seasonal data."""
        signal = seasonal_signal(n=500, period=7)
        system = AEGIS(AEGISConfig(seasonal_periods=[7]))
        system.add_stream("test")
        
        for t, y in enumerate(signal):
            system.observe("test", y, t)
        
        diag = system.get_diagnostics("test")
        assert diag["group_weights"]["periodic"] > 0.2
    
    def test_prediction_interval_coverage(self, ar1_signal):
        """95% prediction intervals should cover ~95% of observations."""
        signal = ar1_signal(n=1000, phi=0.9)
        system = AEGIS(AEGISConfig())
        system.add_stream("test")
        
        covered = 0
        total = 0
        
        for t, y in enumerate(signal):
            if t > 100:  # Burn-in
                pred = system.predict("test", horizon=1)
                lower, upper = pred.interval(0.95)
                if lower <= y <= upper:
                    covered += 1
                total += 1
            
            system.observe("test", y, t)
        
        coverage = covered / total
        assert 0.90 <= coverage <= 0.99  # Allow some deviation


class TestMultiStreamIntegration:
    """Integration tests for multi-stream processing."""
    
    def test_cross_stream_improves_prediction(self, lead_lag_signal):
        """Cross-stream regression should improve prediction for lagged series."""
        leader, follower = lead_lag_signal(n=500, lag=3)
        
        system = AEGIS(AEGISConfig(cross_stream_lags=5))
        system.add_stream("leader")
        system.add_stream("follower")
        
        errors_without_cross = []
        errors_with_cross = []
        
        for t in range(len(leader)):
            # Observe leader first
            system.observe("leader", leader[t], t)
            
            if t > 50:
                pred = system.predict("follower", horizon=1)
                errors_with_cross.append((follower[t] - pred.mean)**2)
            
            system.observe("follower", follower[t], t)
            system.end_period()
        
        # Errors should decrease as cross-stream relationship is learned
        early_rmse = np.sqrt(np.mean(errors_with_cross[:100]))
        late_rmse = np.sqrt(np.mean(errors_with_cross[-100:]))
        assert late_rmse < early_rmse * 0.8
    
    def test_correlated_streams(self, correlated_streams_signal):
        """System should capture correlation between streams."""
        y1, y2 = correlated_streams_signal(n=500, correlation=0.9)
        
        system = AEGIS(AEGISConfig())
        system.add_stream("stream1")
        system.add_stream("stream2")
        
        for t in range(len(y1)):
            system.observe("stream1", y1[t], t)
            system.observe("stream2", y2[t], t)
            system.end_period()
        
        # Both streams should have similar model weights
        diag1 = system.get_diagnostics("stream1")
        diag2 = system.get_diagnostics("stream2")
        
        # Group weights should be correlated
        groups = ["persistence", "reversion"]
        for g in groups:
            assert abs(diag1["group_weights"].get(g, 0) - diag2["group_weights"].get(g, 0)) < 0.3
```

---

## 5. Phase 2: Expected Free Energy

### 5.1 Step 1: Epistemic Value for Mean Reversion (Days 36-38)

```python
def test_mean_reversion_epistemic_higher_far_from_mean():
    model = MeanReversionModel()
    
    # Train near mean
    for t in range(100):
        model.update(np.random.normal(0, 0.1), t)
    
    # Near mean
    model.last_y = 0.1
    ev_near = model.epistemic_value()
    
    # Far from mean
    model.last_y = 5.0
    ev_far = model.epistemic_value()
    
    assert ev_far > ev_near

def test_mean_reversion_phi_var_decreases():
    model = MeanReversionModel()
    
    initial_var = model.phi_var
    for t in range(200):
        model.update(np.random.normal(0, 1), t)
    
    assert model.phi_var < initial_var
```

### 5.2 Step 2: Epistemic Value for Threshold AR (Days 39-41)

```python
def test_threshold_ar_epistemic_peaks_near_threshold():
    model = ThresholdARModel(tau=0.0)
    
    # Far below threshold
    model.last_y = -5.0
    ev_below = model.epistemic_value()
    
    # Near threshold
    model.last_y = 0.1
    ev_near = model.epistemic_value()
    
    # Far above
    model.last_y = 5.0
    ev_above = model.epistemic_value()
    
    assert ev_near > ev_below
    assert ev_near > ev_above
```

### 5.3 Step 3: Epistemic Value for Jump Diffusion (Days 42-43)

```python
def test_jump_diffusion_epistemic_decreases_with_data():
    model = JumpDiffusionModel()
    
    ev_initial = model.epistemic_value()
    
    for t in range(100):
        model.update(np.random.normal(0, 1), t)
    
    ev_after = model.epistemic_value()
    
    assert ev_after < ev_initial
```

### 5.4 Step 4: EFE Combiner Integration (Days 44-46)

```python
def test_efe_combiner_alpha_zero_matches_phase1():
    """With alpha=0, EFE combiner should match Phase 1 behaviour."""
    config_phase1 = AEGISConfig(use_epistemic_value=False)
    config_phase2 = AEGISConfig(use_epistemic_value=True, epistemic_weight=0.0)
    
    combiner1 = EFEModelCombiner(n_models=3, config=config_phase1)
    combiner2 = EFEModelCombiner(n_models=3, config=config_phase2)
    
    models = [MeanReversionModel() for _ in range(3)]
    
    for _ in range(50):
        y = np.random.normal(0, 1)
        combiner1.update(models, y)
        combiner2.update(models, y)
    
    np.testing.assert_array_almost_equal(
        combiner1.get_weights(),
        combiner2.get_weights(),
        decimal=5
    )

def test_efe_combiner_epistemic_affects_weights():
    """With alpha>0, models with high epistemic value should get boosted."""
    config = AEGISConfig(use_epistemic_value=True, epistemic_weight=2.0)
    combiner = EFEModelCombiner(n_models=2, config=config)
    
    class HighEpistemicModel:
        def log_likelihood(self, y): return -1.0
        def epistemic_value(self): return 1.0
        def update(self, y, t): pass
    
    class LowEpistemicModel:
        def log_likelihood(self, y): return -1.0
        def epistemic_value(self): return 0.0
        def update(self, y, t): pass
    
    models = [HighEpistemicModel(), LowEpistemicModel()]
    
    for _ in range(20):
        combiner.update(models, 0.0)
    
    weights = combiner.get_weights()
    assert weights[0] > weights[1]
```

---

## 6. Phase 2 Integration Tests

```python
class TestPhase2Integration:
    """Integration tests comparing Phase 1 and Phase 2 behaviour."""
    
    def test_faster_adaptation_after_regime_change(self, regime_switching_signal):
        """Phase 2 should adapt faster after regime changes."""
        signal = regime_switching_signal(n=300, break_point=150, mean1=0.0, mean2=5.0)
        
        config1 = AEGISConfig(use_epistemic_value=False)
        config2 = AEGISConfig(use_epistemic_value=True, epistemic_weight=1.0)
        
        system1 = AEGIS(config1)
        system2 = AEGIS(config2)
        
        system1.add_stream("test")
        system2.add_stream("test")
        
        errors1_post_break = []
        errors2_post_break = []
        
        for t, y in enumerate(signal):
            if 150 < t < 200:  # First 50 obs after break
                pred1 = system1.predict("test", horizon=1)
                pred2 = system2.predict("test", horizon=1)
                errors1_post_break.append(abs(y - pred1.mean))
                errors2_post_break.append(abs(y - pred2.mean))
            
            system1.observe("test", y, t)
            system2.observe("test", y, t)
        
        # Phase 2 should have lower errors during adaptation
        mae1 = np.mean(errors1_post_break)
        mae2 = np.mean(errors2_post_break)
        
        assert mae2 < mae1  # Phase 2 adapts faster
    
    def test_no_degradation_on_stationary(self, ar1_signal):
        """Phase 2 should not be worse than Phase 1 on stationary data."""
        signal = ar1_signal(n=500, phi=0.9)
        
        config1 = AEGISConfig(use_epistemic_value=False)
        config2 = AEGISConfig(use_epistemic_value=True, epistemic_weight=1.0)
        
        system1 = AEGIS(config1)
        system2 = AEGIS(config2)
        
        system1.add_stream("test")
        system2.add_stream("test")
        
        errors1 = []
        errors2 = []
        
        for t, y in enumerate(signal):
            if t > 100:
                pred1 = system1.predict("test", horizon=1)
                pred2 = system2.predict("test", horizon=1)
                errors1.append((y - pred1.mean)**2)
                errors2.append((y - pred2.mean)**2)
            
            system1.observe("test", y, t)
            system2.observe("test", y, t)
        
        rmse1 = np.sqrt(np.mean(errors1))
        rmse2 = np.sqrt(np.mean(errors2))
        
        # Phase 2 should be no more than 10% worse
        assert rmse2 <= rmse1 * 1.1
    
    def test_epistemic_values_in_diagnostics(self):
        """Phase 2 diagnostics should include epistemic values."""
        config = AEGISConfig(use_epistemic_value=True)
        system = AEGIS(config)
        system.add_stream("test")
        
        for t in range(50):
            system.observe("test", np.random.normal(0, 1), t)
        
        diag = system.get_diagnostics("test")
        
        assert "epistemic_values" in diag
        assert len(diag["epistemic_values"]) > 0
```

---

## 7. Validation Suite

### 7.1 Signal Taxonomy Tests

```python
@pytest.mark.parametrize("signal_type,expected_group", [
    ("constant", "persistence"),
    ("random_walk", "persistence"),
    ("ar1_0.9", "reversion"),
    ("ar1_0.5", "reversion"),
    ("linear_trend", "trend"),
    ("sine_16", "periodic"),
    ("seasonal_7", "periodic"),
    ("threshold_ar", "reversion"),
])
def test_dominant_group(signal_type, expected_group, signal_generators):
    """Test that correct model group dominates for each signal type."""
    signal = signal_generators[signal_type](n=500)
    
    system = AEGIS(AEGISConfig())
    system.add_stream("test")
    
    for t, y in enumerate(signal):
        system.observe("test", y, t)
    
    diag = system.get_diagnostics("test")
    
    # Expected group should have substantial weight
    assert diag["group_weights"].get(expected_group, 0) > 0.15


@pytest.mark.parametrize("signal_type,target_coverage", [
    ("ar1_0.9", 0.95),
    ("random_walk", 0.95),
    ("threshold_ar", 0.95),
])
def test_interval_coverage(signal_type, target_coverage, signal_generators):
    """Test that prediction intervals achieve target coverage."""
    signal = signal_generators[signal_type](n=1000)
    
    system = AEGIS(AEGISConfig(target_coverage=target_coverage))
    system.add_stream("test")
    
    covered = 0
    total = 0
    
    for t, y in enumerate(signal):
        if t > 200:
            pred = system.predict("test", horizon=1)
            lower, upper = pred.interval(target_coverage)
            if lower <= y <= upper:
                covered += 1
            total += 1
        
        system.observe("test", y, t)
    
    actual_coverage = covered / total
    
    # Should be within 5% of target
    assert abs(actual_coverage - target_coverage) < 0.05
```

---

## 8. Timeline and Milestones

### 8.1 Phase 1 Timeline (35 working days)

| Week | Focus | Deliverables |
|------|-------|--------------|
| 1 | Setup, Base Infrastructure | Project structure, fixtures, Prediction/Config |
| 2 | Persistence + Trend Models | RandomWalk, LocalLevel, LocalTrend, DampedTrend |
| 3 | Mean Reversion Models | MeanReversion, AsymmetricMR, ThresholdAR |
| 4 | Periodic + Dynamic Models | OscillatorBank, SeasonalDummy, AR2, MA1 |
| 5 | Special + Variance Models | JumpDiffusion, ChangePoint, VolatilityTracker |
| 6 | Combiner + Scale Manager | EFEModelCombiner, ScaleManager |
| 7 | Stream Manager + Integration | StreamManager, Cross-stream, Full integration tests |

**Phase 1 Milestone**: Working system, all Phase 1 tests passing, benchmark suite established.

### 8.2 Phase 2 Timeline (15 working days)

| Week | Focus | Deliverables |
|------|-------|--------------|
| 8-9 | Epistemic Value | MeanReversion, ThresholdAR, JumpDiffusion epistemic |
| 10 | FEP-Native Models | MixtureOfExperts, HierarchicalMR |
| 11 | Integration + Validation | Comparative tests, hyperparameter analysis |

**Phase 2 Milestone**: EFE weighting functional, demonstrated improvement on regime-change signals, documentation complete.

### 8.3 Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Computational overhead | Profile early, optimise hot paths |
| Epistemic weighting hurts stationary performance | Track both, allow α tuning |
| Integration complexity | Strict TDD, component isolation |
| Scope creep | Firm phase boundaries, defer extensions |

---

*End of Appendix C*
