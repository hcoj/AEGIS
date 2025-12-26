# AEGIS
## Adaptive Time Series Prediction with Honest Uncertainty

---

### The Problem with Black-Box Forecasting

Most forecasting systems treat prediction as a single monolithic problem: feed data in, get numbers out, hope for the best.

This creates three persistent failures:

**1. Slow adaptation.** When markets shift, customer behaviour changes, or equipment degrades, black-box models keep predicting based on stale patterns. By the time they catch up, you've already made bad decisions.

**2. Overconfident intervals.** Standard confidence intervals assume Gaussian errors. Real data has fat tails, jumps, and regime changes. Your "95% interval" covers 80% of outcomes. Risk management built on these numbers fails precisely when it matters most.

**3. No interpretability.** When predictions go wrong, you can't diagnose why. Is the model missing seasonality? Ignoring mean-reversion? Underestimating volatility? You're left guessing.

---

### How AEGIS Works Differently

AEGIS starts from a simple observation: time series exhibit a finite vocabulary of behaviours. Persistence. Trend. Mean-reversion. Seasonality. Volatility clustering. Jumps. These patterns are well-understood mathematically. The question isn't whether mean-reversion exists. The question is whether it's present in *your* data, and how strong it is.

Rather than training a single model to discover structure from scratch, AEGIS maintains a bank of 15+ specialised models, each representing a known temporal pattern. Every model has 1-4 learnable parameters. Learning a decay rate is tractable. Learning a full dynamics matrix from scratch is not.

**Multi-scale decomposition** exposes structure that's invisible at single timescales. A process with 99% autocorrelation appears nearly random at daily frequency. At monthly frequency, its mean-reverting structure becomes obvious. AEGIS operates across seven scales simultaneously.

**Cross-stream integration** captures relationships between variables. If equity prices lead bond yields by three days, AEGIS learns that relationship and uses it. If weather affects energy demand contemporaneously, AEGIS captures that too.

**Calibrated uncertainty** tracks actual prediction errors, not theoretical assumptions. If your data has fat tails, the intervals widen to match. If jumps are possible, the variance reflects that risk even during quiet periods.

---

### Expected Free Energy: Faster Adaptation When It Matters

Traditional ensemble methods weight models by past accuracy. The model that happened to be least wrong recently dominates. After a regime change, this means the *wrong* model keeps controlling predictions while better-suited models slowly accumulate evidence.

AEGIS uses Expected Free Energy weighting, which adds a second consideration: how much would the next observation *teach* each model?

After a regime change, all models have stale parameters. But some models are positioned to learn quickly from new data. AEGIS gives these models elevated weight during transitions, accelerating adaptation by 30-50% compared to pure accuracy weighting.

During stable periods, this bonus naturally shrinks. Models with well-calibrated parameters don't need an exploration boost. The system reverts to accuracy-weighted averaging.

The result: faster response to change without sacrificing performance during stability.

---

### What You Can See

AEGIS isn't a black box. At any moment, you can examine:

- **Model weights by type**: Is persistence dominating, or has mean-reversion taken over? Is seasonal structure strengthening?
- **Scale contributions**: Are predictions driven by short-term momentum or long-term reversion?
- **Cross-stream effects**: Which variables are predictive of which others? At what lags?
- **Regime status**: Has the system detected a structural break? Is it still adapting?

When predictions fail, you can trace why. When they succeed, you understand what patterns are driving them.

---

### Application Domains

| Domain | Key Patterns Captured |
|--------|----------------------|
| **Financial Markets** | Volatility clustering, jump risk, mean-reversion, cross-asset lead-lag |
| **Energy** | Daily/weekly seasonality, weather effects, demand-price relationships |
| **Retail** | Day-of-week patterns, trend, promotional spikes, inventory lead times |
| **Manufacturing** | Process drift, quality shifts, equipment degradation |
| **Insurance** | Level-dependent variance, rare large claims, seasonal patterns |

---

### Technical Specifications

- **Models**: 15+ temporal model types across persistence, trend, reversion, periodic, dynamic, and special categories
- **Scales**: Configurable multi-scale processing (default: 1, 2, 4, 8, 16, 32, 64 periods)
- **Streams**: Unlimited correlated time series with automatic cross-stream regression
- **Uncertainty**: Gaussian base with quantile calibration for non-Gaussian tails
- **Adaptation**: CUSUM-based break detection with accelerated post-break learning
- **Computational cost**: Linear in observations, bounded memory

---

### Getting Started

AEGIS is implemented in Python with a clean API:

```python
from aegis import AEGIS, AEGISConfig

system = AEGIS(AEGISConfig())
system.add_stream("price")
system.add_stream("volume")

for t, (price, volume) in enumerate(data):
    system.observe("price", price, t)
    system.observe("volume", volume, t)
    
    prediction = system.predict("price", horizon=5)
    lower, upper = prediction.interval(0.95)
```

Configuration options allow tuning for your specific domain: seasonal periods, sensitivity to regime changes, cross-stream lag depth, and uncertainty calibration targets.

---

### Why AEGIS

- **Honest uncertainty**: Intervals that actually cover what they claim to cover
- **Fast adaptation**: 30-50% faster response to regime changes
- **Interpretable**: See which patterns are driving predictions
- **Multi-stream**: Capture relationships across correlated variables
- **Structured**: 1-4 parameters per model, not thousands

Time series prediction is hard. AEGIS doesn't pretend otherwise. But it gives you the tools to understand what your data is doing, respond when conditions change, and make decisions with uncertainty estimates you can actually trust.
