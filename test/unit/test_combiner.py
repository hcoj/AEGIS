"""Unit tests for EFEModelCombiner."""

import numpy as np

from aegis.config import AEGISConfig
from aegis.core.combiner import EFEModelCombiner
from aegis.core.prediction import Prediction
from aegis.models.persistence import LocalLevelModel, RandomWalkModel
from aegis.models.reversion import MeanReversionModel
from aegis.models.trend import LocalTrendModel


class TestEFEModelCombiner:
    """Tests for EFEModelCombiner."""

    def test_combiner_initial_weights_uniform(self) -> None:
        """Test initial weights are uniform."""
        config = AEGISConfig()
        combiner = EFEModelCombiner(n_models=3, config=config)

        weights = combiner.get_weights()
        assert len(weights) == 3
        assert np.allclose(weights, [1 / 3, 1 / 3, 1 / 3])

    def test_combiner_weights_sum_to_one(self) -> None:
        """Test weights always sum to one."""
        config = AEGISConfig()
        combiner = EFEModelCombiner(n_models=4, config=config)

        models = [RandomWalkModel(), LocalLevelModel(), LocalTrendModel(), RandomWalkModel()]
        for t in range(50):
            y = float(t)
            combiner.update(models, y, t)

        weights = combiner.get_weights()
        assert np.isclose(np.sum(weights), 1.0)

    def test_combiner_better_model_gets_higher_weight(self) -> None:
        """Test model with better predictions gets higher weight."""
        config = AEGISConfig()
        combiner = EFEModelCombiner(n_models=2, config=config)

        models = [RandomWalkModel(), LocalTrendModel()]

        rng = np.random.default_rng(42)
        for t in range(200):
            y = float(t) + rng.normal(0, 0.5)
            combiner.update(models, y, t)

        weights = combiner.get_weights()
        assert weights[1] > weights[0]

    def test_combiner_softmax_temperature(self) -> None:
        """Test temperature affects weight concentration."""
        config_hot = AEGISConfig(temperature=2.0)
        config_cold = AEGISConfig(temperature=0.5)

        combiner_hot = EFEModelCombiner(n_models=2, config=config_hot)
        combiner_cold = EFEModelCombiner(n_models=2, config=config_cold)

        models = [RandomWalkModel(), LocalTrendModel()]

        for t in range(100):
            y = float(t)
            combiner_hot.update(models, y, t)

        models2 = [RandomWalkModel(), LocalTrendModel()]
        for t in range(100):
            y = float(t)
            combiner_cold.update(models2, y, t)

        weights_hot = combiner_hot.get_weights()
        weights_cold = combiner_cold.get_weights()

        concentration_hot = np.max(weights_hot)
        concentration_cold = np.max(weights_cold)

        assert concentration_cold > concentration_hot

    def test_combiner_forgetting_factor(self) -> None:
        """Test forgetting factor allows adaptation."""
        config = AEGISConfig(likelihood_forget=0.9)
        combiner = EFEModelCombiner(n_models=2, config=config)

        models = [RandomWalkModel(), LocalTrendModel()]

        for t in range(100):
            combiner.update(models, float(t), t)

        for t in range(100, 200):
            combiner.update(models, 100.0, t)

        weights = combiner.get_weights()
        assert weights[0] > weights[1]

    def test_combiner_combine_predictions(self) -> None:
        """Test prediction combination using law of total variance."""
        config = AEGISConfig()
        combiner = EFEModelCombiner(n_models=2, config=config)

        models = [RandomWalkModel(), LocalLevelModel()]
        for t in range(10):
            combiner.update(models, float(t), t)

        predictions = [m.predict(horizon=1) for m in models]
        combined = combiner.combine_predictions(predictions)

        assert isinstance(combined, Prediction)
        assert np.isfinite(combined.mean)
        assert combined.variance > 0

    def test_combiner_combined_variance_includes_disagreement(self) -> None:
        """Test combined variance includes model disagreement."""
        config = AEGISConfig()
        combiner = EFEModelCombiner(n_models=2, config=config)

        models = [RandomWalkModel(), LocalTrendModel()]
        for t in range(100):
            combiner.update(models, float(t), t)

        predictions = [m.predict(horizon=1) for m in models]
        combined = combiner.combine_predictions(predictions)

        weighted_avg_var = np.average(
            [p.variance for p in predictions], weights=combiner.get_weights()
        )
        assert combined.variance >= weighted_avg_var

    def test_combiner_epistemic_value_phase2(self) -> None:
        """Test epistemic value is included in Phase 2."""
        config = AEGISConfig(use_epistemic_value=True, epistemic_weight=1.0)
        combiner = EFEModelCombiner(n_models=2, config=config)

        models = [RandomWalkModel(), LocalLevelModel()]
        for t in range(10):
            combiner.update(models, float(t), t)

        assert combiner.last_epistemic is not None

    def test_combiner_reset_scores(self) -> None:
        """Test resetting cumulative scores."""
        config = AEGISConfig()
        combiner = EFEModelCombiner(n_models=2, config=config)

        models = [RandomWalkModel(), LocalTrendModel()]
        for t in range(100):
            combiner.update(models, float(t), t)

        combiner.reset_scores(partial=1.0)
        weights = combiner.get_weights()
        assert np.allclose(weights, [0.5, 0.5])

    def test_combiner_diagnostics(self) -> None:
        """Test diagnostic information."""
        config = AEGISConfig()
        combiner = EFEModelCombiner(n_models=2, config=config)

        models = [RandomWalkModel(), LocalLevelModel()]
        combiner.update(models, 1.0, 0)

        assert len(combiner.last_pragmatic) == 2
        assert combiner.cumulative_scores is not None

    def test_combiner_complexity_penalty_favors_simpler_models(self) -> None:
        """Simpler models should get relative bonus from complexity penalty.

        RandomWalkModel has 2 parameters, MeanReversionModel has 4.
        With complexity penalty enabled, the simpler model should have
        a higher cumulative score advantage compared to no penalty.

        Note: With likelihood normalization, we verify the mechanism works
        by comparing score differences with and without penalty.
        """
        config_with = AEGISConfig(complexity_penalty_weight=5.0)
        config_without = AEGISConfig(complexity_penalty_weight=0.0)

        combiner_with = EFEModelCombiner(n_models=2, config=config_with)
        combiner_without = EFEModelCombiner(n_models=2, config=config_without)

        # Use models with different complexity
        models_with = [RandomWalkModel(), MeanReversionModel()]
        models_without = [RandomWalkModel(), MeanReversionModel()]

        # Feed random walk (cumulative sum) data - both should predict well
        rng = np.random.default_rng(42)
        value = 0.0
        for t in range(200):
            value += rng.normal(0, 1)
            combiner_with.update(models_with, value, t)
            combiner_without.update(models_without, value, t)

        # With complexity penalty, the score gap should favor the simpler model more
        scores_with = combiner_with.cumulative_scores
        scores_without = combiner_without.cumulative_scores

        # Score difference (RW - MR) should be more positive with penalty
        diff_with = scores_with[0] - scores_with[1]
        diff_without = scores_without[0] - scores_without[1]

        assert diff_with > diff_without, (
            f"Complexity penalty should favor simpler model: "
            f"diff_with={diff_with:.3f}, diff_without={diff_without:.3f}"
        )

    def test_combiner_complexity_penalty_default_enabled(self) -> None:
        """Default complexity penalty weight is 0.5 for balanced model selection."""
        config = AEGISConfig()
        assert config.complexity_penalty_weight == 0.5

    def test_combiner_complexity_penalty_affects_cumulative_scores(self) -> None:
        """Complexity penalty should affect cumulative scores."""
        config_no_penalty = AEGISConfig(complexity_penalty_weight=0.0)
        config_with_penalty = AEGISConfig(complexity_penalty_weight=1.0)

        combiner_no = EFEModelCombiner(n_models=2, config=config_no_penalty)
        combiner_with = EFEModelCombiner(n_models=2, config=config_with_penalty)

        # Use models with different complexity on same data
        models_no = [RandomWalkModel(), MeanReversionModel()]
        models_with = [RandomWalkModel(), MeanReversionModel()]

        # Feed random walk data where both models are competitive
        rng = np.random.default_rng(123)
        value = 0.0
        for t in range(50):
            value += rng.normal(0, 1)
            combiner_no.update(models_no, value, t)
            combiner_with.update(models_with, value, t)

        scores_no = combiner_no.cumulative_scores.copy()
        scores_with = combiner_with.cumulative_scores.copy()

        # With penalty, simpler model (RandomWalk) should have relatively
        # higher score compared to MeanReversion
        # Score difference: (score[0] - score[1]) should be higher with penalty
        diff_no = scores_no[0] - scores_no[1]
        diff_with = scores_with[0] - scores_with[1]

        assert diff_with > diff_no, (
            f"Complexity penalty should favor simpler model: "
            f"diff_no={diff_no:.3f}, diff_with={diff_with:.3f}"
        )

    def test_combiner_high_temperature_promotes_diversity(self) -> None:
        """Higher temperature gives more uniform (diverse) weights.

        This helps prevent winner-take-all behavior in ensemble weighting.
        Use temperature >= 1.5 for more diverse ensembles.
        """
        config_low = AEGISConfig(temperature=0.5)
        config_default = AEGISConfig(temperature=1.0)
        config_high = AEGISConfig(temperature=2.0)

        combiner_low = EFEModelCombiner(n_models=3, config=config_low)
        combiner_default = EFEModelCombiner(n_models=3, config=config_default)
        combiner_high = EFEModelCombiner(n_models=3, config=config_high)

        models_low = [RandomWalkModel(), LocalLevelModel(), LocalTrendModel()]
        models_default = [RandomWalkModel(), LocalLevelModel(), LocalTrendModel()]
        models_high = [RandomWalkModel(), LocalLevelModel(), LocalTrendModel()]

        # Feed trending data that will favor trend model
        for t in range(100):
            y = float(t) * 0.5
            combiner_low.update(models_low, y, t)
            combiner_default.update(models_default, y, t)
            combiner_high.update(models_high, y, t)

        weights_low = combiner_low.get_weights()
        weights_default = combiner_default.get_weights()
        weights_high = combiner_high.get_weights()

        # Measure concentration using max weight
        # Higher temperature should give lower max weight (more distributed)
        max_low = np.max(weights_low)
        max_default = np.max(weights_default)
        max_high = np.max(weights_high)

        assert max_high < max_default < max_low, (
            f"Higher temperature should reduce concentration: "
            f"max_low={max_low:.3f}, max_default={max_default:.3f}, max_high={max_high:.3f}"
        )

    def test_combiner_entropy_penalty_encourages_concentration(self) -> None:
        """Entropy penalty should encourage weight concentration.

        When weights are spread evenly (high entropy), the penalty
        should sharpen the distribution to concentrate on fewer models.
        """
        config_no_penalty = AEGISConfig(entropy_penalty_weight=0.0)
        config_with_penalty = AEGISConfig(entropy_penalty_weight=0.5)

        combiner_no = EFEModelCombiner(n_models=4, config=config_no_penalty)
        combiner_with = EFEModelCombiner(n_models=4, config=config_with_penalty)

        # Create models with similar performance (leads to spread-out weights)
        models_no = [RandomWalkModel(), LocalLevelModel(), LocalTrendModel(), MeanReversionModel()]
        models_with = [
            RandomWalkModel(),
            LocalLevelModel(),
            LocalTrendModel(),
            MeanReversionModel(),
        ]

        # Feed data where models have similar performance
        rng = np.random.default_rng(42)
        for t in range(100):
            y = rng.normal(0, 1)
            combiner_no.update(models_no, y, t)
            combiner_with.update(models_with, y, t)

        weights_no = combiner_no.get_weights()
        weights_with = combiner_with.get_weights()

        # Measure concentration using max weight
        max_no = np.max(weights_no)
        max_with = np.max(weights_with)

        # Entropy penalty should increase concentration
        assert max_with > max_no, (
            f"Entropy penalty should increase concentration: "
            f"max_no={max_no:.3f}, max_with={max_with:.3f}"
        )

    def test_combiner_entropy_penalty_default_zero(self) -> None:
        """Default entropy penalty weight should be 0 (backward compatible)."""
        config = AEGISConfig()
        assert config.entropy_penalty_weight == 0.0

    def test_combiner_adaptive_forgetting_starts_at_base(self) -> None:
        """Adaptive forgetting should start at base rate."""
        config = AEGISConfig(likelihood_forget=0.99, use_adaptive_forgetting=True)
        combiner = EFEModelCombiner(n_models=2, config=config)

        assert combiner.current_forget == 0.99

    def test_combiner_adaptive_forgetting_reduces_on_surprise(self) -> None:
        """Forgetting rate should decrease when errors are surprisingly large.

        Large errors indicate the model needs to adapt faster, so we
        should reduce the forgetting factor (forget more of the past).
        """
        config = AEGISConfig(likelihood_forget=0.99, use_adaptive_forgetting=True, min_forget=0.8)
        combiner = EFEModelCombiner(n_models=2, config=config)

        models = [RandomWalkModel(), LocalLevelModel()]

        # Feed normal data first
        for t in range(50):
            combiner.update(models, float(t), t)

        initial_forget = combiner.current_forget

        # Feed surprising data (big jump)
        for t in range(50, 60):
            combiner.update(models, 1000.0, t)

        # Forgetting should have decreased
        assert combiner.current_forget < initial_forget

    def test_combiner_adaptive_forgetting_recovers_after_stability(self) -> None:
        """Forgetting rate should recover to base after stable period."""
        config = AEGISConfig(likelihood_forget=0.99, use_adaptive_forgetting=True, min_forget=0.8)
        combiner = EFEModelCombiner(n_models=2, config=config)

        models = [RandomWalkModel(), LocalLevelModel()]

        # Create surprise with a regime shift
        for t in range(100):
            combiner.update(models, 100.0 if t < 50 else 0.0, t)

        # Capture forgetting after surprise (should be reduced)
        post_surprise_forget = combiner.current_forget
        assert post_surprise_forget < config.likelihood_forget, "Should have reduced after surprise"

        # Very long stable period - models have adapted
        for t in range(100, 500):
            combiner.update(models, 0.0, t)

        # Should have recovered toward base rate
        assert combiner.current_forget > post_surprise_forget, (
            f"Should have recovered: was {post_surprise_forget:.4f}, "
            f"now {combiner.current_forget:.4f}"
        )

    def test_combiner_adaptive_forgetting_default_off(self) -> None:
        """Adaptive forgetting should be off by default."""
        config = AEGISConfig()
        assert not config.use_adaptive_forgetting


class TestMultiHorizonLikelihood:
    """Tests for multi-horizon likelihood scoring."""

    def test_compute_multi_horizon_likelihood_basic(self) -> None:
        """Score observation against past predictions at multiple horizons."""
        from aegis.core.combiner import compute_multi_horizon_likelihood

        # Past predictions: horizon -> (mean, variance)
        predictions = {
            1: (5.0, 1.0),  # h=1: predicted mean=5, var=1
            4: (4.5, 2.0),  # h=4: predicted mean=4.5, var=2
            16: (4.0, 4.0),  # h=16: predicted mean=4, var=4
        }

        # Observation is 5.0 - close to h=1 prediction
        ll = compute_multi_horizon_likelihood(y=5.0, predictions=predictions)

        # Should return a valid log-likelihood
        assert ll is not None
        assert isinstance(ll, float)
        assert not np.isnan(ll)
        assert not np.isinf(ll)

    def test_compute_multi_horizon_likelihood_perfect_predictions(self) -> None:
        """Perfect predictions should give high likelihood."""
        from aegis.core.combiner import compute_multi_horizon_likelihood

        # All predictions are exactly right
        predictions = {
            1: (5.0, 1.0),
            4: (5.0, 1.0),
            16: (5.0, 1.0),
        }

        ll = compute_multi_horizon_likelihood(y=5.0, predictions=predictions)

        # Perfect predictions should have high likelihood
        assert ll > -1.0  # Should be close to maximum

    def test_compute_multi_horizon_likelihood_missing_horizons(self) -> None:
        """Should handle missing horizon predictions gracefully."""
        from aegis.core.combiner import compute_multi_horizon_likelihood

        # Only h=1 available (cold start scenario)
        predictions = {
            1: (5.0, 1.0),
        }

        ll = compute_multi_horizon_likelihood(y=5.0, predictions=predictions)
        assert ll is not None
        assert not np.isnan(ll)

    def test_compute_multi_horizon_likelihood_empty_predictions(self) -> None:
        """Should handle empty predictions gracefully."""
        from aegis.core.combiner import compute_multi_horizon_likelihood

        ll = compute_multi_horizon_likelihood(y=5.0, predictions={})
        # With no predictions, should return 0 or handle gracefully
        assert ll == 0.0 or ll is None

    def test_combiner_stores_predictions(self) -> None:
        """Combiner should store predictions for future multi-horizon scoring."""
        config = AEGISConfig()
        combiner = EFEModelCombiner(n_models=2, config=config)
        models = [RandomWalkModel(), MeanReversionModel()]

        # After update, predictions should be stored
        combiner.update(models, y=10.0, t=0)

        # Check predictions were stored for horizons [1, 4, 16]
        assert combiner.prediction_buffer is not None
        pred_h1 = combiner.prediction_buffer.get(model_idx=0, horizon=1, t=0)
        assert pred_h1 is not None

    def test_combiner_uses_multi_horizon_scoring(self) -> None:
        """After sufficient history, combiner should score on multiple horizons."""
        config = AEGISConfig()
        combiner = EFEModelCombiner(n_models=2, config=config)
        models = [RandomWalkModel(), MeanReversionModel()]

        # Feed 20 observations to build history
        for t in range(20):
            combiner.update(models, y=float(t), t=t)

        # At t=20, should have multi-horizon scoring available
        breakdown = combiner.get_score_breakdown()
        assert "multi_horizon_available" in breakdown
        assert breakdown["multi_horizon_available"] is True
