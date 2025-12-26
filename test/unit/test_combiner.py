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
        """Simpler models should get bonus when equally accurate.

        RandomWalkModel has 2 parameters, MeanReversionModel has 4.
        For a true random walk signal, both make similar predictions
        but the complexity penalty should favor RandomWalk.
        """
        config = AEGISConfig(complexity_penalty_weight=0.1)
        combiner = EFEModelCombiner(n_models=2, config=config)

        # Use models with different complexity
        models = [RandomWalkModel(), MeanReversionModel()]

        # Feed random walk (cumulative sum) data - both should predict well
        rng = np.random.default_rng(42)
        value = 0.0
        for t in range(200):
            value += rng.normal(0, 1)
            combiner.update(models, value, t)

        weights = combiner.get_weights()
        # RandomWalk (simpler, 2 params) should get higher weight than
        # MeanReversion (4 params) due to complexity penalty
        assert weights[0] > weights[1], (
            f"Simpler model should have higher weight: RW={weights[0]:.3f}, MR={weights[1]:.3f}"
        )

    def test_combiner_complexity_penalty_default_zero(self) -> None:
        """Default complexity penalty weight should be 0 (backward compatible)."""
        config = AEGISConfig()
        assert config.complexity_penalty_weight == 0.0

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
