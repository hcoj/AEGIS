"""Unit tests for EFEModelCombiner."""

import numpy as np

from aegis.config import AEGISConfig
from aegis.core.combiner import EFEModelCombiner
from aegis.core.prediction import Prediction
from aegis.models.persistence import LocalLevelModel, RandomWalkModel
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
