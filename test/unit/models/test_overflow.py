"""Unit tests for overflow protection in models and combiners.

These tests verify that extreme values don't cause NaN or inf in predictions.
"""

import numpy as np
import pytest

from aegis.config import AEGISConfig
from aegis.core.combiner import EFEModelCombiner
from aegis.core.prediction import Prediction
from aegis.core.scale_manager import ScaleManager
from aegis.models.dynamic import AR2Model, MA1Model
from aegis.models.persistence import LocalLevelModel, RandomWalkModel


class TestModelOverflowProtection:
    """Test variance overflow protection in models."""

    def test_ar2_extreme_sigma_sq(self) -> None:
        """AR2Model should not produce inf variance with extreme sigma_sq."""
        model = AR2Model()
        model.sigma_sq = 1e20

        pred = model.predict(horizon=1024)

        assert np.isfinite(pred.variance)
        assert pred.variance <= 1e10

    def test_ar2_extreme_horizon(self) -> None:
        """AR2Model should handle very large horizons."""
        model = AR2Model()
        model.sigma_sq = 1.0

        pred = model.predict(horizon=10000)

        assert np.isfinite(pred.variance)
        assert pred.variance <= 1e10

    def test_ma1_extreme_sigma_sq(self) -> None:
        """MA1Model should not produce inf variance with extreme sigma_sq."""
        model = MA1Model()
        model.sigma_sq = 1e20

        pred = model.predict(horizon=1024)

        assert np.isfinite(pred.variance)
        assert pred.variance <= 1e10

    def test_ma1_extreme_horizon(self) -> None:
        """MA1Model should handle very large horizons."""
        model = MA1Model()
        model.sigma_sq = 1.0

        pred = model.predict(horizon=10000)

        assert np.isfinite(pred.variance)
        assert pred.variance <= 1e10

    def test_random_walk_extreme_sigma_sq(self) -> None:
        """RandomWalkModel should not produce inf variance with extreme sigma_sq."""
        model = RandomWalkModel()
        model.sigma_sq = 1e20

        pred = model.predict(horizon=1024)

        assert np.isfinite(pred.variance)
        assert pred.variance <= 1e10

    def test_random_walk_extreme_horizon(self) -> None:
        """RandomWalkModel should handle very large horizons."""
        model = RandomWalkModel()
        model.sigma_sq = 1.0

        pred = model.predict(horizon=10000)

        assert np.isfinite(pred.variance)
        assert pred.variance <= 1e10

    def test_local_level_extreme_sigma_sq(self) -> None:
        """LocalLevelModel should not produce inf variance with extreme sigma_sq."""
        model = LocalLevelModel()
        model.sigma_sq = 1e20

        pred = model.predict(horizon=1024)

        assert np.isfinite(pred.variance)
        assert pred.variance <= 1e10

    def test_local_level_extreme_horizon(self) -> None:
        """LocalLevelModel should handle very large horizons."""
        model = LocalLevelModel()
        model.sigma_sq = 1.0

        pred = model.predict(horizon=10000)

        assert np.isfinite(pred.variance)
        assert pred.variance <= 1e10

    def test_variance_floor(self) -> None:
        """Models should have minimum variance floor."""
        model = RandomWalkModel()
        model.sigma_sq = 0.0

        pred = model.predict(horizon=1)

        assert pred.variance >= 1e-10


class TestCombinerOverflowProtection:
    """Test overflow protection in prediction combiners."""

    def test_combiner_extreme_means(self) -> None:
        """Combiner should handle extreme mean values without overflow."""
        config = AEGISConfig()
        combiner = EFEModelCombiner(n_models=3, config=config)

        predictions = [
            Prediction(mean=1e8, variance=1.0),
            Prediction(mean=-1e8, variance=1.0),
            Prediction(mean=0.0, variance=1.0),
        ]

        result = combiner.combine_predictions(predictions)

        assert np.isfinite(result.mean)
        assert np.isfinite(result.variance)
        assert result.variance <= 1e10

    def test_combiner_divergent_predictions(self) -> None:
        """Combiner should handle very divergent predictions."""
        config = AEGISConfig()
        combiner = EFEModelCombiner(n_models=2, config=config)

        predictions = [
            Prediction(mean=1e6, variance=1.0),
            Prediction(mean=-1e6, variance=1.0),
        ]

        result = combiner.combine_predictions(predictions)

        assert np.isfinite(result.mean)
        assert np.isfinite(result.variance)
        assert result.variance <= 1e10

    def test_combiner_extreme_variances(self) -> None:
        """Combiner should handle extreme variance values."""
        config = AEGISConfig()
        combiner = EFEModelCombiner(n_models=2, config=config)

        predictions = [
            Prediction(mean=0.0, variance=1e15),
            Prediction(mean=0.0, variance=1e15),
        ]

        result = combiner.combine_predictions(predictions)

        assert np.isfinite(result.variance)
        assert result.variance <= 1e10

    def test_combiner_variance_floor(self) -> None:
        """Combiner should maintain minimum variance."""
        config = AEGISConfig()
        combiner = EFEModelCombiner(n_models=2, config=config)

        predictions = [
            Prediction(mean=0.0, variance=0.0),
            Prediction(mean=0.0, variance=0.0),
        ]

        result = combiner.combine_predictions(predictions)

        assert result.variance >= 1e-10


class TestScaleManagerOverflowProtection:
    """Test overflow protection in scale manager."""

    @pytest.fixture
    def config(self) -> AEGISConfig:
        """Create test config with small scales for fast testing."""
        return AEGISConfig(scales=[1, 2, 4])

    @pytest.fixture
    def scale_manager(self, config: AEGISConfig) -> ScaleManager:
        """Create scale manager with test config."""
        from aegis.models import create_model_bank

        return ScaleManager(config, lambda: create_model_bank(config))

    def test_scale_manager_extreme_observations(self, scale_manager: ScaleManager) -> None:
        """Scale manager should handle extreme observation values."""
        for i in range(10):
            scale_manager.observe(1e6 * (i % 2))

        pred = scale_manager.predict(horizon=64)

        assert np.isfinite(pred.mean)
        assert np.isfinite(pred.variance)
        assert pred.variance <= 1e10

    def test_scale_manager_polynomial_trend(self, scale_manager: ScaleManager) -> None:
        """Scale manager should handle polynomial trend without NaN."""
        for t in range(200):
            y = 0.0001 * t**2 + 0.05 * t
            scale_manager.observe(y)

        pred = scale_manager.predict(horizon=1024)

        assert np.isfinite(pred.mean)
        assert np.isfinite(pred.variance)

    def test_scale_manager_variance_bounds(self, scale_manager: ScaleManager) -> None:
        """Scale manager variance should be within config bounds."""
        for i in range(20):
            scale_manager.observe(float(i))

        pred = scale_manager.predict(horizon=64)

        config = scale_manager.config
        assert pred.variance >= config.min_variance
        assert pred.variance <= config.max_variance
