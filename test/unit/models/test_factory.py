"""Unit tests for model factory."""


from aegis.config import AEGISConfig
from aegis.models import create_model_bank
from aegis.models.base import TemporalModel


class TestModelFactory:
    """Tests for create_model_bank factory."""

    def test_factory_returns_list(self) -> None:
        """Test factory returns a list of models."""
        config = AEGISConfig()
        models = create_model_bank(config)

        assert isinstance(models, list)
        assert len(models) > 0

    def test_factory_returns_temporal_models(self) -> None:
        """Test all returned models are TemporalModels."""
        config = AEGISConfig()
        models = create_model_bank(config)

        for model in models:
            assert isinstance(model, TemporalModel)

    def test_factory_includes_persistence_models(self) -> None:
        """Test factory includes persistence models."""
        config = AEGISConfig()
        models = create_model_bank(config)

        groups = [m.group for m in models]
        assert "persistence" in groups

    def test_factory_includes_trend_models(self) -> None:
        """Test factory includes trend models."""
        config = AEGISConfig()
        models = create_model_bank(config)

        groups = [m.group for m in models]
        assert "trend" in groups

    def test_factory_includes_reversion_models(self) -> None:
        """Test factory includes reversion models."""
        config = AEGISConfig()
        models = create_model_bank(config)

        groups = [m.group for m in models]
        assert "reversion" in groups

    def test_factory_includes_periodic_models(self) -> None:
        """Test factory includes periodic models."""
        config = AEGISConfig()
        models = create_model_bank(config)

        groups = [m.group for m in models]
        assert "periodic" in groups

    def test_factory_includes_dynamic_models(self) -> None:
        """Test factory includes dynamic models."""
        config = AEGISConfig()
        models = create_model_bank(config)

        groups = [m.group for m in models]
        assert "dynamic" in groups

    def test_factory_includes_variance_models(self) -> None:
        """Test factory includes variance models."""
        config = AEGISConfig()
        models = create_model_bank(config)

        groups = [m.group for m in models]
        assert "variance" in groups

    def test_factory_unique_model_names(self) -> None:
        """Test all models have unique names."""
        config = AEGISConfig()
        models = create_model_bank(config)

        names = [m.name for m in models]
        assert len(names) == len(set(names))

    def test_factory_respects_oscillator_periods(self) -> None:
        """Test factory uses config oscillator periods."""
        config = AEGISConfig(oscillator_periods=[10, 20])
        models = create_model_bank(config)

        oscillator_models = [m for m in models if "oscillator" in m.name.lower()]
        assert len(oscillator_models) > 0

    def test_factory_respects_seasonal_periods(self) -> None:
        """Test factory uses config seasonal periods."""
        config = AEGISConfig(seasonal_periods=[7, 30])
        models = create_model_bank(config)

        seasonal_models = [m for m in models if "seasonal" in m.name.lower()]
        assert len(seasonal_models) > 0

    def test_factory_models_can_update(self) -> None:
        """Test all models can update."""
        config = AEGISConfig()
        models = create_model_bank(config)

        for model in models:
            model.update(1.0, t=0)
            model.update(2.0, t=1)

    def test_factory_models_can_predict(self) -> None:
        """Test all models can predict."""
        config = AEGISConfig()
        models = create_model_bank(config)

        for model in models:
            model.update(1.0, t=0)
            pred = model.predict(horizon=1)
            assert pred.variance > 0
