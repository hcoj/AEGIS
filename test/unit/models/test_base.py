"""Unit tests for TemporalModel abstract base class."""

import numpy as np
import pytest

from aegis.core.prediction import Prediction
from aegis.models.base import TemporalModel


class TestTemporalModelABC:
    """Tests for TemporalModel abstract base class."""

    def test_temporal_model_is_abstract(self) -> None:
        """Test that TemporalModel cannot be instantiated directly."""
        with pytest.raises(TypeError):
            TemporalModel()

    def test_temporal_model_concrete_implementation(self) -> None:
        """Test that a concrete implementation can be created."""

        class DummyModel(TemporalModel):
            def update(self, y: float, t: int) -> None:
                pass

            def predict(self, horizon: int) -> Prediction:
                return Prediction(mean=0.0, variance=1.0)

            def log_likelihood(self, y: float) -> float:
                return -0.5 * np.log(2 * np.pi) - 0.5 * y**2

            def reset(self, partial: float = 1.0) -> None:
                pass

        model = DummyModel()
        assert model is not None

    def test_temporal_model_default_name(self) -> None:
        """Test default name property returns class name."""

        class MyCustomModel(TemporalModel):
            def update(self, y: float, t: int) -> None:
                pass

            def predict(self, horizon: int) -> Prediction:
                return Prediction(mean=0.0, variance=1.0)

            def log_likelihood(self, y: float) -> float:
                return 0.0

            def reset(self, partial: float = 1.0) -> None:
                pass

        model = MyCustomModel()
        assert model.name == "MyCustomModel"

    def test_temporal_model_default_n_parameters(self) -> None:
        """Test default n_parameters is 0."""

        class DummyModel(TemporalModel):
            def update(self, y: float, t: int) -> None:
                pass

            def predict(self, horizon: int) -> Prediction:
                return Prediction(mean=0.0, variance=1.0)

            def log_likelihood(self, y: float) -> float:
                return 0.0

            def reset(self, partial: float = 1.0) -> None:
                pass

        model = DummyModel()
        assert model.n_parameters == 0

    def test_temporal_model_default_group(self) -> None:
        """Test default group is 'unknown'."""

        class DummyModel(TemporalModel):
            def update(self, y: float, t: int) -> None:
                pass

            def predict(self, horizon: int) -> Prediction:
                return Prediction(mean=0.0, variance=1.0)

            def log_likelihood(self, y: float) -> float:
                return 0.0

            def reset(self, partial: float = 1.0) -> None:
                pass

        model = DummyModel()
        assert model.group == "unknown"

    def test_temporal_model_default_epistemic_value(self) -> None:
        """Test default epistemic_value returns 0.0."""

        class DummyModel(TemporalModel):
            def update(self, y: float, t: int) -> None:
                pass

            def predict(self, horizon: int) -> Prediction:
                return Prediction(mean=0.0, variance=1.0)

            def log_likelihood(self, y: float) -> float:
                return 0.0

            def reset(self, partial: float = 1.0) -> None:
                pass

        model = DummyModel()
        assert model.epistemic_value() == 0.0

    def test_temporal_model_pragmatic_value(self) -> None:
        """Test pragmatic_value uses Gaussian entropy formula."""

        class DummyModel(TemporalModel):
            def update(self, y: float, t: int) -> None:
                pass

            def predict(self, horizon: int) -> Prediction:
                return Prediction(mean=0.0, variance=1.0)

            def log_likelihood(self, y: float) -> float:
                return 0.0

            def reset(self, partial: float = 1.0) -> None:
                pass

        model = DummyModel()
        pv = model.pragmatic_value()
        expected = -0.5 * np.log(2 * np.pi * 1.0) - 0.5
        assert pv == pytest.approx(expected, abs=0.01)

    def test_temporal_model_pragmatic_value_scales_with_variance(self) -> None:
        """Test pragmatic_value decreases with higher variance."""

        class LowVarModel(TemporalModel):
            def update(self, y: float, t: int) -> None:
                pass

            def predict(self, horizon: int) -> Prediction:
                return Prediction(mean=0.0, variance=1.0)

            def log_likelihood(self, y: float) -> float:
                return 0.0

            def reset(self, partial: float = 1.0) -> None:
                pass

        class HighVarModel(TemporalModel):
            def update(self, y: float, t: int) -> None:
                pass

            def predict(self, horizon: int) -> Prediction:
                return Prediction(mean=0.0, variance=10.0)

            def log_likelihood(self, y: float) -> float:
                return 0.0

            def reset(self, partial: float = 1.0) -> None:
                pass

        low_var = LowVarModel()
        high_var = HighVarModel()

        assert low_var.pragmatic_value() > high_var.pragmatic_value()

    def test_temporal_model_custom_group(self) -> None:
        """Test that group can be overridden."""

        class PersistenceModel(TemporalModel):
            def update(self, y: float, t: int) -> None:
                pass

            def predict(self, horizon: int) -> Prediction:
                return Prediction(mean=0.0, variance=1.0)

            def log_likelihood(self, y: float) -> float:
                return 0.0

            def reset(self, partial: float = 1.0) -> None:
                pass

            @property
            def group(self) -> str:
                return "persistence"

        model = PersistenceModel()
        assert model.group == "persistence"

    def test_temporal_model_custom_n_parameters(self) -> None:
        """Test that n_parameters can be overridden."""

        class TwoParamModel(TemporalModel):
            def update(self, y: float, t: int) -> None:
                pass

            def predict(self, horizon: int) -> Prediction:
                return Prediction(mean=0.0, variance=1.0)

            def log_likelihood(self, y: float) -> float:
                return 0.0

            def reset(self, partial: float = 1.0) -> None:
                pass

            @property
            def n_parameters(self) -> int:
                return 2

        model = TwoParamModel()
        assert model.n_parameters == 2
