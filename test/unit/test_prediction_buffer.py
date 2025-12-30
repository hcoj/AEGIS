"""Unit tests for PredictionBuffer class."""


from aegis.core.prediction_buffer import PredictionBuffer


class TestPredictionBuffer:
    """Tests for PredictionBuffer class."""

    def test_prediction_buffer_stores_and_retrieves(self) -> None:
        """Buffer should store and retrieve predictions correctly."""
        buffer = PredictionBuffer(horizons=[1, 4, 16], n_models=3)
        buffer.store(model_idx=0, horizon=4, t=10, mean=5.0, variance=1.0)

        pred = buffer.get(model_idx=0, horizon=4, t=10)
        assert pred is not None
        assert pred[0] == 5.0  # mean
        assert pred[1] == 1.0  # variance

    def test_prediction_buffer_returns_none_for_missing(self) -> None:
        """Buffer should return None for missing entries."""
        buffer = PredictionBuffer(horizons=[1, 4, 16], n_models=3)
        assert buffer.get(model_idx=0, horizon=4, t=10) is None

    def test_prediction_buffer_multiple_models(self) -> None:
        """Buffer should store predictions for multiple models independently."""
        buffer = PredictionBuffer(horizons=[1, 4, 16], n_models=3)

        buffer.store(model_idx=0, horizon=1, t=5, mean=1.0, variance=0.5)
        buffer.store(model_idx=1, horizon=1, t=5, mean=2.0, variance=0.6)
        buffer.store(model_idx=2, horizon=1, t=5, mean=3.0, variance=0.7)

        pred0 = buffer.get(model_idx=0, horizon=1, t=5)
        pred1 = buffer.get(model_idx=1, horizon=1, t=5)
        pred2 = buffer.get(model_idx=2, horizon=1, t=5)

        assert pred0[0] == 1.0
        assert pred1[0] == 2.0
        assert pred2[0] == 3.0

    def test_prediction_buffer_multiple_horizons(self) -> None:
        """Buffer should store predictions for multiple horizons independently."""
        buffer = PredictionBuffer(horizons=[1, 4, 16], n_models=2)

        buffer.store(model_idx=0, horizon=1, t=10, mean=10.0, variance=1.0)
        buffer.store(model_idx=0, horizon=4, t=10, mean=10.5, variance=2.0)
        buffer.store(model_idx=0, horizon=16, t=10, mean=11.0, variance=4.0)

        pred_h1 = buffer.get(model_idx=0, horizon=1, t=10)
        pred_h4 = buffer.get(model_idx=0, horizon=4, t=10)
        pred_h16 = buffer.get(model_idx=0, horizon=16, t=10)

        assert pred_h1[0] == 10.0
        assert pred_h4[0] == 10.5
        assert pred_h16[0] == 11.0

    def test_prediction_buffer_circular_eviction(self) -> None:
        """Buffer should evict old entries when exceeding max_history."""
        buffer = PredictionBuffer(horizons=[1, 4, 16], n_models=2, max_history=20)

        for t in range(30):
            buffer.store(model_idx=0, horizon=1, t=t, mean=float(t), variance=1.0)

        # Old entries should be evicted
        assert buffer.get(model_idx=0, horizon=1, t=5) is None
        # Recent entries should still exist
        assert buffer.get(model_idx=0, horizon=1, t=25) is not None
        assert buffer.get(model_idx=0, horizon=1, t=29) is not None

    def test_prediction_buffer_get_predictions_for_scoring(self) -> None:
        """Buffer should retrieve all horizon predictions for a target time."""
        buffer = PredictionBuffer(horizons=[1, 4, 16], n_models=2)

        # At t=0, model 0 makes predictions for future times
        buffer.store(model_idx=0, horizon=1, t=0, mean=1.0, variance=0.1)  # predicts t=1
        buffer.store(model_idx=0, horizon=4, t=0, mean=4.0, variance=0.4)  # predicts t=4
        buffer.store(model_idx=0, horizon=16, t=0, mean=16.0, variance=1.6)  # predicts t=16

        # At t=1, we want h=1 prediction made at t=0
        pred_h1 = buffer.get_for_scoring(model_idx=0, horizon=1, current_t=1)
        assert pred_h1 is not None
        assert pred_h1[0] == 1.0

        # At t=4, we want h=4 prediction made at t=0
        pred_h4 = buffer.get_for_scoring(model_idx=0, horizon=4, current_t=4)
        assert pred_h4 is not None
        assert pred_h4[0] == 4.0

        # At t=16, we want h=16 prediction made at t=0
        pred_h16 = buffer.get_for_scoring(model_idx=0, horizon=16, current_t=16)
        assert pred_h16 is not None
        assert pred_h16[0] == 16.0

    def test_prediction_buffer_store_all_horizons(self) -> None:
        """Buffer should support storing all horizon predictions at once."""
        buffer = PredictionBuffer(horizons=[1, 4, 16], n_models=2)

        predictions = {
            1: (1.0, 0.1),
            4: (4.0, 0.4),
            16: (16.0, 1.6),
        }
        buffer.store_all(model_idx=0, t=10, predictions=predictions)

        assert buffer.get(model_idx=0, horizon=1, t=10)[0] == 1.0
        assert buffer.get(model_idx=0, horizon=4, t=10)[0] == 4.0
        assert buffer.get(model_idx=0, horizon=16, t=10)[0] == 16.0
