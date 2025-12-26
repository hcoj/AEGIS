"""Unit tests for AEGIS main system."""

import numpy as np
import pytest

from aegis.config import AEGISConfig
from aegis.system import AEGIS


class TestAEGIS:
    """Tests for AEGIS main system."""

    def test_aegis_creation(self) -> None:
        """Test AEGIS initializes with default config."""
        aegis = AEGIS()
        assert aegis.config is not None
        assert len(aegis.streams) == 0

    def test_aegis_custom_config(self) -> None:
        """Test AEGIS uses custom config."""
        config = AEGISConfig(temperature=2.0)
        aegis = AEGIS(config=config)
        assert aegis.config.temperature == 2.0

    def test_aegis_add_stream(self) -> None:
        """Test adding a stream."""
        aegis = AEGIS()
        aegis.add_stream("price")
        assert "price" in aegis.streams
        assert len(aegis.streams) == 1

    def test_aegis_add_multiple_streams(self) -> None:
        """Test adding multiple streams."""
        aegis = AEGIS()
        aegis.add_stream("price")
        aegis.add_stream("volume")
        aegis.add_stream("volatility")
        assert len(aegis.streams) == 3

    def test_aegis_observe_single_stream(self) -> None:
        """Test observing values on a stream."""
        aegis = AEGIS()
        aegis.add_stream("price")

        aegis.observe("price", 100.0)
        aegis.observe("price", 101.0)

        assert aegis.t == 0

    def test_aegis_observe_unknown_stream_raises(self) -> None:
        """Test observing unknown stream raises error."""
        aegis = AEGIS()
        with pytest.raises(ValueError, match="Unknown stream"):
            aegis.observe("unknown", 1.0)

    def test_aegis_predict_single_stream(self) -> None:
        """Test prediction on a stream."""
        aegis = AEGIS()
        aegis.add_stream("price")

        for i in range(20):
            aegis.observe("price", 100.0 + float(i))

        pred = aegis.predict("price", horizon=1)
        assert np.isfinite(pred.mean)
        assert pred.variance > 0

    def test_aegis_predict_unknown_stream_raises(self) -> None:
        """Test predicting unknown stream raises error."""
        aegis = AEGIS()
        with pytest.raises(ValueError, match="Unknown stream"):
            aegis.predict("unknown", horizon=1)

    def test_aegis_end_period(self) -> None:
        """Test end_period increments time."""
        aegis = AEGIS()
        aegis.add_stream("price")

        aegis.observe("price", 100.0)
        aegis.end_period()

        assert aegis.t == 1

    def test_aegis_multi_stream_observe(self) -> None:
        """Test observing multiple streams."""
        aegis = AEGIS()
        aegis.add_stream("A")
        aegis.add_stream("B")

        aegis.observe("A", 1.0)
        aegis.observe("B", 2.0)
        aegis.end_period()

        assert aegis.t == 1

    def test_aegis_cross_stream_created(self) -> None:
        """Test cross-stream regression created for multi-stream."""
        aegis = AEGIS()
        aegis.add_stream("A")
        aegis.add_stream("B")

        assert aegis.cross_stream is not None

    def test_aegis_get_diagnostics(self) -> None:
        """Test diagnostic information."""
        aegis = AEGIS()
        aegis.add_stream("price")

        for i in range(20):
            aegis.observe("price", float(i))

        diag = aegis.get_diagnostics("price")
        assert "model_weights" in diag
        assert "volatility" in diag

    def test_aegis_get_all_diagnostics(self) -> None:
        """Test getting diagnostics for all streams."""
        aegis = AEGIS()
        aegis.add_stream("A")
        aegis.add_stream("B")

        for i in range(10):
            aegis.observe("A", float(i))
            aegis.observe("B", float(i) * 2)

        all_diag = aegis.get_all_diagnostics()
        assert "A" in all_diag
        assert "B" in all_diag

    def test_aegis_prediction_tracks_trend(self) -> None:
        """Test prediction captures upward trend."""
        aegis = AEGIS()
        aegis.add_stream("price")

        for i in range(100):
            aegis.observe("price", float(i))

        pred = aegis.predict("price", horizon=1)
        assert pred.mean > 95.0

    def test_aegis_observation_order_tracked(self) -> None:
        """Test observation order is tracked for lag-0 effects."""
        aegis = AEGIS()
        aegis.add_stream("A")
        aegis.add_stream("B")

        aegis.observe("A", 1.0)
        assert "A" in aegis.observed_this_period

        aegis.observe("B", 2.0)
        assert "B" in aegis.observed_this_period

        aegis.end_period()
        assert len(aegis.observed_this_period) == 0

    def test_aegis_time_tracking(self) -> None:
        """Test time index is tracked."""
        aegis = AEGIS()
        aegis.add_stream("price")

        aegis.observe("price", 1.0, t=10)
        aegis.end_period()

        assert aegis.t == 11

    def test_aegis_single_stream_no_cross_stream(self) -> None:
        """Test single stream has no cross-stream regression."""
        aegis = AEGIS()
        aegis.add_stream("price")
        assert aegis.cross_stream is None
