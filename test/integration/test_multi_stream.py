"""Integration tests for multi-stream AEGIS."""

import numpy as np

from aegis.config import AEGISConfig
from aegis.system import AEGIS


class TestMultiStreamIntegration:
    """Integration tests for multi-stream predictions."""

    def test_two_streams_independent(self) -> None:
        """Test two independent streams can be processed."""
        aegis = AEGIS()
        aegis.add_stream("A")
        aegis.add_stream("B")

        rng = np.random.default_rng(42)

        for t in range(100):
            aegis.observe("A", rng.normal(0, 1))
            aegis.observe("B", rng.normal(10, 2))
            aegis.end_period()

        pred_a = aegis.predict("A", horizon=1)
        pred_b = aegis.predict("B", horizon=1)

        assert abs(pred_a.mean) < 5
        assert abs(pred_b.mean - 10) < 5

    def test_cross_stream_regression_created(self) -> None:
        """Test cross-stream regression is active for multi-stream."""
        aegis = AEGIS()
        aegis.add_stream("leader")
        aegis.add_stream("follower")

        assert aegis.cross_stream is not None

        rng = np.random.default_rng(42)
        for t in range(50):
            aegis.observe("leader", rng.normal(0, 1))
            aegis.observe("follower", rng.normal(0, 1))
            aegis.end_period()

        diag = aegis.get_diagnostics("follower")
        assert "cross_stream" in diag

    def test_lead_lag_relationship(self, lead_lag_signal) -> None:
        """Test cross-stream learns lead-lag relationship."""
        config = AEGISConfig(cross_stream_lags=3)
        aegis = AEGIS(config=config)
        aegis.add_stream("leader")
        aegis.add_stream("follower")

        leader, follower = lead_lag_signal(n=300, lag=1, sigma=0.5)

        for t in range(len(leader)):
            # Call predict() to enable cross-stream learning via residuals
            if t > 0:
                aegis.predict("leader", horizon=1)
                aegis.predict("follower", horizon=1)
            aegis.observe("leader", leader[t])
            aegis.observe("follower", follower[t])
            aegis.end_period()

        diag = aegis.get_diagnostics("follower")
        cross_coef = diag["cross_stream"]["coefficients"]["follower"]
        assert np.any(np.abs(cross_coef) > 0.01)

    def test_three_streams(self) -> None:
        """Test three stream system works."""
        aegis = AEGIS()
        aegis.add_stream("X")
        aegis.add_stream("Y")
        aegis.add_stream("Z")

        rng = np.random.default_rng(42)

        for t in range(100):
            aegis.observe("X", rng.normal(0, 1))
            aegis.observe("Y", float(t) * 0.1)
            aegis.observe("Z", np.sin(t / 10) * 2)
            aegis.end_period()

        pred_x = aegis.predict("X", horizon=1)
        pred_y = aegis.predict("Y", horizon=1)
        pred_z = aegis.predict("Z", horizon=1)

        assert np.isfinite(pred_x.mean)
        assert np.isfinite(pred_y.mean)
        assert np.isfinite(pred_z.mean)

    def test_observation_order_tracked(self) -> None:
        """Test observation order affects cross-stream."""
        config = AEGISConfig(include_lag_zero=True)
        aegis = AEGIS(config=config)
        aegis.add_stream("A")
        aegis.add_stream("B")

        aegis.observe("A", 1.0)
        assert aegis.observed_this_period == ["A"]

        aegis.observe("B", 2.0)
        assert aegis.observed_this_period == ["A", "B"]

        aegis.end_period()
        assert aegis.observed_this_period == []

    def test_diagnostics_all_streams(self) -> None:
        """Test get_all_diagnostics returns info for all streams."""
        aegis = AEGIS()
        aegis.add_stream("price")
        aegis.add_stream("volume")

        for t in range(50):
            aegis.observe("price", float(t))
            aegis.observe("volume", float(t) * 10)
            aegis.end_period()

        all_diag = aegis.get_all_diagnostics()

        assert "price" in all_diag
        assert "volume" in all_diag
        assert "model_weights" in all_diag["price"]
        assert "model_weights" in all_diag["volume"]

    def test_streams_have_separate_state(self) -> None:
        """Test each stream maintains separate state."""
        aegis = AEGIS()
        aegis.add_stream("high_vol")
        aegis.add_stream("low_vol")

        rng = np.random.default_rng(42)

        for t in range(100):
            aegis.observe("high_vol", rng.normal(0, 10))
            aegis.observe("low_vol", rng.normal(0, 0.1))
            aegis.end_period()

        pred_high = aegis.predict("high_vol", horizon=1)
        pred_low = aegis.predict("low_vol", horizon=1)

        assert pred_high.variance > pred_low.variance * 10
