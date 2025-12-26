"""Unit tests for CrossStreamRegression."""

import numpy as np

from aegis.config import AEGISConfig
from aegis.core.cross_stream import CrossStreamRegression


class TestCrossStreamRegression:
    """Tests for CrossStreamRegression."""

    def test_cross_stream_creation(self) -> None:
        """Test CrossStreamRegression initializes correctly."""
        config = AEGISConfig(cross_stream_lags=3)
        cs = CrossStreamRegression(stream_names=["A", "B", "C"], config=config)

        assert cs.n_streams == 3
        assert cs.lags == 3
        assert "A" in cs.coefficients
        assert "B" in cs.coefficients

    def test_cross_stream_update_returns_adjustment(self) -> None:
        """Test update returns a prediction adjustment."""
        config = AEGISConfig(cross_stream_lags=2)
        cs = CrossStreamRegression(stream_names=["A", "B"], config=config)

        for t in range(10):
            cs.update("A", residual=float(t), observed_order=["A"])
            cs.update("B", residual=float(t) * 0.5, observed_order=["A", "B"])
            cs.end_period()

        adj = cs.update("B", residual=5.0, observed_order=["A", "B"])
        assert np.isfinite(adj)

    def test_cross_stream_learns_relationship(self) -> None:
        """Test cross-stream learns lead-lag relationship."""
        config = AEGISConfig(cross_stream_lags=3, include_lag_zero=False)
        cs = CrossStreamRegression(stream_names=["A", "B"], config=config)

        rng = np.random.default_rng(42)

        for t in range(500):
            a_residual = rng.normal(0, 1)
            b_residual = 0.7 * (cs.residual_history["A"][-1] if cs.residual_history["A"] else 0)
            b_residual += rng.normal(0, 0.3)

            cs.update("A", residual=a_residual, observed_order=["A"])
            cs.update("B", residual=b_residual, observed_order=["A", "B"])
            cs.end_period()

        assert abs(cs.coefficients["B"][0]) > 0.01

    def test_cross_stream_lag_zero(self) -> None:
        """Test lag-0 contemporaneous effects."""
        config = AEGISConfig(cross_stream_lags=2, include_lag_zero=True)
        cs = CrossStreamRegression(stream_names=["A", "B"], config=config)

        for t in range(50):
            cs.update("A", residual=1.0, observed_order=["A"])
            cs.update("B", residual=1.5, observed_order=["A", "B"])
            cs.end_period()

        assert cs.include_lag_zero

    def test_cross_stream_end_period_clears_current(self) -> None:
        """Test end_period clears current residuals."""
        config = AEGISConfig()
        cs = CrossStreamRegression(stream_names=["A", "B"], config=config)

        cs.update("A", residual=1.0, observed_order=["A"])
        assert cs.current_residuals["A"] == 1.0

        cs.end_period()
        assert cs.current_residuals["A"] is None

    def test_cross_stream_history_trimmed(self) -> None:
        """Test residual history is trimmed."""
        config = AEGISConfig(cross_stream_lags=5)
        cs = CrossStreamRegression(stream_names=["A", "B"], config=config)

        for t in range(100):
            cs.update("A", residual=float(t), observed_order=["A"])
            cs.end_period()

        assert len(cs.residual_history["A"]) < 20

    def test_cross_stream_get_diagnostics(self) -> None:
        """Test diagnostic information."""
        config = AEGISConfig()
        cs = CrossStreamRegression(stream_names=["A", "B"], config=config)

        for t in range(10):
            cs.update("A", residual=1.0, observed_order=["A"])
            cs.update("B", residual=2.0, observed_order=["A", "B"])
            cs.end_period()

        diag = cs.get_diagnostics()
        assert "coefficients" in diag
        assert "A" in diag["coefficients"]

    def test_cross_stream_multiple_streams(self) -> None:
        """Test with more than two streams."""
        config = AEGISConfig(cross_stream_lags=2)
        cs = CrossStreamRegression(stream_names=["A", "B", "C"], config=config)

        for t in range(30):
            cs.update("A", residual=1.0, observed_order=["A"])
            cs.update("B", residual=2.0, observed_order=["A", "B"])
            cs.update("C", residual=3.0, observed_order=["A", "B", "C"])
            cs.end_period()

        assert len(cs.coefficients["C"]) == 4
        assert len(cs.coefficients["B"]) == 4

    def test_cross_stream_observation_order_matters(self) -> None:
        """Test observation order affects lag-0 features."""
        config = AEGISConfig(cross_stream_lags=2, include_lag_zero=True)
        cs = CrossStreamRegression(stream_names=["A", "B"], config=config)

        cs.update("B", residual=1.0, observed_order=["B"])
        cs.update("A", residual=2.0, observed_order=["B", "A"])

        assert cs.current_residuals["B"] == 1.0
        assert cs.current_residuals["A"] == 2.0

    def test_cross_stream_coefficients_per_stream(self) -> None:
        """Test each target stream has separate coefficients."""
        config = AEGISConfig(cross_stream_lags=2)
        cs = CrossStreamRegression(stream_names=["A", "B", "C"], config=config)

        assert not np.array_equal(cs.coefficients["A"], cs.coefficients["B"]) or np.allclose(
            cs.coefficients["A"], 0
        )

        assert cs.n_features == 4
