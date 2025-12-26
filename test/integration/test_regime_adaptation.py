"""Integration tests for regime adaptation in AEGIS."""

import numpy as np

from aegis.config import AEGISConfig
from aegis.system import AEGIS


class TestRegimeAdaptation:
    """Integration tests for regime change handling."""

    def test_adapts_to_mean_shift(self) -> None:
        """Test system adapts after mean shift."""
        config = AEGISConfig(break_threshold=2.0, post_break_duration=20)
        aegis = AEGIS(config=config)
        aegis.add_stream("test")

        rng = np.random.default_rng(42)

        for t in range(100):
            aegis.predict("test", horizon=1)
            aegis.observe("test", rng.normal(0, 1))
            aegis.end_period()

        for t in range(100):
            aegis.predict("test", horizon=1)
            aegis.observe("test", rng.normal(10, 1))
            aegis.end_period()

        pred = aegis.predict("test", horizon=1)
        assert pred.mean > 5.0

    def test_adapts_to_volatility_shift(self) -> None:
        """Test system adapts after volatility shift."""
        aegis = AEGIS()
        aegis.add_stream("test")

        rng = np.random.default_rng(42)

        for t in range(100):
            aegis.predict("test", horizon=1)
            aegis.observe("test", rng.normal(0, 0.1))
            aegis.end_period()

        for t in range(100):
            aegis.predict("test", horizon=1)
            aegis.observe("test", rng.normal(0, 5.0))
            aegis.end_period()

        diag = aegis.get_diagnostics("test")
        assert diag["volatility"] > 1.0

    def test_adapts_to_trend_change(self) -> None:
        """Test system adapts after trend change."""
        aegis = AEGIS()
        aegis.add_stream("test")

        for t in range(100):
            aegis.predict("test", horizon=1)
            aegis.observe("test", float(t) * 0.5)
            aegis.end_period()

        for t in range(100, 200):
            aegis.predict("test", horizon=1)
            aegis.observe("test", 50.0 + float(t - 100) * (-0.3))
            aegis.end_period()

        pred = aegis.predict("test", horizon=1)
        assert pred.mean < 50.0

    def test_regime_switching_signal(self, regime_switching_signal) -> None:
        """Test adaptation on regime-switching signal."""
        config = AEGISConfig(break_threshold=2.5)
        aegis = AEGIS(config=config)
        aegis.add_stream("test")

        signal = regime_switching_signal(n=300)

        errors = []
        for t, y in enumerate(signal):
            if t > 50:
                pred = aegis.predict("test", horizon=1)
                errors.append((y - pred.mean) ** 2)
            aegis.observe("test", y)
            aegis.end_period()

        rmse = np.sqrt(np.mean(errors))
        assert rmse < 10.0

    def test_threshold_ar_signal(self, threshold_ar_signal) -> None:
        """Test handling of threshold AR signal."""
        aegis = AEGIS()
        aegis.add_stream("test")

        signal = threshold_ar_signal(n=300)

        for t, y in enumerate(signal):
            if t > 0:
                aegis.predict("test", horizon=1)
            aegis.observe("test", y)
            aegis.end_period()

        diag = aegis.get_diagnostics("test")
        weights = diag["group_weights"]

        assert weights.get("reversion", 0) > 0.01

    def test_jump_diffusion_signal(self, jump_diffusion_signal) -> None:
        """Test handling of jump diffusion signal."""
        config = AEGISConfig(break_threshold=3.0)
        aegis = AEGIS(config=config)
        aegis.add_stream("test")

        signal = jump_diffusion_signal(n=300, jump_prob=0.02, jump_size=5.0)

        for t, y in enumerate(signal):
            if t > 0:
                aegis.predict("test", horizon=1)
            aegis.observe("test", y)
            aegis.end_period()

        pred = aegis.predict("test", horizon=1)
        assert np.isfinite(pred.mean)
        assert pred.variance > 0

    def test_break_detection_triggers_reset(self) -> None:
        """Test break detection triggers model reset."""
        config = AEGISConfig(break_threshold=2.0)
        aegis = AEGIS(config=config)
        aegis.add_stream("test")

        rng = np.random.default_rng(42)

        for t in range(100):
            aegis.predict("test", horizon=1)
            aegis.observe("test", rng.normal(0, 1))
            aegis.end_period()

        weights_before = aegis.streams["test"].scale_manager.scale_weights.copy()

        for t in range(50):
            aegis.predict("test", horizon=1)
            aegis.observe("test", rng.normal(50, 1))
            aegis.end_period()

        weights_after = aegis.streams["test"].scale_manager.scale_weights

        assert not np.allclose(weights_before, weights_after, atol=0.1)
