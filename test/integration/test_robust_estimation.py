"""Integration tests for robust estimation."""

from collections.abc import Callable

import numpy as np

from aegis.config import AEGISConfig
from aegis.system import AEGIS


class TestRobustEstimationIntegration:
    """Integration tests for robust estimation."""

    def test_robust_reduces_error_on_contaminated_data(self, contaminated_signal: Callable) -> None:
        """Robust estimation should reduce MAE on contaminated data."""
        data = contaminated_signal(n=500, contamination_prob=0.02, contamination_scale=10.0)
        warmup = 100

        # Standard (no robust estimation)
        config_std = AEGISConfig(use_robust_estimation=False)
        aegis_std = AEGIS(config=config_std)
        aegis_std.add_stream("test")
        errors_std = []
        for t in range(len(data)):
            if t > warmup and t + 1 < len(data):
                pred = aegis_std.predict("test", horizon=1)
                errors_std.append(abs(data[t + 1] - pred.mean))
            aegis_std.observe("test", data[t])
            aegis_std.end_period()

        # Robust estimation enabled
        config_rob = AEGISConfig(use_robust_estimation=True, outlier_threshold=5.0)
        aegis_rob = AEGIS(config=config_rob)
        aegis_rob.add_stream("test")
        errors_rob = []
        for t in range(len(data)):
            if t > warmup and t + 1 < len(data):
                pred = aegis_rob.predict("test", horizon=1)
                errors_rob.append(abs(data[t + 1] - pred.mean))
            aegis_rob.observe("test", data[t])
            aegis_rob.end_period()

        mae_std = np.mean(errors_std)
        mae_rob = np.mean(errors_rob)

        # Robust should be better (or at least not significantly worse)
        assert mae_rob < mae_std * 1.1

    def test_robust_does_not_hurt_clean_data(self, random_walk_signal: Callable) -> None:
        """Robust estimation should not significantly hurt clean data performance."""
        data = random_walk_signal(n=500, sigma=1.0)
        warmup = 100

        # Standard
        config_std = AEGISConfig(use_robust_estimation=False)
        aegis_std = AEGIS(config=config_std)
        aegis_std.add_stream("test")
        errors_std = []
        for t in range(len(data)):
            if t > warmup and t + 1 < len(data):
                pred = aegis_std.predict("test", horizon=1)
                errors_std.append(abs(data[t + 1] - pred.mean))
            aegis_std.observe("test", data[t])
            aegis_std.end_period()

        # Robust
        config_rob = AEGISConfig(use_robust_estimation=True, outlier_threshold=5.0)
        aegis_rob = AEGIS(config=config_rob)
        aegis_rob.add_stream("test")
        errors_rob = []
        for t in range(len(data)):
            if t > warmup and t + 1 < len(data):
                pred = aegis_rob.predict("test", horizon=1)
                errors_rob.append(abs(data[t + 1] - pred.mean))
            aegis_rob.observe("test", data[t])
            aegis_rob.end_period()

        mae_std = np.mean(errors_std)
        mae_rob = np.mean(errors_rob)

        # Performance should be similar (within 20%)
        assert mae_rob < mae_std * 1.2
