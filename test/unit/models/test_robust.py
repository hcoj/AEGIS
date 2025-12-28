"""Unit tests for robust estimation."""

import numpy as np
import pytest

from aegis.config import AEGISConfig
from aegis.models.persistence import LocalLevelModel, RandomWalkModel
from aegis.models.robust import robust_weight


class TestRobustWeight:
    """Tests for robust_weight function."""

    def test_normal_observation_returns_one(self) -> None:
        """Errors within threshold return weight 1.0."""
        weight = robust_weight(error=2.0, sigma=1.0, threshold=5.0)
        assert weight == 1.0

    def test_at_threshold_returns_one(self) -> None:
        """Error exactly at threshold returns 1.0."""
        weight = robust_weight(error=5.0, sigma=1.0, threshold=5.0)
        assert weight == 1.0

    def test_outlier_diminished(self) -> None:
        """Error beyond threshold returns weight < 1.0."""
        weight = robust_weight(error=10.0, sigma=1.0, threshold=5.0)
        assert 0 < weight < 1.0
        assert weight == pytest.approx(0.25, rel=0.01)  # (5/10)^2

    def test_large_outlier_bounded(self) -> None:
        """Even large outliers have bounded (non-zero) weight."""
        weight = robust_weight(error=100.0, sigma=1.0, threshold=5.0)
        assert weight > 0.001

    def test_symmetric(self) -> None:
        """Positive and negative errors get same weight."""
        w_pos = robust_weight(error=10.0, sigma=1.0, threshold=5.0)
        w_neg = robust_weight(error=-10.0, sigma=1.0, threshold=5.0)
        assert w_pos == w_neg

    def test_scales_with_sigma(self) -> None:
        """Threshold is in units of sigma."""
        weight = robust_weight(error=10.0, sigma=2.0, threshold=5.0)
        assert weight == 1.0  # |10|/2 = 5 = threshold

    def test_zero_sigma_protected(self) -> None:
        """Zero sigma doesn't cause division by zero."""
        weight = robust_weight(error=10.0, sigma=0.0, threshold=5.0)
        assert np.isfinite(weight)


class TestRandomWalkRobust:
    """Tests for robust estimation in RandomWalkModel."""

    def test_variance_stable_with_outlier(self) -> None:
        """Variance should not explode from single large outlier."""
        config = AEGISConfig(use_robust_estimation=True, outlier_threshold=5.0)
        model = RandomWalkModel(config=config)

        # Warm up with normal observations
        for i in range(50):
            model.update(float(i % 2), t=i)

        variance_before = model.sigma_sq
        model.update(100.0, t=50)  # Large outlier
        variance_after = model.sigma_sq

        # Variance should increase but not by 10000x
        assert variance_after < 10 * variance_before

    def test_without_robust_explodes(self) -> None:
        """Without robust estimation, variance explodes from outlier."""
        model = RandomWalkModel()  # Default: no robust

        for i in range(50):
            model.update(float(i % 2), t=i)

        variance_before = model.sigma_sq
        model.update(100.0, t=50)
        variance_after = model.sigma_sq

        # Without robust, variance should increase dramatically
        assert variance_after > 50 * variance_before

    def test_normal_data_unchanged(self) -> None:
        """Robust estimation should not affect normal data."""
        config_robust = AEGISConfig(use_robust_estimation=True, outlier_threshold=5.0)
        model_robust = RandomWalkModel(config=config_robust)
        model_normal = RandomWalkModel()

        # Feed same normal data
        data = [1.0, 1.1, 0.9, 1.0, 1.05, 0.95]
        for i, y in enumerate(data):
            model_robust.update(y, t=i)
            model_normal.update(y, t=i)

        # Variances should be similar (within 1%)
        assert model_robust.sigma_sq == pytest.approx(model_normal.sigma_sq, rel=0.01)


class TestLocalLevelRobust:
    """Tests for robust estimation in LocalLevelModel."""

    def test_variance_stable_with_outlier(self) -> None:
        """Variance should not explode from single large outlier."""
        config = AEGISConfig(use_robust_estimation=True, outlier_threshold=5.0)
        model = LocalLevelModel(config=config)

        for i in range(50):
            model.update(float(i % 2), t=i)

        variance_before = model.sigma_sq
        model.update(100.0, t=50)  # Large outlier
        variance_after = model.sigma_sq

        # Variance should increase but not by 10000x
        assert variance_after < 10 * variance_before

    def test_without_robust_explodes(self) -> None:
        """Without robust estimation, variance explodes from outlier."""
        model = LocalLevelModel()  # Default: no robust

        for i in range(50):
            model.update(float(i % 2), t=i)

        variance_before = model.sigma_sq
        model.update(100.0, t=50)
        variance_after = model.sigma_sq

        # Without robust, variance should increase dramatically
        assert variance_after > 50 * variance_before
