"""Unit tests for AEGISConfig dataclass."""

import pytest

from aegis.config import AEGISConfig


class TestAEGISConfig:
    """Tests for AEGISConfig dataclass."""

    def test_config_defaults(self) -> None:
        """Test default configuration values."""
        config = AEGISConfig()
        assert config.use_epistemic_value is False
        assert config.epistemic_weight == 1.0
        assert config.likelihood_forget == 0.99
        assert config.temperature == 1.0
        assert config.scales == [1, 2, 4, 8, 16, 32, 64]

    def test_config_default_scales_length(self) -> None:
        """Test default scales list has 7 elements."""
        config = AEGISConfig()
        assert len(config.scales) == 7

    def test_config_default_oscillator_periods(self) -> None:
        """Test default oscillator periods."""
        config = AEGISConfig()
        assert config.oscillator_periods == [4, 8, 16, 32, 64, 128, 256]

    def test_config_default_seasonal_periods(self) -> None:
        """Test default seasonal periods."""
        config = AEGISConfig()
        assert config.seasonal_periods == [7, 12]

    def test_config_default_cross_stream(self) -> None:
        """Test default cross-stream configuration."""
        config = AEGISConfig()
        assert config.cross_stream_lags == 5
        assert config.include_lag_zero is False
        assert config.n_factors == 3

    def test_config_default_regime_adaptation(self) -> None:
        """Test default regime adaptation configuration."""
        config = AEGISConfig()
        assert config.break_threshold == 3.0
        assert config.post_break_forget == 0.9
        assert config.post_break_duration == 50

    def test_config_default_calibration(self) -> None:
        """Test default calibration configuration."""
        config = AEGISConfig()
        assert config.target_coverage == 0.95
        assert config.use_quantile_calibration is True

    def test_config_default_robustness(self) -> None:
        """Test default robustness configuration."""
        config = AEGISConfig()
        assert config.outlier_threshold == 5.0
        assert config.min_variance == 1e-10
        assert config.max_variance == 1e10

    def test_config_default_use_robust_estimation(self) -> None:
        """Test default use_robust_estimation is False for backward compatibility."""
        config = AEGISConfig()
        assert config.use_robust_estimation is False

    def test_config_enable_robust_estimation(self) -> None:
        """Test robust estimation can be enabled."""
        config = AEGISConfig(use_robust_estimation=True)
        assert config.use_robust_estimation is True

    def test_config_phase2(self) -> None:
        """Test Phase 2 configuration."""
        config = AEGISConfig(use_epistemic_value=True, epistemic_weight=2.0)
        assert config.use_epistemic_value is True
        assert config.epistemic_weight == 2.0

    def test_config_custom_scales(self) -> None:
        """Test custom scales configuration."""
        config = AEGISConfig(scales=[1, 4, 16])
        assert config.scales == [1, 4, 16]

    def test_config_validation_likelihood_forget_too_high(self) -> None:
        """Test validation rejects likelihood_forget > 1."""
        config = AEGISConfig(likelihood_forget=1.5)
        with pytest.raises(AssertionError):
            config.validate()

    def test_config_validation_likelihood_forget_negative(self) -> None:
        """Test validation rejects negative likelihood_forget."""
        config = AEGISConfig(likelihood_forget=-0.1)
        with pytest.raises(AssertionError):
            config.validate()

    def test_config_validation_temperature_negative(self) -> None:
        """Test validation rejects negative temperature."""
        config = AEGISConfig(temperature=-1.0)
        with pytest.raises(AssertionError):
            config.validate()

    def test_config_validation_temperature_zero(self) -> None:
        """Test validation rejects zero temperature."""
        config = AEGISConfig(temperature=0.0)
        with pytest.raises(AssertionError):
            config.validate()

    def test_config_validation_epistemic_weight_negative(self) -> None:
        """Test validation rejects negative epistemic_weight."""
        config = AEGISConfig(epistemic_weight=-0.5)
        with pytest.raises(AssertionError):
            config.validate()

    def test_config_validation_break_threshold_negative(self) -> None:
        """Test validation rejects negative break_threshold."""
        config = AEGISConfig(break_threshold=-1.0)
        with pytest.raises(AssertionError):
            config.validate()

    def test_config_validation_target_coverage_out_of_range(self) -> None:
        """Test validation rejects target_coverage outside (0, 1)."""
        config = AEGISConfig(target_coverage=1.5)
        with pytest.raises(AssertionError):
            config.validate()

    def test_config_validation_passes(self) -> None:
        """Test validation passes for valid config."""
        config = AEGISConfig()
        config.validate()

    def test_config_validation_edge_cases(self) -> None:
        """Test validation passes for edge case values."""
        config = AEGISConfig(
            likelihood_forget=0.0,
            temperature=0.01,
            epistemic_weight=0.0,
        )
        config.validate()

    def test_config_scales_not_shared(self) -> None:
        """Test that default scales list is not shared between instances."""
        config1 = AEGISConfig()
        config2 = AEGISConfig()
        config1.scales.append(128)
        assert 128 not in config2.scales

    def test_config_temperature_for_diversity(self) -> None:
        """Higher temperature promotes ensemble diversity.

        temperature=1.0 is the default for standard softmax.
        temperature=1.5-2.0 gives more uniform weights, reducing winner-take-all.
        temperature<1.0 makes weights more concentrated (winner-take-all).
        """
        # Default temperature is 1.0
        config = AEGISConfig()
        assert config.temperature == 1.0

        # Higher temperature is valid
        config_high = AEGISConfig(temperature=2.0)
        assert config_high.temperature == 2.0
        config_high.validate()  # Should pass

        # Lower temperature is valid
        config_low = AEGISConfig(temperature=0.5)
        assert config_low.temperature == 0.5
        config_low.validate()  # Should pass
