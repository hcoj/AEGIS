"""Configuration dataclass for AEGIS system."""

from dataclasses import dataclass, field


@dataclass
class AEGISConfig:
    """Configuration for AEGIS forecasting system.

    Attributes:
        use_epistemic_value: Enable Phase 2 EFE weighting (default False for Phase 1)
        epistemic_weight: Weight on epistemic value in model combination (alpha)
        scales: Multi-scale lookback periods for return computation
        oscillator_periods: Periods for oscillator bank model
        seasonal_periods: Periods for seasonal dummy models
        likelihood_forget: Forgetting factor for cumulative log-likelihood scores
        temperature: Softmax temperature for model weights
        complexity_penalty_weight: Weight on BIC-like complexity penalty (default 0)
        volatility_decay: EWMA decay for volatility tracking
        cross_stream_lags: Number of lags in cross-stream regression
        include_lag_zero: Whether to include contemporaneous terms
        n_factors: Number of factors in factor model
        break_threshold: CUSUM threshold for regime break detection
        post_break_forget: Reduced forgetting factor after break
        post_break_duration: Steps to maintain elevated adaptation
        target_coverage: Target coverage for prediction intervals
        use_quantile_calibration: Enable quantile-based interval calibration
        outlier_threshold: Standard deviations for outlier detection
        min_variance: Minimum variance floor to prevent numerical issues
        max_variance: Maximum variance ceiling to prevent overflow
    """

    use_epistemic_value: bool = False
    epistemic_weight: float = 1.0

    scales: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32, 64])

    oscillator_periods: list[int] = field(default_factory=lambda: [4, 8, 16, 32, 64, 128, 256])

    seasonal_periods: list[int] = field(default_factory=lambda: [7, 12])

    likelihood_forget: float = 0.99
    temperature: float = 1.0
    complexity_penalty_weight: float = 0.0

    volatility_decay: float = 0.94

    cross_stream_lags: int = 5
    include_lag_zero: bool = False
    cross_stream_forget: float = 0.99
    n_factors: int = 3

    break_threshold: float = 3.0
    post_break_forget: float = 0.9
    post_break_duration: int = 50

    target_coverage: float = 0.95
    use_quantile_calibration: bool = True

    outlier_threshold: float = 5.0
    min_variance: float = 1e-10
    max_variance: float = 1e10

    def validate(self) -> None:
        """Validate configuration parameters.

        Raises:
            AssertionError: If any parameter is invalid
        """
        assert 0.0 <= self.likelihood_forget <= 1.0, (
            f"likelihood_forget must be in [0, 1], got {self.likelihood_forget}"
        )

        assert self.temperature > 0, f"temperature must be positive, got {self.temperature}"

        assert self.epistemic_weight >= 0, (
            f"epistemic_weight must be non-negative, got {self.epistemic_weight}"
        )

        assert self.break_threshold >= 0, (
            f"break_threshold must be non-negative, got {self.break_threshold}"
        )

        assert 0.0 < self.target_coverage < 1.0, (
            f"target_coverage must be in (0, 1), got {self.target_coverage}"
        )
