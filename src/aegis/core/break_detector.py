"""Break detection for AEGIS.

Implements CUSUM-based regime break detection.
"""


class CUSUMBreakDetector:
    """CUSUM-based regime break detection.

    Accumulates standardized prediction errors and triggers when
    cumulative sum exceeds threshold, indicating regime change.

    Attributes:
        threshold: Number of std devs for break signal
        drift: Allowance subtracted each step
        cusum_pos: Positive CUSUM statistic
        cusum_neg: Negative CUSUM statistic
        sigma: Running volatility estimate
    """

    def __init__(
        self,
        threshold: float = 3.0,
        drift: float = 1.5,
        decay: float = 0.95,
    ) -> None:
        """Initialize CUSUMBreakDetector.

        Args:
            threshold: Number of standard deviations for break signal
            drift: Allowance subtracted each step (in std devs)
            decay: EWMA decay for volatility estimation
        """
        self.threshold: float = threshold
        self.drift: float = drift
        self.decay: float = decay

        self.cusum_pos: float = 0.0
        self.cusum_neg: float = 0.0
        self.sigma: float = 1.0
        self._n_obs: int = 0

    def update(self, error: float) -> bool:
        """Update detector with prediction error.

        Args:
            error: Prediction error (observed - predicted)

        Returns:
            True if break detected, False otherwise
        """
        self._n_obs += 1

        self.sigma = self.decay * self.sigma + (1 - self.decay) * abs(error)

        z = error / max(self.sigma, 1e-10)

        self.cusum_pos = max(0.0, self.cusum_pos + z - self.drift)
        self.cusum_neg = max(0.0, self.cusum_neg - z - self.drift)

        if self.cusum_pos > self.threshold or self.cusum_neg > self.threshold:
            return True

        return False

    def reset(self) -> None:
        """Reset detector state."""
        self.cusum_pos = 0.0
        self.cusum_neg = 0.0

    def get_state(self) -> dict:
        """Get current detector state.

        Returns:
            Dictionary with CUSUM values and sigma
        """
        return {
            "cusum_pos": self.cusum_pos,
            "cusum_neg": self.cusum_neg,
            "sigma": self.sigma,
        }
