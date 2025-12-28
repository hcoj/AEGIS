"""Robust estimation utilities for AEGIS models."""


def robust_weight(error: float, sigma: float, threshold: float) -> float:
    """Compute Huber-like downweighting factor for outliers.

    Uses bounded influence function: weight = min(1, (threshold*sigma/|error|)^2)
    This gives full weight to normal observations and soft downweighting to outliers.

    Args:
        error: Prediction error (y - y_hat)
        sigma: Current standard deviation estimate
        threshold: Number of standard deviations for outlier cutoff

    Returns:
        Weight in (0, 1] for variance update. Returns 1.0 for normal observations,
        diminishing weight for outliers.
    """
    sigma_safe = max(sigma, 1e-10)
    z = abs(error) / sigma_safe

    if z <= threshold:
        return 1.0

    return (threshold / z) ** 2
