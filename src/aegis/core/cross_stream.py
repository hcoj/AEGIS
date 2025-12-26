"""Cross-stream regression for AEGIS.

Captures relationships between multiple streams via residual regression.
"""

import numpy as np

from aegis.config import AEGISConfig


class CrossStreamRegression:
    """Captures relationships between multiple streams via residual regression.

    Operates on prediction residuals to avoid double-counting temporal structure.

    Attributes:
        stream_names: List of stream identifiers
        n_streams: Number of streams
        lags: Number of lagged features
        include_lag_zero: Whether to include contemporaneous effects
    """

    def __init__(
        self,
        stream_names: list[str],
        config: AEGISConfig,
    ) -> None:
        """Initialize CrossStreamRegression.

        Args:
            stream_names: List of stream identifiers
            config: AEGIS configuration
        """
        self.stream_names: list[str] = stream_names
        self.n_streams: int = len(stream_names)
        self.config: AEGISConfig = config
        self.lags: int = config.cross_stream_lags
        self.include_lag_zero: bool = config.include_lag_zero

        n_lags = self.lags + (1 if self.include_lag_zero else 0)
        self.n_features: int = (self.n_streams - 1) * n_lags

        self.coefficients: dict[str, np.ndarray] = {
            name: np.zeros(self.n_features) for name in stream_names
        }

        self.P: dict[str, np.ndarray] = {
            name: np.eye(self.n_features) * 100.0 for name in stream_names
        }

        self.residual_history: dict[str, list[float]] = {name: [] for name in stream_names}

        self.current_residuals: dict[str, float | None] = {name: None for name in stream_names}

    def update(
        self,
        stream_name: str,
        residual: float,
        observed_order: list[str],
    ) -> float:
        """Update regression and return adjusted prediction.

        Args:
            stream_name: Name of stream being updated
            residual: Prediction residual (observed - predicted)
            observed_order: Order in which streams were observed this period

        Returns:
            Cross-stream adjustment to apply to prediction
        """
        self.current_residuals[stream_name] = residual

        features = self._build_features(stream_name, observed_order)

        pred = 0.0
        if features is not None and len(self.residual_history[stream_name]) > self.lags:
            beta = self.coefficients[stream_name]
            P = self.P[stream_name]

            pred = np.dot(features, beta)
            error = residual - pred

            forget = self.config.cross_stream_forget
            Pf = P @ features
            denom = forget + np.dot(features, Pf)
            gain = Pf / denom

            self.coefficients[stream_name] = beta + gain * error
            self.P[stream_name] = (P - np.outer(gain, Pf)) / forget

        self.residual_history[stream_name].append(residual)
        if len(self.residual_history[stream_name]) > self.lags + 10:
            self.residual_history[stream_name] = self.residual_history[stream_name][
                -(self.lags + 10) :
            ]

        return pred

    def _build_features(
        self,
        target: str,
        observed_order: list[str],
    ) -> np.ndarray | None:
        """Build feature vector for prediction.

        Args:
            target: Target stream name
            observed_order: Order streams were observed

        Returns:
            Feature array or None if insufficient data
        """
        features: list[float] = []

        for source in self.stream_names:
            if source == target:
                continue

            hist = self.residual_history[source]

            for lag in range(1, self.lags + 1):
                if len(hist) >= lag:
                    features.append(hist[-lag])
                else:
                    features.append(0.0)

            if self.include_lag_zero:
                if source in observed_order:
                    source_idx = observed_order.index(source)
                    target_idx = (
                        observed_order.index(target)
                        if target in observed_order
                        else len(observed_order)
                    )

                    if source_idx < target_idx and self.current_residuals[source] is not None:
                        features.append(self.current_residuals[source])
                    else:
                        features.append(0.0)
                else:
                    features.append(0.0)

        if len(features) == 0:
            return None

        return np.array(features)

    def end_period(self) -> None:
        """Reset current residuals at end of observation period."""
        for name in self.stream_names:
            self.current_residuals[name] = None

    def get_diagnostics(self) -> dict:
        """Get diagnostic information.

        Returns:
            Dictionary with coefficient values
        """
        return {"coefficients": {name: coef.copy() for name, coef in self.coefficients.items()}}
