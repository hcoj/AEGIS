"""Main AEGIS system class.

The AEGIS class orchestrates multi-stream time series prediction
using structured model ensembles with EFE weighting.
"""

from collections.abc import Callable

from aegis.config import AEGISConfig
from aegis.core.cross_stream import CrossStreamRegression
from aegis.core.prediction import Prediction
from aegis.core.stream_manager import StreamManager
from aegis.models import create_model_bank
from aegis.models.base import TemporalModel


class AEGIS:
    """Main AEGIS forecasting system.

    Manages multiple data streams with cross-stream integration,
    multi-scale processing, and uncertainty calibration.

    Attributes:
        config: AEGIS configuration
        streams: Dictionary of stream managers by name
        cross_stream: Cross-stream regression (if multi-stream)
    """

    def __init__(
        self,
        config: AEGISConfig | None = None,
        model_factory: Callable[[], list[TemporalModel]] | None = None,
    ) -> None:
        """Initialize AEGIS system.

        Args:
            config: AEGIS configuration (uses defaults if None)
            model_factory: Callable that returns a list of TemporalModels
                           (uses create_model_bank if None)
        """
        self.config: AEGISConfig = config or AEGISConfig()
        self.config.validate()

        if model_factory is None:
            self.model_factory: Callable[[], list[TemporalModel]] = lambda: create_model_bank(
                self.config
            )
        else:
            self.model_factory = model_factory

        self.streams: dict[str, StreamManager] = {}
        self.stream_order: list[str] = []

        self.cross_stream: CrossStreamRegression | None = None

        self.t: int = 0
        self.observed_this_period: list[str] = []

    def add_stream(self, name: str) -> None:
        """Add a new data stream.

        Args:
            name: Stream identifier
        """
        self.streams[name] = StreamManager(
            name=name,
            config=self.config,
            model_factory=self.model_factory,
        )
        self.stream_order.append(name)

        if len(self.streams) > 1:
            self.cross_stream = CrossStreamRegression(
                stream_names=list(self.streams.keys()),
                config=self.config,
            )

    def observe(self, stream_name: str, value: float, t: int | None = None) -> None:
        """Record observation for a stream.

        Args:
            stream_name: Name of the stream
            value: Observed value
            t: Optional time index (managed automatically if not provided)

        Raises:
            ValueError: If stream_name is not known
        """
        if stream_name not in self.streams:
            raise ValueError(f"Unknown stream: {stream_name}")

        if t is not None:
            self.t = t

        self.observed_this_period.append(stream_name)

        stream = self.streams[stream_name]

        if self.cross_stream is not None and stream.last_prediction is not None:
            residual = value - stream.last_prediction.mean
            self.cross_stream.update(
                stream_name,
                residual,
                self.observed_this_period,
            )

        stream.observe(value, self.t)

    def predict(self, stream_name: str, horizon: int = 1) -> Prediction:
        """Generate prediction for a stream.

        Args:
            stream_name: Name of the stream
            horizon: Steps ahead to predict

        Returns:
            Prediction with mean and calibrated uncertainty

        Raises:
            ValueError: If stream_name is not known
        """
        if stream_name not in self.streams:
            raise ValueError(f"Unknown stream: {stream_name}")

        return self.streams[stream_name].predict(horizon)

    def end_period(self) -> None:
        """Signal end of observation period.

        Call after all streams have been observed for the current time step.
        """
        if self.cross_stream is not None:
            self.cross_stream.end_period()

        self.observed_this_period = []
        self.t += 1

    def get_diagnostics(self, stream_name: str) -> dict:
        """Get diagnostic information for a stream.

        Args:
            stream_name: Name of the stream

        Returns:
            Dictionary with model weights, volatility, and other diagnostics

        Raises:
            ValueError: If stream_name is not known
        """
        if stream_name not in self.streams:
            raise ValueError(f"Unknown stream: {stream_name}")

        diag = self.streams[stream_name].get_diagnostics()

        if self.cross_stream is not None:
            diag["cross_stream"] = self.cross_stream.get_diagnostics()

        return diag

    def get_all_diagnostics(self) -> dict:
        """Get diagnostic information for all streams.

        Returns:
            Dictionary mapping stream names to their diagnostics
        """
        return {name: self.get_diagnostics(name) for name in self.streams}
