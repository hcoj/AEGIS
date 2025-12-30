"""Prediction buffer for multi-horizon model scoring.

Stores predictions made at each time step for later evaluation
against realized observations.
"""

from collections import defaultdict


class PredictionBuffer:
    """Buffer to store predictions for multi-horizon scoring.

    Stores (mean, variance) tuples keyed by (model_idx, horizon, t).
    Implements circular eviction to bound memory usage.

    Attributes:
        horizons: List of horizons to track (e.g., [1, 4, 16])
        n_models: Number of models being tracked
        max_history: Maximum number of time steps to retain
    """

    def __init__(
        self,
        horizons: list[int],
        n_models: int,
        max_history: int = 32,
    ) -> None:
        """Initialize PredictionBuffer.

        Args:
            horizons: List of horizons to track
            n_models: Number of models to track
            max_history: Maximum time steps to retain (older entries evicted)
        """
        self.horizons = horizons
        self.n_models = n_models
        self.max_history = max_history

        # Storage: {(model_idx, horizon, t): (mean, variance)}
        self._storage: dict[tuple[int, int, int], tuple[float, float]] = {}

        # Track oldest time step per (model_idx, horizon) for eviction
        self._oldest_t: dict[tuple[int, int], int] = defaultdict(lambda: 0)

    def store(
        self,
        model_idx: int,
        horizon: int,
        t: int,
        mean: float,
        variance: float,
    ) -> None:
        """Store a prediction.

        Args:
            model_idx: Index of the model
            horizon: Prediction horizon
            t: Time step when prediction was made
            mean: Predicted mean
            variance: Predicted variance
        """
        key = (model_idx, horizon, t)
        self._storage[key] = (mean, variance)

        # Evict old entries if needed
        self._evict_if_needed(model_idx, horizon, t)

    def store_all(
        self,
        model_idx: int,
        t: int,
        predictions: dict[int, tuple[float, float]],
    ) -> None:
        """Store predictions for all horizons at once.

        Args:
            model_idx: Index of the model
            t: Time step when predictions were made
            predictions: Dict mapping horizon -> (mean, variance)
        """
        for horizon, (mean, variance) in predictions.items():
            self.store(model_idx, horizon, t, mean, variance)

    def get(
        self,
        model_idx: int,
        horizon: int,
        t: int,
    ) -> tuple[float, float] | None:
        """Retrieve a stored prediction.

        Args:
            model_idx: Index of the model
            horizon: Prediction horizon
            t: Time step when prediction was made

        Returns:
            Tuple of (mean, variance) or None if not found
        """
        key = (model_idx, horizon, t)
        return self._storage.get(key)

    def get_for_scoring(
        self,
        model_idx: int,
        horizon: int,
        current_t: int,
    ) -> tuple[float, float] | None:
        """Get prediction made `horizon` steps ago for scoring.

        At time `current_t`, retrieves the prediction made at time
        `current_t - horizon` with the given horizon.

        Args:
            model_idx: Index of the model
            horizon: Prediction horizon
            current_t: Current time step (when observation arrived)

        Returns:
            Tuple of (mean, variance) or None if not found
        """
        prediction_time = current_t - horizon
        if prediction_time < 0:
            return None
        return self.get(model_idx, horizon, prediction_time)

    def _evict_if_needed(self, model_idx: int, horizon: int, current_t: int) -> None:
        """Evict old entries to maintain max_history limit.

        Args:
            model_idx: Model index being updated
            horizon: Horizon being updated
            current_t: Current time step
        """
        key_prefix = (model_idx, horizon)
        oldest = self._oldest_t[key_prefix]

        # Evict entries older than (current_t - max_history)
        cutoff = current_t - self.max_history
        while oldest <= cutoff:
            old_key = (model_idx, horizon, oldest)
            if old_key in self._storage:
                del self._storage[old_key]
            oldest += 1

        self._oldest_t[key_prefix] = oldest

    def clear(self) -> None:
        """Clear all stored predictions."""
        self._storage.clear()
        self._oldest_t.clear()
