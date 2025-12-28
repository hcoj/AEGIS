"""Base class for all temporal models in AEGIS."""

from abc import ABC, abstractmethod

import numpy as np

from aegis.core.prediction import Prediction

# Maximum variance to prevent overflow in variance * horizon calculations
MAX_SIGMA_SQ: float = 1e8


class TemporalModel(ABC):
    """Abstract base class for all temporal models.

    All temporal models must implement the core interface:
        - update: Process new observation
        - predict: Generate prediction for horizon
        - log_likelihood: Score observation under predictive distribution
        - reset: Reset parameters toward priors

    The EFE interface has working defaults:
        - pragmatic_value: Expected log-likelihood (from Gaussian formula)
        - epistemic_value: Expected information gain (default 0.0)
    """

    @abstractmethod
    def update(self, y: float, t: int) -> None:
        """Update model state with new observation.

        Args:
            y: Observed value
            t: Time index
        """
        ...

    @abstractmethod
    def predict(self, horizon: int) -> Prediction:
        """Generate prediction for given horizon.

        Args:
            horizon: Steps ahead to predict

        Returns:
            Prediction with mean and variance
        """
        ...

    @abstractmethod
    def log_likelihood(self, y: float) -> float:
        """Compute log-likelihood of observation under current predictive distribution.

        Args:
            y: Observed value

        Returns:
            Log probability density at y
        """
        ...

    @abstractmethod
    def reset(self, partial: float = 1.0) -> None:
        """Reset parameters toward priors.

        Used after regime breaks to accelerate adaptation.

        Args:
            partial: Interpolation weight. 1.0 = full reset, 0.0 = no change
        """
        ...

    def pragmatic_value(self) -> float:
        """Expected log-likelihood under own predictive distribution.

        Default implementation assumes Gaussian predictive distribution.

        Returns:
            Expected log-likelihood (higher is better)
        """
        pred = self.predict(horizon=1)
        return -0.5 * np.log(2 * np.pi * pred.variance) - 0.5

    def epistemic_value(self) -> float:
        """Expected information gain about parameters from next observation.

        Default returns 0.0. Override in models that track parameter uncertainty.

        Returns:
            Expected information gain (higher means more to learn)
        """
        return 0.0

    @property
    def name(self) -> str:
        """Human-readable model name.

        Returns:
            Class name as string
        """
        return self.__class__.__name__

    @property
    def n_parameters(self) -> int:
        """Number of learnable parameters.

        Returns:
            Parameter count (default 0)
        """
        return 0

    @property
    def group(self) -> str:
        """Model group classification.

        Valid groups: persistence, trend, reversion, periodic, dynamic, special, variance

        Returns:
            Group name (default 'unknown')
        """
        return "unknown"
