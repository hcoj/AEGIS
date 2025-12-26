"""AEGIS: Active Epistemic Generative Inference System for time series prediction.

AEGIS is a multi-stream time series prediction system using structured model
ensembles with Expected Free Energy (EFE) weighting.
"""

from aegis.config import AEGISConfig
from aegis.core.prediction import Prediction
from aegis.models import create_model_bank
from aegis.system import AEGIS

__version__ = "0.1.0"

__all__ = [
    "AEGIS",
    "AEGISConfig",
    "Prediction",
    "create_model_bank",
    "__version__",
]
