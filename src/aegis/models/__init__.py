"""Temporal models for AEGIS.

This module provides all temporal models and the factory function
to create a standard model bank.
"""

from aegis.config import AEGISConfig
from aegis.models.base import TemporalModel
from aegis.models.dynamic import AR2Model, MA1Model
from aegis.models.periodic import OscillatorBankModel, SeasonalDummyModel
from aegis.models.persistence import LocalLevelModel, RandomWalkModel
from aegis.models.reversion import (
    AsymmetricMeanReversionModel,
    MeanReversionModel,
    ThresholdARModel,
)
from aegis.models.special import ChangePointModel, JumpDiffusionModel
from aegis.models.trend import DampedTrendModel, LocalTrendModel
from aegis.models.variance import LevelDependentVolModel, VolatilityTrackerModel

__all__ = [
    "TemporalModel",
    "RandomWalkModel",
    "LocalLevelModel",
    "LocalTrendModel",
    "DampedTrendModel",
    "MeanReversionModel",
    "AsymmetricMeanReversionModel",
    "ThresholdARModel",
    "OscillatorBankModel",
    "SeasonalDummyModel",
    "AR2Model",
    "MA1Model",
    "JumpDiffusionModel",
    "ChangePointModel",
    "VolatilityTrackerModel",
    "LevelDependentVolModel",
    "create_model_bank",
]


def create_model_bank(config: AEGISConfig) -> list[TemporalModel]:
    """Create a standard model bank for AEGIS.

    Creates one instance of each model type with appropriate
    parameters from the configuration.

    Args:
        config: AEGIS configuration

    Returns:
        List of TemporalModel instances
    """
    models: list[TemporalModel] = []

    models.append(RandomWalkModel())
    models.append(LocalLevelModel())

    models.append(LocalTrendModel())
    models.append(DampedTrendModel())

    models.append(MeanReversionModel())
    models.append(AsymmetricMeanReversionModel())
    models.append(ThresholdARModel())

    for period in config.oscillator_periods:
        models.append(OscillatorBankModel(periods=[period]))

    for period in config.seasonal_periods:
        models.append(SeasonalDummyModel(period=period))

    models.append(AR2Model())
    models.append(MA1Model())

    models.append(JumpDiffusionModel())
    models.append(ChangePointModel())

    models.append(VolatilityTrackerModel(decay=config.volatility_decay))
    models.append(LevelDependentVolModel())

    return models
