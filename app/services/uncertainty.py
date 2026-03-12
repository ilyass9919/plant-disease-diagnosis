"""
uncertainty.py
Applies confidence thresholds to produce a prediction status.
Kept separate so thresholds can be changed or extended without
touching inference or routing logic.
"""
from app.schemas.response import PredictionStatus
from app.models.model_loader import ModelConfig


def resolve_status(confidence: float, config: ModelConfig) -> PredictionStatus:
    """
    Maps a confidence score to a PredictionStatus.

    >= threshold_confident  → CONFIDENT
    >= threshold_uncertain  → UNCERTAIN
    < threshold_uncertain   → FAILED
    """
    if confidence >= config.threshold_confident:
        return PredictionStatus.CONFIDENT
    elif confidence >= config.threshold_uncertain:
        return PredictionStatus.UNCERTAIN
    else:
        return PredictionStatus.FAILED