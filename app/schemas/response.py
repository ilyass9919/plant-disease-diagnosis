from pydantic import BaseModel, ConfigDict
from typing import Optional
from enum import Enum


class PredictionStatus(str, Enum):
    CONFIDENT = "CONFIDENT"
    UNCERTAIN = "UNCERTAIN"
    FAILED    = "FAILED"


class TopKPrediction(BaseModel):
    disease_class: str
    confidence:    float


# Optional metadata sent by the mobile app alongside the image 
class PredictionMetadata(BaseModel):
    """
    Optional metadata the mobile app can include with each prediction request.
    All fields are optional — the API works fine without any of them.
    Stored in the prediction log for continuous improvement and bias analysis.
    """
    latitude:     Optional[float] = None   
    longitude:    Optional[float] = None   
    device_model: Optional[str]   = None   
    app_version:  Optional[str]   = None   
    notes:        Optional[str]   = None   


# Prediction response
class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    prediction_id:         str                  # for human review submission
    status:                PredictionStatus
    predicted_class:       Optional[str]   = None
    confidence:            Optional[float] = None
    summary:               Optional[str]   = None
    recommended_treatment: Optional[str]   = None
    top_k:                 list[TopKPrediction] = []

    # Explanation fields
    model_version:         str
    inference_time_ms:     float


# Human review submission
class HumanReviewRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    """Submitted by an agronomist after reviewing an UNCERTAIN prediction."""
    reviewer_id:       str
    correct_class:     str
    model_was_correct: bool
    notes:             Optional[str] = None


class HumanReviewResponse(BaseModel):
    prediction_id: str
    status:        str
    correct_class: str
    reviewed_at:   str


# Health check
class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status:        str
    model_version: str
    architecture:  str
    num_classes:   int
    class_names:   list[str]