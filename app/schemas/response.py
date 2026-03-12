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


class PredictionResponse(BaseModel):

    model_config = ConfigDict(protected_namespaces={})

    status:               PredictionStatus
    predicted_class:      Optional[str]   = None
    confidence:           Optional[float] = None
    summary:              Optional[str]   = None
    recommended_treatment: Optional[str]  = None
    top_k:                list[TopKPrediction] = []

    # Explanation fields (transparency for the caller)
    model_version:        str
    inference_time_ms:    float


class HealthResponse(BaseModel):

    model_config = ConfigDict(protected_namespaces={})

    status:        str
    model_version: str
    architecture:  str
    num_classes:   int
    class_names:   list[str]