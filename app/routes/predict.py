"""
predict.py
POST /predict — main inference endpoint.
GET  /         — health check.
"""
import logging
from fastapi import APIRouter, File, UploadFile, HTTPException

from app.models.model_loader import get_model
from app.services.inference import run_inference
from app.services.report.static_report import generate_report
from app.storage.prediction_store import log_prediction
from app.schemas.response import (
    PredictionResponse,
    PredictionStatus,
    HealthResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/bmp",
    "image/webp",
}


@router.get("/", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Returns model info and confirms the API is running."""
    loaded = get_model()
    return HealthResponse(
        status       = "ok",
        model_version= loaded.config.version,
        architecture = loaded.config.architecture,
        num_classes  = loaded.config.num_classes,
        class_names  = loaded.config.class_names,
    )


@router.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(file: UploadFile = File(...)):
    """
    Accepts a tomato leaf image and returns a disease prediction.

    - **file**: image file (JPEG, PNG, BMP, WebP)

    Returns:
    - **status**: CONFIDENT | UNCERTAIN | FAILED
    - **predicted_class**: disease class name
    - **confidence**: model confidence score (0-1)
    - **summary**: plain-language disease description
    - **recommended_treatment**: step-by-step treatment advice
    - **top_k**: top 3 predictions with confidence scores
    - **model_version**: which model version produced this result
    - **inference_time_ms**: server-side processing time
    """
    # Validate file type
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type '{file.content_type}'. "
                f"Please upload a JPEG, PNG, BMP, or WebP image."
            ),
        )

    # Read image bytes
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Run inference 
    loaded = get_model()
    result = run_inference(image_bytes, loaded)

    # Generate report 
    summary, treatment = generate_report(result["predicted_class"], result["status"])

    # Log prediction for continuous improvement
    log_prediction(
        predicted_class   = result["predicted_class"],
        confidence        = result["confidence"],
        status            = result["status"].value,
        top_k             = result["top_k"],
        model_version     = loaded.config.version,
        inference_time_ms = result["inference_time_ms"],
        image_filename    = file.filename or "unknown",
    )

    # Build response 
    return PredictionResponse(
        status                = result["status"],
        predicted_class       = result["predicted_class"],
        confidence            = result["confidence"],
        summary               = summary,
        recommended_treatment = treatment,
        top_k                 = result["top_k"],
        model_version         = loaded.config.version,
        inference_time_ms     = result["inference_time_ms"],
    )
