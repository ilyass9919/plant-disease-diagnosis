import base64
import json
import logging
from typing import Optional

from fastapi import APIRouter, File, Form, UploadFile, HTTPException

from app.models.model_loader import get_model
from app.services.inference import run_inference
from app.services.report.static_report import generate_report
from app.storage.prediction_store import log_prediction, get_prediction, update_human_review
from app.schemas.response import (
    PredictionResponse,
    PredictionStatus,
    PredictionMetadata,
    HumanReviewRequest,
    HumanReviewResponse,
    HealthResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()

ALLOWED_CONTENT_TYPES = {
    "image/jpeg", "image/jpg", "image/png", "image/bmp", "image/webp",
}

@router.get("/", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Returns model info and confirms the API is running."""
    loaded = get_model()
    return HealthResponse(
        status        = "ok",
        model_version = loaded.config.version,
        architecture  = loaded.config.architecture,
        num_classes   = loaded.config.num_classes,
        class_names   = loaded.config.class_names,
    )


@router.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Inference"],
    openapi_extra={
        "requestBody": {
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "file": {
                                "type": "string",
                                "format": "binary",
                                "description": "Image file (JPEG, PNG, BMP, WebP)",
                            },
                            "image_base64": {
                                "type": "string",
                                "description": "Base64-encoded image (alternative to file upload)",
                            },
                            "metadata_json": {
                                "type": "string",
                                "description": 'Optional JSON: {"latitude": 33.57, "longitude": -7.59, "device_model": "...", "app_version": "...", "notes": "..."}',
                            },
                        },
                    }
                }
            }
        }
    },
)
async def predict(
    file:          Optional[UploadFile] = File(default=None),
    image_base64:  Optional[str]        = Form(default=None),
    metadata_json: Optional[str]        = Form(default=None),
):
    """
    Accepts a tomato leaf image and returns a disease prediction.

    **Two ways to send the image (use one):**
    - `file`: multipart file upload (JPEG, PNG, BMP, WebP)
    - `image_base64`: base64-encoded image string

    **Optional metadata** (JSON string in `metadata_json` form field):
    ```json
    {
      "latitude": 33.57,
      "longitude": -7.59,
      "device_model": "Samsung Galaxy S23",
      "app_version": "1.2.0",
      "notes": "Lower leaf, spotted 2 days ago"
    }
    ```

    **Returns:**
    - `prediction_id`: use this to submit a human review later
    - `status`: CONFIDENT | UNCERTAIN | FAILED
    - `predicted_class`, `confidence`, `top_k`
    - `summary` + `recommended_treatment` (null when FAILED)
    - `model_version`, `inference_time_ms`
    """
    image_bytes  = None
    image_filename = "unknown"

    if file is not None:
        if file.content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(
                status_code=415,
                detail=(
                    f"Unsupported file type '{file.content_type}'. "
                    f"Please upload a JPEG, PNG, BMP, or WebP image."
                ),
            )
        image_bytes    = await file.read()
        image_filename = file.filename or "unknown"

    elif image_base64 is not None:
        try:
            # Strip data URI prefix if present: "data:image/jpeg;base64,..."
            if "," in image_base64:
                image_base64 = image_base64.split(",", 1)[1]
            image_bytes    = base64.b64decode(image_base64)
            image_filename = "base64_upload"
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="Invalid base64 image string. Could not decode.",
            )
    else:
        raise HTTPException(
            status_code=422,
            detail="Provide either 'file' (multipart) or 'image_base64' (form field).",
        )

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    # Parse optional metadata 
    metadata_dict = {}
    if metadata_json:
        try:
            metadata_dict = json.loads(metadata_json)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=400,
                detail="Invalid metadata_json — must be a valid JSON string.",
            )

    # Run inference 
    loaded = get_model()
    result = run_inference(image_bytes, loaded)

    # Generate report 
    summary, treatment = generate_report(result["predicted_class"], result["status"])

    # Log prediction + save image for future review/training
    prediction_id = log_prediction(
        predicted_class   = result["predicted_class"],
        confidence        = result["confidence"],
        status            = result["status"].value,
        top_k             = result["top_k"],
        model_version     = loaded.config.version,
        inference_time_ms = result["inference_time_ms"],
        image_bytes       = image_bytes,
        image_filename    = image_filename,
        metadata          = metadata_dict,
    )

    # Build response
    return PredictionResponse(
        prediction_id         = prediction_id,
        status                = result["status"],
        predicted_class       = result["predicted_class"],
        confidence            = result["confidence"],
        summary               = summary,
        recommended_treatment = treatment,
        top_k                 = result["top_k"],
        model_version         = loaded.config.version,
        inference_time_ms     = result["inference_time_ms"],
    )


@router.patch(
    "/predictions/{prediction_id}/review",
    response_model=HumanReviewResponse,
    tags=["Human Review"],
)
def submit_review(prediction_id: str, review: HumanReviewRequest):
    """
    Submit a human review for an UNCERTAIN prediction.

    Called by an agronomist after inspecting the image.
    Stores the correction in the prediction log for future retraining.

    - **prediction_id**: the ID returned by `POST /predict`
    - **reviewer_id**: identifier of the agronomist
    - **correct_class**: the actual disease (may differ from model prediction)
    - **model_was_correct**: whether the model got it right
    - **notes**: optional free-text observations
    """
    from datetime import datetime, timezone

    record = get_prediction(prediction_id)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"Prediction '{prediction_id}' not found.",
        )

    if record.get("human_review") is not None:
        raise HTTPException(
            status_code=409,
            detail=f"Prediction '{prediction_id}' has already been reviewed.",
        )

    reviewed_at = datetime.now(timezone.utc).isoformat()
    review_dict = {
        "reviewer_id"      : review.reviewer_id,
        "reviewed_at"      : reviewed_at,
        "correct_class"    : review.correct_class,
        "model_was_correct": review.model_was_correct,
        "notes"            : review.notes,
    }

    updated = update_human_review(prediction_id, review_dict)
    if not updated:
        raise HTTPException(status_code=500, detail="Failed to save review.")

    return HumanReviewResponse(
        prediction_id = prediction_id,
        status        = "review_recorded",
        correct_class = review.correct_class,
        reviewed_at   = reviewed_at,
    )