import base64
import json
import logging
import os
from urllib.error import URLError, HTTPError
from urllib.request import urlopen
from typing import Optional
from pathlib import Path

from fastapi import APIRouter, File, Form, UploadFile, HTTPException, Header
from fastapi.responses import FileResponse, Response

from app.models.model_loader import get_model
from app.services.inference import run_inference
from app.services.report import generate_report
from app.storage.prediction_store import (
    log_prediction, get_prediction,
    update_human_review, get_pending_reviews,
)
from app.schemas.response import (
    PredictionResponse,
    PredictionStatus,
    HumanReviewRequest,
    HumanReviewResponse,
    HealthResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()

ALLOWED_CONTENT_TYPES = {
    "image/jpeg", "image/jpg", "image/png", "image/bmp", "image/webp",
}

IMAGES_DIR = Path(os.getenv("PREDICTIONS_IMAGES_DIR", "data/images"))


# Auth helper
def verify_agronomist(x_password: Optional[str]):
    expected = os.getenv("AGRONOMIST_PASSWORD", "")
    if not expected:
        return   # no password set - allow access (dev mode)
    if x_password != expected:
        raise HTTPException(
            status_code=401,
            detail="Invalid agronomist password.",
        )


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

@router.get("/debug/storage", tags=["Debug"])
def debug_storage():
    """Temporary debug endpoint — shows storage config on Render."""
    import os
    from app.storage.prediction_store import USE_CLOUD
    return {
        "USE_CLOUD"             : USE_CLOUD,
        "SUPABASE_URL_set"      : bool(os.getenv("SUPABASE_URL")),
        "SUPABASE_KEY_set"      : bool(os.getenv("SUPABASE_KEY")),
        "CLOUDINARY_NAME_set"   : bool(os.getenv("CLOUDINARY_CLOUD_NAME")),
        "CLOUDINARY_KEY_set"    : bool(os.getenv("CLOUDINARY_API_KEY")),
        "CLOUDINARY_SECRET_set" : bool(os.getenv("CLOUDINARY_API_SECRET")),
        "SUPABASE_URL_value"    : os.getenv("SUPABASE_URL", "NOT SET")[:30],
    }


@router.post("/predict", response_model=PredictionResponse, tags=["Inference"])
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
    """
    image_bytes    = None
    image_filename = "unknown"

    if file is not None:
        if file.content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type '{file.content_type}'.",
            )
        image_bytes    = await file.read()
        image_filename = file.filename or "unknown"

    elif image_base64 is not None:
        try:
            if "," in image_base64:
                image_base64 = image_base64.split(",", 1)[1]
            image_bytes    = base64.b64decode(image_base64)
            image_filename = "base64_upload"
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 image string.")
    else:
        raise HTTPException(
            status_code=422,
            detail="Provide either 'file' (multipart) or 'image_base64' (form field).",
        )

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    metadata_dict = {}
    if metadata_json:
        try:
            metadata_dict = json.loads(metadata_json)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=400,
                detail="Invalid metadata_json — must be a valid JSON string.",
            )

    loaded  = get_model()
    result  = run_inference(image_bytes, loaded)
    summary, treatment = generate_report(result["predicted_class"], result["status"])

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


@router.get("/predictions/pending-review", tags=["Human Review"])
def pending_review(x_password: Optional[str] = Header(default=None)):
    """
    Returns all UNCERTAIN predictions that have not yet been reviewed.
    Protected by X-Password header.
    """
    verify_agronomist(x_password)
    records = get_pending_reviews()
    return {"count": len(records), "predictions": records}


@router.get("/predictions/{prediction_id}/image", tags=["Human Review"])
def get_image(
    prediction_id: str,
    x_password: Optional[str] = Header(default=None),
):
    """
    Serves the leaf image for a prediction.
    - If Cloudinary: proxies bytes from the stored URL
    - If local: serves the file directly
    """
    verify_agronomist(x_password)

    record = get_prediction(prediction_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Prediction not found.")

    from app.storage.prediction_store import USE_CLOUD

    # Cloud mode - proxy image bytes.
    # NOTE: We avoid browser-side redirects here because the review UI fetches
    # this endpoint with auth headers, and cross-origin redirects can fail due
    # to CORS when converted to a Blob in JS.
    if USE_CLOUD:
        image_url = record.get("image_url")
        if not image_url:
            raise HTTPException(status_code=404, detail="Image URL not found.")
        try:
            with urlopen(image_url, timeout=10) as upstream:
                image_bytes = upstream.read()
                media_type = upstream.headers.get_content_type() or "image/jpeg"
            return Response(content=image_bytes, media_type=media_type)
        except (HTTPError, URLError, TimeoutError) as e:
            logger.error(f"Failed to fetch cloud image for {prediction_id}: {e}")
            raise HTTPException(status_code=502, detail="Failed to fetch cloud image.")

    # Local mode - serve file from disk
    image_path = IMAGES_DIR / f"{prediction_id}.jpg"
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found.")
    return FileResponse(str(image_path), media_type="image/jpeg")


@router.patch(
    "/predictions/{prediction_id}/review",
    response_model=HumanReviewResponse,
    tags=["Human Review"],
)
def submit_review(
    prediction_id: str,
    review: HumanReviewRequest,
    x_password: Optional[str] = Header(default=None),
):
    """
    Submit a human review for an UNCERTAIN prediction.
    Protected by X-Password header.
    """
    from datetime import datetime, timezone

    verify_agronomist(x_password)

    record = get_prediction(prediction_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Prediction '{prediction_id}' not found.")

    if record.get("human_review") is not None:
        raise HTTPException(status_code=409, detail="Already reviewed.")

    reviewed_at = datetime.now(timezone.utc).isoformat()
    review_dict = {
        "reviewer_id"      : review.reviewer_id,
        "reviewed_at"      : reviewed_at,
        "correct_class"    : review.correct_class,
        "model_was_correct": review.model_was_correct,
        "notes"            : review.notes,
    }

    if not update_human_review(prediction_id, review_dict):
        raise HTTPException(status_code=500, detail="Failed to save review.")

    return HumanReviewResponse(
        prediction_id = prediction_id,
        status        = "review_recorded",
        correct_class = review.correct_class,
        reviewed_at   = reviewed_at,
    )
