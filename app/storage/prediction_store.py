import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

LOG_PATH   = Path(os.getenv("PREDICTIONS_LOG", "data/predictions.jsonl"))
IMAGES_DIR = Path(os.getenv("PREDICTIONS_IMAGES_DIR", "data/images"))


def _ensure_dirs():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    if not LOG_PATH.exists():
        LOG_PATH.touch()


def _save_image(prediction_id: str, image_bytes: bytes) -> str | None:
    """Saves raw image bytes to data/images/<id>.jpg. Returns the path string."""
    try:
        image_path = IMAGES_DIR / f"{prediction_id}.jpg"
        image_path.write_bytes(image_bytes)
        return str(image_path)
    except Exception as e:
        logger.error(f"Failed to save image for {prediction_id}: {e}")
        return None


def log_prediction(
    predicted_class:   str | None,
    confidence:        float,
    status:            str,
    top_k:             list,
    model_version:     str,
    inference_time_ms: float,
    image_bytes:       bytes,
    image_filename:    str = "unknown",
    metadata:          dict | None = None,
) -> str:
    """
    Saves the image to disk and appends a full prediction record to the JSONL log.
    Returns the generated prediction ID.
    """
    _ensure_dirs()
    prediction_id = str(uuid.uuid4())
    saved_path    = _save_image(prediction_id, image_bytes)

    record = {
        "id"               : prediction_id,
        "timestamp"        : datetime.now(timezone.utc).isoformat(),
        "predicted_class"  : predicted_class,
        "confidence"       : confidence,
        "status"           : status,
        "top_k"            : [t.model_dump() for t in top_k],
        "model_version"    : model_version,
        "inference_time_ms": inference_time_ms,
        "image_filename"   : image_filename,
        "image_saved_path" : saved_path,
        "metadata"         : metadata or {},
        "human_review"     : None,
    }

    try:
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")
        logger.debug(f"Prediction logged: {prediction_id}")
    except Exception as e:
        logger.error(f"Failed to log prediction {prediction_id}: {e}")

    return prediction_id


def get_prediction(prediction_id: str) -> dict | None:
    """Finds and returns a single prediction record by ID."""
    if not LOG_PATH.exists():
        return None
    with open(LOG_PATH) as f:
        for line in f:
            try:
                record = json.loads(line)
                if record.get("id") == prediction_id:
                    return record
            except json.JSONDecodeError:
                continue
    return None


def update_human_review(prediction_id: str, review: dict) -> bool:
    """
    Updates the human_review field of a prediction record in-place.
    Returns True if found and updated, False otherwise.
    """
    if not LOG_PATH.exists():
        return False

    lines   = LOG_PATH.read_text().splitlines()
    updated = False
    new_lines = []

    for line in lines:
        try:
            record = json.loads(line)
            if record.get("id") == prediction_id:
                record["human_review"] = review
                updated = True
            new_lines.append(json.dumps(record))
        except json.JSONDecodeError:
            new_lines.append(line)

    if updated:
        LOG_PATH.write_text("\n".join(new_lines) + "\n")
        logger.info(f"Human review recorded for {prediction_id}")

    return updated