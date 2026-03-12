"""
prediction_store.py
Logs every prediction to a JSONL file for continuous improvement.

Each line is a self-contained JSON record:
{
    "id":               "uuid",
    "timestamp":        "ISO8601",
    "predicted_class":  "...",
    "confidence":       0.93,
    "status":           "CONFIDENT",
    "top_k":            [...],
    "model_version":    "v1_20260309_1347",
    "inference_time_ms": 42.3,
    "image_filename":   "upload.jpg",    # original filename from client
    "human_review":     null             # filled in later by support/validation
}

The JSONL format means each prediction is one line — easy to stream,
grep, and load into pandas for retraining analysis.
"""
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

LOG_PATH = Path(os.getenv("PREDICTIONS_LOG", "data/predictions.jsonl"))


def _ensure_log_file():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not LOG_PATH.exists():
        LOG_PATH.touch()


def log_prediction(
    predicted_class: str | None,
    confidence: float,
    status: str,
    top_k: list,
    model_version: str,
    inference_time_ms: float,
    image_filename: str = "unknown",
) -> str:
    """
    Appends a prediction record to the JSONL log.
    Returns the generated prediction ID.
    """
    _ensure_log_file()

    prediction_id = str(uuid.uuid4())
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
        "human_review"     : None,   # to be filled by support validation later
    }

    try:
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")
        logger.debug(f"Prediction logged: {prediction_id}")
    except Exception as e:
        # Logging failure should never break the API response
        logger.error(f"Failed to log prediction {prediction_id}: {e}")

    return prediction_id