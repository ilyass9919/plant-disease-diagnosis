import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from io import BytesIO
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Detect environment 
SUPABASE_URL  = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY  = os.getenv("SUPABASE_KEY", "")
CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME", "")
CLOUDINARY_API_KEY    = os.getenv("CLOUDINARY_API_KEY", "")
CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET", "")
CLOUDINARY_URL        = os.getenv("CLOUDINARY_URL", "")


def _derive_cloudinary_credentials_from_url() -> tuple[str, str, str]:
    if not CLOUDINARY_URL:
        return "", "", ""
    try:
        parsed = urlparse(CLOUDINARY_URL)
        cloud_name = parsed.hostname or ""
        api_key = parsed.username or ""
        api_secret = parsed.password or ""
        return cloud_name, api_key, api_secret
    except Exception:
        return "", "", ""


if CLOUDINARY_URL and not (CLOUDINARY_CLOUD_NAME and CLOUDINARY_API_KEY and CLOUDINARY_API_SECRET):
    derived_name, derived_key, derived_secret = _derive_cloudinary_credentials_from_url()
    CLOUDINARY_CLOUD_NAME = CLOUDINARY_CLOUD_NAME or derived_name
    CLOUDINARY_API_KEY = CLOUDINARY_API_KEY or derived_key
    CLOUDINARY_API_SECRET = CLOUDINARY_API_SECRET or derived_secret

USE_SUPABASE = bool(SUPABASE_URL and SUPABASE_KEY)
USE_CLOUDINARY = bool(CLOUDINARY_CLOUD_NAME and CLOUDINARY_API_KEY and CLOUDINARY_API_SECRET)
USE_CLOUD = USE_SUPABASE and USE_CLOUDINARY

# Local fallback paths 
LOG_PATH   = Path(os.getenv("PREDICTIONS_LOG", "data/predictions.jsonl"))
IMAGES_DIR = Path(os.getenv("PREDICTIONS_IMAGES_DIR", "data/images"))


# Lazy clients (only initialised if cloud vars are set) 
_supabase_client  = None
_cloudinary_ready = False


def _get_supabase():
    global _supabase_client
    if _supabase_client is None:
        from supabase import create_client
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _supabase_client


def _get_cloudinary():
    global _cloudinary_ready
    if not _cloudinary_ready:
        import cloudinary
        cloudinary.config(
            cloud_name = CLOUDINARY_CLOUD_NAME,
            api_key    = CLOUDINARY_API_KEY,
            api_secret = CLOUDINARY_API_SECRET,
            secure     = True,
        )
        _cloudinary_ready = True


# Image upload
def _upload_image_cloudinary(prediction_id: str, image_bytes: bytes) -> str | None:
    """Uploads image to Cloudinary. Returns the secure URL."""
    try:
        import cloudinary.uploader
        _get_cloudinary()
        result = cloudinary.uploader.upload(
            BytesIO(image_bytes),
            public_id = f"leafscan/{prediction_id}",
            folder    = "leafscan",
            overwrite = True,
        )
        return result.get("secure_url")
    except Exception as e:
        logger.error(f"Cloudinary upload failed for {prediction_id}: {e}")
        return None


def _save_image_local(prediction_id: str, image_bytes: bytes) -> str | None:
    """Fallback: save image to local disk."""
    try:
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        image_path = IMAGES_DIR / f"{prediction_id}.jpg"
        image_path.write_bytes(image_bytes)
        return str(image_path)
    except Exception as e:
        logger.error(f"Local image save failed for {prediction_id}: {e}")
        return None


# Prediction logging
def _log_to_supabase(record: dict) -> bool:
    """Inserts prediction record into Supabase."""
    try:
        db = _get_supabase()
        db.table("predictions").insert(record).execute()
        return True
    except Exception as e:
        logger.error(f"Supabase insert failed: {e}")
        return False


def _log_to_local(record: dict) -> bool:
    """Fallback: append prediction to local JSONL file."""
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not LOG_PATH.exists():
            LOG_PATH.touch()
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")
        return True
    except Exception as e:
        logger.error(f"Local log failed: {e}")
        return False


# Public API 
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
    Saves image (Cloudinary or local) and logs prediction (Supabase or JSONL).
    Returns the prediction ID.
    """
    prediction_id = str(uuid.uuid4())

    # Save image 
    if USE_CLOUDINARY:
        image_url = _upload_image_cloudinary(prediction_id, image_bytes)
        if image_url:
            logger.info(f"Image uploaded to Cloudinary: {image_url}")
        else:
            logger.warning("Cloudinary upload failed — falling back to local image storage")
            image_url = _save_image_local(prediction_id, image_bytes)
    else:
        image_url = _save_image_local(prediction_id, image_bytes)
        logger.debug(f"Image saved locally: {image_url}")

    # Build record 
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
        "image_url"        : image_url,
        "metadata"         : metadata or {},
        # human_review intentionally omitted - stays as SQL NULL in Supabase
    }

    # Log prediction 
    if USE_SUPABASE:
        success = _log_to_supabase(record)
        if not success:
            logger.warning("Supabase failed — falling back to local JSONL")
            _log_to_local(record)
    else:
        _log_to_local(record)

    logger.debug(f"Prediction logged: {prediction_id}")
    return prediction_id


def get_prediction(prediction_id: str) -> dict | None:
    """Fetches a single prediction by ID."""
    if USE_SUPABASE:
        try:
            db     = _get_supabase()
            result = db.table("predictions").select("*").eq("id", prediction_id).execute()
            if result.data:
                return result.data[0]
            return None
        except Exception as e:
            logger.error(f"Supabase fetch failed: {e}")
            return None
    else:
        # Local fallback
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


def get_pending_reviews() -> list[dict]:
    if USE_SUPABASE:
        try:
            db     = _get_supabase()
            result = (
                db.table("predictions")
                .select("*")
                .eq("status", "UNCERTAIN")
                .is_("human_review", "null")   # column is SQL NULL
                .order("timestamp", desc=True)
                .execute()
            )
            return result.data or []
        except Exception as e:
            logger.error(f"Supabase pending-review fetch failed: {e}")
            return []
    else:
        if not LOG_PATH.exists():
            return []
        results = []
        with open(LOG_PATH) as f:
            for line in f:
                try:
                    record = json.loads(line)
                    if record.get("status") == "UNCERTAIN" and record.get("human_review") is None:
                        results.append(record)
                except json.JSONDecodeError:
                    continue
        return sorted(results, key=lambda r: r.get("timestamp", ""), reverse=True)


def update_human_review(prediction_id: str, review: dict) -> bool:
    """Updates the human_review field for a prediction."""
    if USE_SUPABASE:
        try:
            db = _get_supabase()
            db.table("predictions").update({"human_review": review}).eq("id", prediction_id).execute()
            logger.info(f"Human review recorded in Supabase: {prediction_id}")
            return True
        except Exception as e:
            logger.error(f"Supabase update failed: {e}")
            return False
    else:
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
            logger.info(f"Human review recorded locally: {prediction_id}")
        return updated
