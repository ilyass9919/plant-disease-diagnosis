import time
import logging
import numpy as np
from io import BytesIO
from PIL import Image

import tensorflow as tf

from app.models.model_loader import LoadedModel, ModelConfig
from app.schemas.response import PredictionStatus, TopKPrediction
from app.services.uncertainty import resolve_status

logger = logging.getLogger(__name__)

TOP_K = 3  


def _preprocess_image(image_bytes: bytes, input_size: tuple) -> np.ndarray:
    """
    Converts raw image bytes → model-ready numpy array.
    - Decodes image (supports JPEG, PNG, BMP, WebP)
    - Converts to RGB (handles RGBA / greyscale uploads)
    - Resizes to model input size
    - Expands to batch dimension: (1, H, W, 3)
    - EfficientNetB0: pixel values stay in [0, 255] (no normalization here)
    """
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img = img.resize(input_size, Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)   # (1, 224, 224, 3)
    return arr


def run_inference(image_bytes: bytes, loaded_model: LoadedModel) -> dict:
    """
    Full inference pipeline.

    Returns a dict with:
        predicted_class  : str
        confidence       : float
        status           : PredictionStatus
        top_k            : list[TopKPrediction]
        inference_time_ms: float
    """
    config: ModelConfig = loaded_model.config

    # Preprocess 
    try:
        arr = _preprocess_image(image_bytes, config.input_size)
    except Exception as e:
        logger.warning(f"Image preprocessing failed: {e}")
        return {
            "predicted_class"  : None,
            "confidence"       : 0.0,
            "status"           : PredictionStatus.FAILED,
            "top_k"            : [],
            "inference_time_ms": 0.0,
        }

    # Predict 
    t0    = time.perf_counter()
    probs = loaded_model.model.predict(arr, verbose=0)[0]   # shape: (num_classes,)
    inference_time_ms = (time.perf_counter() - t0) * 1000

    # Decode 
    top_indices    = np.argsort(probs)[::-1]
    best_idx       = int(top_indices[0])
    confidence     = float(probs[best_idx])
    predicted_class = config.class_names[best_idx]

    status = resolve_status(confidence, config)

    top_k = [
        TopKPrediction(
            disease_class=config.class_names[int(i)],
            confidence=round(float(probs[i]), 4),
        )
        for i in top_indices[:TOP_K]
    ]

    logger.info(
        f"Inference done | class={predicted_class} conf={confidence:.3f} "
        f"status={status} time={inference_time_ms:.1f}ms"
    )

    return {
        "predicted_class"  : predicted_class,
        "confidence"       : round(confidence, 4),
        "status"           : status,
        "top_k"            : top_k,
        "inference_time_ms": round(inference_time_ms, 2),
    }