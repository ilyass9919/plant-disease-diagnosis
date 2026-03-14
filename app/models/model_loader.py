import os
import yaml
import logging
from pathlib import Path
from functools import lru_cache

import tensorflow as tf
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class ModelConfig:
    """Holds everything read from model_config.yaml + .env."""

    def __init__(self, config_dict: dict):
        self.version           = config_dict["version"]
        self.model_file        = config_dict["model_file"]
        self.architecture      = config_dict["architecture"]
        self.input_size: tuple = tuple(config_dict["input_size"])      # (224, 224)
        self.num_classes: int  = config_dict["num_classes"]
        self.class_names: list = config_dict["class_names"]
        self.preprocessing     = config_dict["preprocessing"]          

        raw_thresholds         = config_dict.get("thresholds", {})
        # .env can override yaml thresholds
        self.threshold_confident: float = float(
            os.getenv("THRESHOLD_CONFIDENT", raw_thresholds.get("confident", 0.75))
        )
        self.threshold_uncertain: float = float(
            os.getenv("THRESHOLD_UNCERTAIN", raw_thresholds.get("uncertain", 0.50))
        )

    def __repr__(self):
        return (
            f"ModelConfig(version={self.version}, arch={self.architecture}, "
            f"classes={self.num_classes}, "
            f"thresholds=confident:{self.threshold_confident}/"
            f"uncertain:{self.threshold_uncertain})"
        )


class LoadedModel:
    """Container for the Keras model + its config."""

    def __init__(self, model: tf.keras.Model, config: ModelConfig):
        self.model  = model
        self.config = config


@lru_cache(maxsize=1)
def get_model() -> LoadedModel:
    """
    Load model exactly once (cached).
    Called at startup via FastAPI lifespan, then reused on every request.
    """
    saved_models_dir = Path(os.getenv("SAVED_MODELS_DIR", "saved_models"))
    model_file       = os.getenv("MODEL_FILE")
    config_file      = Path(os.getenv("CONFIG_FILE", ""))

    if not model_file:
        raise RuntimeError("MODEL_FILE not set in .env")

    model_path = saved_models_dir / model_file

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Make sure you downloaded it from Kaggle and placed it in '{saved_models_dir}/'"
        )

    if not config_file.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_file}\n"
            f"Make sure model_config_<version>.yaml is in 'training/configs/'"
        )

    logger.info(f"Loading model from {model_path} ...")
    model = tf.keras.models.load_model(str(model_path))
    logger.info("Model loaded ✓")

    with open(config_file) as f:
        config_dict = yaml.safe_load(f)

    config = ModelConfig(config_dict)
    logger.info(f"Config loaded: {config}")
    # Warmup: run one dummy prediction to compile TF graph at startup 
    import numpy as np
    dummy = np.zeros((1, *config.input_size, 3), dtype=np.float32)
    model.predict(dummy, verbose=0)
    logger.info("Model warmup complete - first request will be fast ✓")

    return LoadedModel(model=model, config=config)