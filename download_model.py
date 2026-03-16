import os
import logging
from pathlib import Path
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

HF_REPO_ID  = os.getenv("HF_REPO_ID")
MODEL_FILE  = os.getenv("MODEL_FILE")
CONFIG_FILE = os.getenv("CONFIG_FILE", "")


def download_if_needed():
    if not HF_REPO_ID:
        logger.info("HF_REPO_ID not set - skipping model download (using local files).")
        return

    if not MODEL_FILE:
        raise RuntimeError("MODEL_FILE not set in .env")

    model_path  = Path("saved_models") / MODEL_FILE
    config_path = Path(CONFIG_FILE) if CONFIG_FILE else None

    Path("saved_models").mkdir(exist_ok=True)
    Path("training/configs").mkdir(parents=True, exist_ok=True)

    # Download model 
    if not model_path.exists():
        logger.info(f"Downloading model from HuggingFace ({HF_REPO_ID})...")
        hf_hub_download(
            repo_id   = HF_REPO_ID,
            filename  = MODEL_FILE,
            local_dir = "saved_models",
        )
        logger.info("Model downloaded ✓")
    else:
        logger.info(f"Model already exists locally — skipping download.")

    # Download config 
    if config_path and not config_path.exists():
        config_filename = Path(CONFIG_FILE).name
        logger.info(f"Downloading config ({config_filename})...")
        hf_hub_download(
            repo_id   = HF_REPO_ID,
            filename  = config_filename,
            local_dir = "training/configs",
        )
        logger.info("Config downloaded ✓")
    else:
        logger.info("Config already exists locally — skipping download.")

    # Download registry 
    registry_path = Path("saved_models/registry.json")
    if not registry_path.exists():
        logger.info("Downloading registry.json...")
        hf_hub_download(
            repo_id   = HF_REPO_ID,
            filename  = "registry.json",
            local_dir = "saved_models",
        )
        logger.info("Registry downloaded ✓")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    download_if_needed()