# Tomato Leaf Disease Diagnostic API

An AI-powered REST API that classifies tomato leaf diseases from smartphone photos and generates diagnostic reports with treatment recommendations.

---

## Overview

This system is built around a computer vision model trained on **13,832 images** from two datasets (PlantDoc + PlantVillage). It detects **8 tomato disease classes** with **93.06% validation accuracy** and exposes the results through a FastAPI backend with full uncertainty handling, continuous improvement logging, and an optional LLM-powered report generator.

---

## Disease Classes

| Class | Description |
|-------|-------------|
| `Tomato_Bacterial_Spot` | Xanthomonas spp. — dark lesions with yellow halos |
| `Tomato_Early_Blight` | Alternaria solani — concentric ring pattern on older leaves |
| `Tomato_Healthy` | No disease detected |
| `Tomato_Late_Blight` | Phytophthora infestans — large greasy dark lesions |
| `Tomato_Leaf_Mold` | Passalora fulva - olive-green mold on leaf underside |
| `Tomato_Mosaic_Virus` | ToMV - mottled mosaic pattern, leaf distortion |
| `Tomato_Septoria_Leaf_Spot` | Septoria lycopersici - small circular spots with dark borders |
| `Tomato_Yellow_Leaf_Curl` | TYLCV via whitefly - severe upward curl, yellowing |

---

## Model Performance

| Metric | Value |
|--------|-------|
| Architecture | EfficientNetB0 (transfer learning) |
| Training data | PlantDoc + PlantVillage (13,832 images) |
| Validation accuracy | **93.06%** |
| Macro F1 | **0.895** |
| Training platform | Kaggle (GPU T4) |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check - returns model info |
| `POST` | `/predict` | Disease prediction from image |
| `PATCH` | `/predictions/{id}/review` | Submit human review for uncertain predictions |

### POST /predict

Accepts image as **multipart file upload** or **base64 string**, plus optional metadata.

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@leaf_photo.jpg" \
  -F 'metadata_json={"latitude": 33.57, "longitude": -7.59, "device_model": "Samsung Galaxy S23"}'
```

**Response:**
```json
{
  "prediction_id": "fb8539e4-2347-4fa3-af47-e1363ad0fbe2",
  "status": "CONFIDENT",
  "predicted_class": "Tomato_Bacterial_Spot",
  "confidence": 0.87,
  "summary": "Bacterial spot is caused by Xanthomonas spp...",
  "recommended_treatment": "1. Remove infected material...",
  "top_k": [
    { "disease_class": "Tomato_Bacterial_Spot", "confidence": 0.87 },
    { "disease_class": "Tomato_Septoria_Leaf_Spot", "confidence": 0.08 },
    { "disease_class": "Tomato_Late_Blight", "confidence": 0.03 }
  ],
  "model_version": "v1_20260309_1523",
  "inference_time_ms": 243.1
}
```

### Prediction Status

| Status | Condition | Action |
|--------|-----------|--------|
| `CONFIDENT` | confidence ≥ 0.60 | Display result to farmer |
| `UNCERTAIN` | confidence ≥ 0.30 | Show result + flag for agronomist review |
| `FAILED` | confidence < 0.30 | Ask farmer to retake photo |

---

## Project Structure

```
tomato-disease-api/
├── app/
│   ├── main.py                         # FastAPI app + CORS + lifespan
│   ├── models/
│   │   └── model_loader.py             # Loads Keras model once at startup
│   ├── routes/
│   │   └── predict.py                  # All endpoints
│   ├── schemas/
│   │   └── response.py                 # Pydantic I/O models
│   ├── services/
│   │   ├── inference.py                # Image preprocessing + prediction
│   │   ├── uncertainty.py              # Confidence → status mapping
│   │   └── report/
│   │       ├── static_report.py        # Hardcoded disease reports
│   │       └── agent_report.py         # LLM-powered reports (optional)
│   └── storage/
│       └── prediction_store.py         # Logs predictions + saves images
├── saved_models/
│   ├── tomato_efficientnetb0_v1_*.keras
│   └── registry.json
├── training/
│   ├── configs/
│   │   └── model_config_v1_*.yaml
│   └── train_plantdoc_tomato.ipynb     # Kaggle training notebook
├── data/
│   ├── predictions.jsonl               # Prediction log (auto-created)
│   └── images/                         # Saved prediction images
├── interface.html                      # Browser-based test interface
├── test_api.py                         # Automated test suite
├── DATASET_STRATEGY.md                 # Data collection + annotation strategy
├── .env                                # Local config (not committed)
└── requirements.txt
```

---

## Setup

```bash
# 1. Clone the repository
git clone <repo-url>
cd tomato-disease-api

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add model files (download from Kaggle after training)
#    saved_models/tomato_efficientnetb0_v1_<timestamp>.keras
#    saved_models/registry.json
#    training/configs/model_config_v1_<timestamp>.yaml

# 5. Configure environment
cp .env.example .env
# Edit .env with your model filename and optional GitHub token

# 6. Run the API
uvicorn app.main:app --reload
```

API runs at `http://localhost:8000`
Swagger UI at `http://localhost:8000/docs`

---

## Configuration (.env)

```dotenv
# Model
MODEL_FILE=tomato_efficientnetb0_v1_<timestamp>.keras
SAVED_MODELS_DIR=saved_models
CONFIG_FILE=training/configs/model_config_v1_<timestamp>.yaml

# Thresholds
THRESHOLD_CONFIDENT=0.60
THRESHOLD_UNCERTAIN=0.30

# Storage
PREDICTIONS_LOG=data/predictions.jsonl
PREDICTIONS_IMAGES_DIR=data/images

# Report mode: "static" (default) or "agent" (requires GITHUB_TOKEN)
REPORT_MODE=static
GITHUB_TOKEN=

# Environment
APP_ENV=development
```

---

## Report Modes

**Static mode** (`REPORT_MODE=static`) - hardcoded agronomic summaries and treatment steps for all 8 disease classes. Works offline, no API key needed.

**Agent mode** (`REPORT_MODE=agent`) - uses `gpt-4o-mini` via GitHub Models to generate dynamic, contextual reports. Requires a GitHub personal access token.

---

## Training

The model was trained on Kaggle using the notebook at `training/train_plantdoc_tomato.ipynb`.

**Datasets:**
- [PlantDoc](https://www.kaggle.com/datasets/nirmalsankalana/plantdoc-dataset) - 901 tomato images (real field photos)
- [PlantVillage](https://www.kaggle.com/datasets/emmarex/plantdisease) - 12,931 tomato images (controlled lab photos)

**Training strategy:**
- Phase 1: Train classification head only (base frozen) - 20 epochs
- Phase 2: Fine-tune top 30 layers of EfficientNetB0 - 20 epochs
- Balanced class weights to handle imbalance
- EarlyStopping + ReduceLROnPlateau callbacks

---

## Testing

```bash
# Make sure uvicorn is running first
python test_api.py
```

The test suite covers all endpoints, edge cases, storage, versioning, and performance - 60+ checks.

---

## Dataset Strategy

See [DATASET_STRATEGY.md](DATASET_STRATEGY.md) for the full data collection plan, annotation guidelines, quality control process, and continuous improvement schema.