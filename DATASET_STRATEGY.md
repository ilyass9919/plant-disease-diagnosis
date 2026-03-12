# Dataset Strategy - Tomato Leaf Disease Diagnostic System

## 1. Data Collection Plan

### Sources Used
| Source | Type | Volume | License |
|--------|------|--------|---------|
| PlantDoc | Real field photos, multi-plant | 901 tomato images | Open research |
| PlantVillage | Controlled lab photos | 12,931 tomato images | CC0 Public Domain |
| **Total** | | **13,832 images** | |

### Volume Targets
- Minimum per class for reliable training: **500 images**
- Current weakest class: `Tomato_Mosaic_Virus` at 427 images (flagged)
- Target for next iteration: **1,000+ images per class** via field collection

### Future Field Collection Plan
- Partner with local agricultural extension offices and farming cooperatives
- Mobile app users submit photos with opt-in consent
- Target: 500 new field photos/month across all disease classes
- Consent: explicit in-app consent before any image is stored or used for training

---

## 2. Annotation Process

### Label Guidelines
- One label per image (single dominant disease visible)
- Minimum 70% of leaf surface must be visible
- Image must be taken within 30cm–100cm of the leaf
- Reject: blurry (motion blur), night shots, non-tomato plants

### Annotation Tooling
- **Current**: PlantDoc and PlantVillage provide pre-labeled data with verified labels from plant pathologists
- **Future (field photos)**: Label Studio (open source) with 2-annotator consensus
  - Annotator 1 labels → Annotator 2 reviews → Disagreements go to agronomist arbitration

### Label Schema
```json
{
  "image_id": "uuid",
  "canonical_class": "Tomato_Bacterial_Spot",
  "source_class": "Tomato_Bacterial_spot",       // original dataset label
  "source_dataset": "PlantVillage",
  "annotator": "plantvillage_verified",
  "confidence": "high",                          // high / medium / low
  "notes": ""
}
```

---

## 3. Quality Control

### Automated Checks (run at ingestion time)
| Check | Method | Action on Failure |
|-------|--------|-------------------|
| Blurry images | Laplacian variance < 50 | Reject |
| Wrong file format | PIL open + verify | Reject |
| Corrupt files | PIL.verify() | Reject |
| Duplicate images | MD5 hash comparison | Keep one, discard duplicates |
| Wrong aspect ratio | Ratio outside 0.3-3.0 | Flag for manual review |
| Non-leaf images | Colour histogram check (green channel) | Flag for manual review |

### Bias Checks
- **Class imbalance**: monitor per-class counts at each dataset build; flag if any class < 50% of the median
- **Source bias**: track image source (PlantVillage = controlled lab, PlantDoc = field) to ensure field photos are not underrepresented
- **Geographic bias**: log geolocation metadata on field-collected images to detect regional gaps

### Current Known Issues
- `Tomato_Mosaic_Virus`: only 427 images - flagged for priority collection
- PlantVillage images are lab/controlled; real field performance may vary (observed in testing)
- No data from North Africa or Middle East farming conditions currently

---

## 4. Storage for Continuous Improvement

### What is stored per prediction
Every API call logs the following to `data/predictions.jsonl`:

```json
{
  "id": "uuid",
  "timestamp": "2026-03-10T16:45:00Z",
  "predicted_class": "Tomato_Bacterial_Spot",
  "confidence": 0.87,
  "status": "CONFIDENT",
  "top_k": [
    {"disease_class": "Tomato_Bacterial_Spot", "confidence": 0.87},
    {"disease_class": "Tomato_Septoria_Leaf_Spot", "confidence": 0.09}
  ],
  "model_version": "v1_20260309_1523",
  "inference_time_ms": 243.1,
  "image_filename": "photo_field.jpg",
  "human_review": null
}
```

### Human Review Workflow
1. Mobile app flags UNCERTAIN predictions → sent to agronomist support queue
2. Agronomist reviews image → sets `human_review` field:
```json
"human_review": {
  "reviewer": "agronomist_id",
  "reviewed_at": "2026-03-11T09:00:00Z",
  "correct_class": "Tomato_Bacterial_Spot",
  "model_was_correct": true,
  "notes": "Confident diagnosis, good image quality"
}
```
3. Validated records feed the next training iteration

### Model Version Linkage
- Every prediction record stores `model_version`
- `saved_models/registry.json` stores full metadata per version:
  - Architecture, training date, val accuracy
  - Per-class F1/precision/recall
  - Training dataset fingerprint (which split, how many images)
  - Confidence thresholds used
- This allows tracing any prediction back to the exact model and dataset version that produced it

### Training/Validation Split Tracking
- Fixed `SEED=42` ensures reproducible train/val splits
- Split strategy recorded in `model_config.yaml` per version
- Val set is never used for training (enforced by `image_dataset_from_directory` with fixed seed)

---

## 5. Dataset Classes - Final Mapping

| Canonical Class | PlantDoc Source | PlantVillage Source | Total |
|----------------|----------------|--------------------|----|
| Tomato_Bacterial_Spot | Tomato_leaf_bacterial_spot | Tomato_Bacterial_spot | 2,237 |
| Tomato_Early_Blight | Tomato_Early_blight_leaf | Tomato_Early_blight | 1,088 |
| Tomato_Healthy | Tomato_leaf | Tomato_healthy | 1,643 |
| Tomato_Late_Blight | Tomato_leaf_late_blight | Tomato_Late_blight | 2,020 |
| Tomato_Leaf_Mold | Tomato_mold_leaf | Tomato_Leaf_Mold | 1,043 |
| Tomato_Mosaic_Virus | Tomato_leaf_mosaic_virus | Tomato__Tomato_mosaic_virus | 427 |
| Tomato_Septoria_Leaf_Spot | Tomato_Septoria_leaf_spot | Tomato_Septoria_leaf_spot | 1,928 |
| Tomato_Yellow_Leaf_Curl | Tomato_leaf_yellow_virus | Tomato__Tomato_YellowLeaf__Curl_Virus | 3,446 |
| **TOTAL** | **901** | **12,931** | **13,832** |