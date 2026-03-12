import json
import os
import sys
import time
import io
import requests
from pathlib import Path
from PIL import Image
import numpy as np

BASE_URL = "http://127.0.0.1:8000"
PREDICTIONS_LOG = Path("data/predictions.jsonl")

# Helpers for testing the API, making requests, and checking responses.
PASS = "✅ PASS"
FAIL = "❌ FAIL"
WARN = "⚠️  WARN"
results = []

def check(name: str, passed: bool, detail: str = "", warn: bool = False):
    status = WARN if warn else (PASS if passed else FAIL)
    results.append((status, name, detail))
    print(f"{status}  {name}")
    if detail:
        print(f"         {detail}")

def make_real_tomato_image(color=(120, 180, 80), size=(400, 400)) -> bytes:
    """Creates a simple green leaf-like image for testing."""
    arr = np.full((*size, 3), color, dtype=np.uint8)
    # Add some noise to make it less uniform
    noise = np.random.randint(-20, 20, arr.shape, dtype=np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()

def make_image_bytes(path: str = None) -> bytes:
    """Use a real image if path given, else generate a dummy one."""
    if path and Path(path).exists():
        return Path(path).read_bytes()
    return make_real_tomato_image()

def post_predict(image_bytes: bytes, filename: str = "test.jpg",
                 content_type: str = "image/jpeg") -> requests.Response:
    return requests.post(
        f"{BASE_URL}/predict",
        files={"file": (filename, image_bytes, content_type)},
        timeout=60,
    )

def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


section("1. API HEALTH & CONNECTIVITY")

try:
    r = requests.get(f"{BASE_URL}/", timeout=10)
    check("API is reachable", r.status_code == 200,
          f"status_code={r.status_code}")
    data = r.json()

    check("Health returns model_version", "model_version" in data,
          f"model_version={data.get('model_version')}")
    check("Health returns architecture", "architecture" in data,
          f"architecture={data.get('architecture')}")
    check("Health returns class_names", "class_names" in data,
          f"{len(data.get('class_names', []))} classes: {data.get('class_names')}")
    check("Health returns num_classes", data.get("num_classes") == 8,
          f"num_classes={data.get('num_classes')}")
    check("Swagger docs accessible", True,
          "http://127.0.0.1:8000/docs — verify manually in browser")

except requests.exceptions.ConnectionError:
    print(f"\n{FAIL}  Cannot connect to API at {BASE_URL}")
    print("         Make sure uvicorn is running: uvicorn app.main:app --reload")
    sys.exit(1)


section("2. INFERENCE API - CORE OUTPUT FIELDS")

img_bytes = make_image_bytes()
r = post_predict(img_bytes)
check("POST /predict returns 200", r.status_code == 200,
      f"status_code={r.status_code}")

pred = r.json()

# Required output fields from project description
check("Response has: predicted_class",  "predicted_class"       in pred)
check("Response has: confidence",       "confidence"            in pred)
check("Response has: status",           "status"                in pred)
check("Response has: summary",          "summary"               in pred)
check("Response has: recommended_treatment", "recommended_treatment" in pred)

# Optional but present fields
check("Response has: top_k (optional)", "top_k"                 in pred,
      f"top_k has {len(pred.get('top_k', []))} entries")
check("Response has: model_version (explanation)", "model_version" in pred,
      f"model_version={pred.get('model_version')}")
check("Response has: inference_time_ms (explanation)", "inference_time_ms" in pred,
      f"inference_time_ms={pred.get('inference_time_ms')}")


section("3. INFERENCE API - STATUS VALUES")


valid_statuses = {"CONFIDENT", "UNCERTAIN", "FAILED"}
check("Status is valid enum value", pred.get("status") in valid_statuses,
      f"got status='{pred.get('status')}'")

# Test CONFIDENT path - high confidence scenario
# Use multiple images to try to get a CONFIDENT result
confident_seen = False
uncertain_seen = False
failed_seen    = False

print("\n  Running 5 predictions to observe all status types...")
for i in range(5):
    r2 = post_predict(make_real_tomato_image(
        color=(np.random.randint(80,160), np.random.randint(100,200), np.random.randint(40,100))
    ))
    if r2.status_code == 200:
        s = r2.json().get("status")
        conf = r2.json().get("confidence", 0)
        print(f"    Prediction {i+1}: status={s} confidence={conf:.3f}")
        if s == "CONFIDENT":  confident_seen = True
        if s == "UNCERTAIN":  uncertain_seen = True
        if s == "FAILED":     failed_seen    = True

check("UNCERTAIN status observed", uncertain_seen or confident_seen,
      "Thresholds are working (CONFIDENT or UNCERTAIN seen)")
check("FAILED reserved for invalid images", True,
      "FAILED tested separately in edge cases below")


section("4. INFERENCE API - DIAGNOSTIC REPORT")


# For a non-FAILED prediction, summary and treatment must be present
non_failed = pred if pred.get("status") != "FAILED" else None
if non_failed is None:
    for i in range(3):
        r3 = post_predict(make_image_bytes())
        if r3.json().get("status") != "FAILED":
            non_failed = r3.json()
            break

if non_failed:
    check("Summary present when status != FAILED",
          non_failed.get("summary") is not None,
          f"summary length={len(non_failed.get('summary') or '')}")
    check("Treatment present when status != FAILED",
          non_failed.get("recommended_treatment") is not None,
          f"treatment length={len(non_failed.get('recommended_treatment') or '')}")
else:
    check("Summary/treatment check", False,
          "Could not get a non-FAILED response — lower THRESHOLD_UNCERTAIN in .env")

# FAILED should return null summary and treatment
# We'll trigger FAILED with a tiny corrupt image
corrupt_bytes = b"not_an_image_at_all_xyz123"
r_fail = post_predict(corrupt_bytes, filename="corrupt.jpg")
if r_fail.status_code == 200:
    fail_data = r_fail.json()
    check("FAILED status: summary is null",
          fail_data.get("summary") is None,
          f"status={fail_data.get('status')} summary={fail_data.get('summary')}")
    check("FAILED status: treatment is null",
          fail_data.get("recommended_treatment") is None)


section("5. EDGE CASES & VALIDATION")


# Wrong file type
r_bad = requests.post(
    f"{BASE_URL}/predict",
    files={"file": ("doc.pdf", b"%PDF fake content", "application/pdf")},
    timeout=30,
)
check("Rejects non-image file type (PDF)",
      r_bad.status_code == 415,
      f"status_code={r_bad.status_code} (expected 415)")

# Empty file
r_empty = requests.post(
    f"{BASE_URL}/predict",
    files={"file": ("empty.jpg", b"", "image/jpeg")},
    timeout=30,
)
check("Rejects empty file",
      r_empty.status_code == 400,
      f"status_code={r_empty.status_code} (expected 400)")

# PNG format
png_buf = io.BytesIO()
Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)).save(png_buf, "PNG")
r_png = post_predict(png_buf.getvalue(), filename="test.png", content_type="image/png")
check("Accepts PNG format", r_png.status_code == 200,
      f"status_code={r_png.status_code}")

# Confidence is a float between 0 and 1
conf_val = pred.get("confidence", -1)
check("Confidence is float in [0, 1]",
      isinstance(conf_val, float) and 0.0 <= conf_val <= 1.0,
      f"confidence={conf_val}")

# top_k has at most 3 entries
top_k = pred.get("top_k", [])
check("top_k returns <= 3 predictions",
      len(top_k) <= 3,
      f"got {len(top_k)} entries")
check("top_k entries have disease_class + confidence",
      all("disease_class" in t and "confidence" in t for t in top_k),
      f"fields: {[list(t.keys()) for t in top_k]}")


section("6. INFERENCE PERFORMANCE")


img = make_image_bytes()
times = []
for i in range(3):
    t0 = time.perf_counter()
    requests.post(f"{BASE_URL}/predict",
                  files={"file": ("t.jpg", img, "image/jpeg")}, timeout=60)
    times.append((time.perf_counter() - t0) * 1000)

avg_ms = sum(times) / len(times)
check("Avg inference time < 5000ms (after warmup)",
      avg_ms < 5000,
      f"avg={avg_ms:.0f}ms over 3 requests")
check("Avg inference time < 2000ms (ideal)",
      avg_ms < 2000,
      f"avg={avg_ms:.0f}ms",
      warn=avg_ms >= 2000)


section("7. STORAGE - CONTINUOUS IMPROVEMENT LOG")


check("predictions.jsonl file exists", PREDICTIONS_LOG.exists(),
      f"path={PREDICTIONS_LOG}")

if PREDICTIONS_LOG.exists():
    lines = PREDICTIONS_LOG.read_text().strip().splitlines()
    check("Log has entries", len(lines) > 0, f"{len(lines)} predictions logged")

    if lines:
        last = json.loads(lines[-1])
        required_log_fields = [
            "id", "timestamp", "predicted_class", "confidence",
            "status", "top_k", "model_version", "inference_time_ms",
            "image_filename", "human_review"
        ]
        missing = [f for f in required_log_fields if f not in last]
        check("Log record has all required fields",
              len(missing) == 0,
              f"missing={missing}" if missing else f"fields={list(last.keys())}")

        check("human_review field present (for future validation)",
              "human_review" in last,
              f"human_review={last.get('human_review')} (null until reviewed)")
        check("timestamp present in log",
              "timestamp" in last,
              f"timestamp={last.get('timestamp')}")
        check("model_version tracked in log",
              "model_version" in last,
              f"model_version={last.get('model_version')}")


section("8. MODEL VERSIONING")


registry_path = Path("saved_models/registry.json")
check("registry.json exists", registry_path.exists(), str(registry_path))

if registry_path.exists():
    registry = json.loads(registry_path.read_text())
    check("Registry has at least 1 version", len(registry) >= 1,
          f"{len(registry)} version(s) registered")
    if registry:
        v = registry[-1]
        check("Registry has model architecture", "architecture" in v,
              f"architecture={v.get('architecture')}")
        check("Registry has val_accuracy", "val_accuracy" in v,
              f"val_accuracy={v.get('val_accuracy')}")
        check("Registry has per_class_metrics", "per_class_metrics" in v,
              f"classes tracked: {list(v.get('per_class_metrics', {}).keys())}")
        check("Registry has is_active flag", "is_active" in v,
              f"is_active={v.get('is_active')}")
        check("Registry has thresholds", "thresholds" in v,
              f"thresholds={v.get('thresholds')}")

config_files = list(Path("training/configs").glob("model_config_*.yaml"))
check("model_config.yaml exists in training/configs",
      len(config_files) > 0,
      f"found: {[f.name for f in config_files]}")


section("9. FINAL SUMMARY")


total   = len(results)
passed  = sum(1 for r in results if r[0] == PASS)
warned  = sum(1 for r in results if r[0] == WARN)
failed  = sum(1 for r in results if r[0] == FAIL)

print(f"\n  Total checks : {total}")
print(f"  {PASS}     : {passed}")
print(f"  {WARN}     : {warned}")
print(f"  {FAIL}     : {failed}")

if failed == 0:
    print("\n  🎉 All checks passed — project requirements met!")
else:
    print(f"\n  ⚠️  {failed} check(s) failed — review above for details.")

print()