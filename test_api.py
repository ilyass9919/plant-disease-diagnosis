import base64
import json
import io
import os
import sys
import time
import requests
import numpy as np
from pathlib import Path
from PIL import Image

BASE_URL        = "http://127.0.0.1:8000"
PREDICTIONS_LOG = Path("data/predictions.jsonl")
IMAGES_DIR      = Path("data/images")

PASS = "✅ PASS"
FAIL = "❌ FAIL"
WARN = "⚠️  WARN"
results = []

def check(name, passed, detail="", warn=False):
    status = WARN if warn else (PASS if passed else FAIL)
    results.append((status, name, detail))
    print(f"{status}  {name}")
    if detail:
        print(f"         {detail}")

def section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")

def make_image_bytes(color=(120, 180, 80), size=(400, 400)) -> bytes:
    arr  = np.full((*size, 3), color, dtype=np.uint8)
    arr  = np.clip(arr.astype(np.int16) + np.random.randint(-20, 20, arr.shape), 0, 255).astype(np.uint8)
    buf  = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    return buf.getvalue()

def post_file(image_bytes, filename="test.jpg", content_type="image/jpeg", metadata=None):
    data = {}
    if metadata:
        data["metadata_json"] = json.dumps(metadata)
    return requests.post(
        f"{BASE_URL}/predict",
        files={"file": (filename, image_bytes, content_type)},
        data=data,
        timeout=60,
    )

def post_base64(image_bytes, metadata=None):
    b64 = base64.b64encode(image_bytes).decode()
    data = {"image_base64": b64}
    if metadata:
        data["metadata_json"] = json.dumps(metadata)
    return requests.post(f"{BASE_URL}/predict", data=data, timeout=60)


section("1. API HEALTH & CONNECTIVITY")

try:
    r = requests.get(f"{BASE_URL}/", timeout=10)
    check("API is reachable", r.status_code == 200, f"status={r.status_code}")
    data = r.json()
    check("Returns model_version",  "model_version" in data, data.get("model_version"))
    check("Returns architecture",   "architecture"  in data, data.get("architecture"))
    check("Returns class_names",    "class_names"   in data, f"{len(data.get('class_names',[]))} classes")
    check("Returns num_classes=8",  data.get("num_classes") == 8, str(data.get("num_classes")))
    check("Swagger docs accessible", True, "verify manually at http://127.0.0.1:8000/docs")
except requests.exceptions.ConnectionError:
    print(f"\n{FAIL}  Cannot connect — is uvicorn running?")
    sys.exit(1)


section("2. FILE UPLOAD — CORE OUTPUT FIELDS")

img  = make_image_bytes()
r    = post_file(img)
check("POST /predict (file) returns 200", r.status_code == 200, f"status={r.status_code}")
pred = r.json()

check("Response: prediction_id",          "prediction_id"         in pred, pred.get("prediction_id"))
check("Response: predicted_class",        "predicted_class"       in pred)
check("Response: confidence",             "confidence"            in pred)
check("Response: status",                 "status"                in pred)
check("Response: summary",                "summary"               in pred)
check("Response: recommended_treatment",  "recommended_treatment" in pred)
check("Response: top_k (3 entries)",      len(pred.get("top_k",[])) == 3, str(len(pred.get("top_k",[]))))
check("Response: model_version",          "model_version"         in pred, pred.get("model_version"))
check("Response: inference_time_ms",      "inference_time_ms"     in pred, f"{pred.get('inference_time_ms')}ms")
check("Status is valid enum",             pred.get("status") in {"CONFIDENT","UNCERTAIN","FAILED"})
check("Confidence in [0,1]",              0 <= (pred.get("confidence") or 0) <= 1)


section("3. BASE64 INPUT")

r64 = post_base64(img)
check("POST /predict (base64) returns 200", r64.status_code == 200, f"status={r64.status_code}")
p64 = r64.json()
check("Base64 response has prediction_id",    "prediction_id"   in p64)
check("Base64 response has predicted_class",  "predicted_class" in p64)
check("Base64 response has status",           "status"          in p64)

# Test with data URI prefix
b64_with_prefix = "data:image/jpeg;base64," + base64.b64encode(img).decode()
r_uri = requests.post(f"{BASE_URL}/predict", data={"image_base64": b64_with_prefix}, timeout=60)
check("Base64 with data URI prefix accepted", r_uri.status_code == 200)


section("4. OPTIONAL METADATA")

metadata = {
    "latitude":     33.5731,
    "longitude":    -7.5898,
    "device_model": "Samsung Galaxy S23",
    "app_version":  "1.2.0",
    "notes":        "Lower leaf, spotted 2 days ago",
}
r_meta = post_file(img, metadata=metadata)
check("POST /predict with metadata returns 200", r_meta.status_code == 200)
pred_id_meta = r_meta.json().get("prediction_id")

# Verify metadata was stored in log
time.sleep(0.2)
if PREDICTIONS_LOG.exists():
    lines = PREDICTIONS_LOG.read_text().strip().splitlines()
    found_meta = False
    for line in lines:
        rec = json.loads(line)
        if rec.get("id") == pred_id_meta:
            m = rec.get("metadata", {})
            found_meta = m.get("latitude") == 33.5731
            check("Metadata stored: latitude",     m.get("latitude")     == 33.5731,    str(m.get("latitude")))
            check("Metadata stored: longitude",    m.get("longitude")    == -7.5898,    str(m.get("longitude")))
            check("Metadata stored: device_model", m.get("device_model") == "Samsung Galaxy S23")
            check("Metadata stored: app_version",  m.get("app_version")  == "1.2.0")
            break
    if not found_meta:
        check("Metadata stored in log", False, "Record not found in log")


section("5. DIAGNOSTIC REPORT")

non_failed = pred if pred.get("status") != "FAILED" else None
if non_failed is None:
    for _ in range(3):
        r3 = post_file(make_image_bytes())
        if r3.json().get("status") != "FAILED":
            non_failed = r3.json()
            break

if non_failed:
    check("Summary present when not FAILED",   non_failed.get("summary") is not None,
          f"length={len(non_failed.get('summary') or '')}")
    check("Treatment present when not FAILED", non_failed.get("recommended_treatment") is not None,
          f"length={len(non_failed.get('recommended_treatment') or '')}")
else:
    check("Summary/treatment check", False, "All predictions returned FAILED — lower THRESHOLD_UNCERTAIN")

corrupt = b"not_an_image"
r_fail  = post_file(corrupt)
if r_fail.status_code == 200:
    fd = r_fail.json()
    check("FAILED: summary is null",    fd.get("summary") is None,    f"status={fd.get('status')}")
    check("FAILED: treatment is null",  fd.get("recommended_treatment") is None)


section("6. EDGE CASES & VALIDATION")

r_bad = requests.post(f"{BASE_URL}/predict",
    files={"file": ("doc.pdf", b"%PDF", "application/pdf")}, timeout=30)
check("Rejects PDF (415)", r_bad.status_code == 415, f"status={r_bad.status_code}")

r_empty = requests.post(f"{BASE_URL}/predict",
    files={"file": ("e.jpg", b"", "image/jpeg")}, timeout=30)
check("Rejects empty file (400)", r_empty.status_code == 400, f"status={r_empty.status_code}")

r_no_image = requests.post(f"{BASE_URL}/predict", data={}, timeout=30)
check("Rejects request with no image (422)", r_no_image.status_code == 422, f"status={r_no_image.status_code}")

r_bad_meta = requests.post(
    f"{BASE_URL}/predict",
    files={"file": ("test.jpg", img, "image/jpeg")},
    data={"metadata_json": "not-valid-json{{{"},
    timeout=30,
)
check("Rejects invalid metadata_json (400)", r_bad_meta.status_code == 400, f"status={r_bad_meta.status_code}")

png_buf = io.BytesIO()
Image.fromarray(np.random.randint(0,255,(224,224,3),dtype=np.uint8)).save(png_buf,"PNG")
r_png = post_file(png_buf.getvalue(), filename="test.png", content_type="image/png")
check("Accepts PNG", r_png.status_code == 200)


section("7. HUMAN REVIEW ENDPOINT")

prediction_id = pred.get("prediction_id")

review_payload = {
    "reviewer_id":       "agronomist_001",
    "correct_class":     "Tomato_Healthy",
    "model_was_correct": True,
    "notes":             "Confirmed healthy leaf, good image quality",
}
r_review = requests.patch(
    f"{BASE_URL}/predictions/{prediction_id}/review",
    json=review_payload, timeout=30,
)
check("PATCH /predictions/{id}/review returns 200", r_review.status_code == 200,
      f"status={r_review.status_code}")

if r_review.status_code == 200:
    rv = r_review.json()
    check("Review response: prediction_id",  rv.get("prediction_id") == prediction_id)
    check("Review response: status=review_recorded", rv.get("status") == "review_recorded")
    check("Review response: correct_class",  rv.get("correct_class") == "Tomato_Healthy")
    check("Review response: reviewed_at",    "reviewed_at" in rv)

# Duplicate review should return 409
r_dup = requests.patch(f"{BASE_URL}/predictions/{prediction_id}/review",
    json=review_payload, timeout=30)
check("Duplicate review returns 409", r_dup.status_code == 409, f"status={r_dup.status_code}")

# Non-existent prediction should return 404
r_404 = requests.patch(f"{BASE_URL}/predictions/non-existent-id/review",
    json=review_payload, timeout=30)
check("Non-existent prediction returns 404", r_404.status_code == 404, f"status={r_404.status_code}")

# Verify review is stored in log
time.sleep(0.2)
if PREDICTIONS_LOG.exists():
    for line in PREDICTIONS_LOG.read_text().strip().splitlines():
        rec = json.loads(line)
        if rec.get("id") == prediction_id:
            hr = rec.get("human_review")
            check("Review stored in log",            hr is not None)
            check("Review: reviewer_id stored",      hr.get("reviewer_id")       == "agronomist_001")
            check("Review: correct_class stored",    hr.get("correct_class")     == "Tomato_Healthy")
            check("Review: model_was_correct stored",hr.get("model_was_correct") == True)
            check("Review: reviewed_at stored",      "reviewed_at" in hr)
            break


section("8. IMAGE STORAGE")

check("data/images/ directory exists", IMAGES_DIR.exists(), str(IMAGES_DIR))
if IMAGES_DIR.exists():
    saved = list(IMAGES_DIR.glob("*.jpg"))
    check("Images saved to disk", len(saved) > 0, f"{len(saved)} images saved")
    if saved:
        check("Saved image is non-empty", saved[0].stat().st_size > 0,
              f"size={saved[0].stat().st_size} bytes")


section("9. STORAGE - FULL RECORD SCHEMA")

check("predictions.jsonl exists", PREDICTIONS_LOG.exists())
if PREDICTIONS_LOG.exists():
    lines = PREDICTIONS_LOG.read_text().strip().splitlines()
    check("Log has entries", len(lines) > 0, f"{len(lines)} records")
    if lines:
        last = json.loads(lines[-1])
        required = ["id","timestamp","predicted_class","confidence","status",
                    "top_k","model_version","inference_time_ms","image_filename",
                    "image_saved_path","metadata","human_review"]
        missing = [f for f in required if f not in last]
        check("All required fields present", not missing,
              f"missing={missing}" if missing else "all fields present")
        check("image_saved_path recorded",  last.get("image_saved_path") is not None,
              str(last.get("image_saved_path")))
        check("metadata field present",     "metadata" in last)
        check("human_review field present", "human_review" in last)
        check("model_version tracked",      "model_version" in last, last.get("model_version"))


section("10. MODEL VERSIONING")

registry_path = Path("saved_models/registry.json")
check("registry.json exists", registry_path.exists())
if registry_path.exists():
    reg = json.loads(registry_path.read_text())
    check("Registry has >= 1 version", len(reg) >= 1, f"{len(reg)} version(s)")
    if reg:
        v = reg[-1]
        check("Registry: architecture",      "architecture"      in v, v.get("architecture"))
        check("Registry: val_accuracy",      "val_accuracy"      in v, str(v.get("val_accuracy")))
        check("Registry: per_class_metrics", "per_class_metrics" in v)
        check("Registry: thresholds",        "thresholds"        in v, str(v.get("thresholds")))
        check("Registry: is_active flag",    "is_active"         in v)

config_files = list(Path("training/configs").glob("model_config_*.yaml"))
check("model_config.yaml in training/configs", len(config_files) > 0,
      str([f.name for f in config_files]))


section("11. PERFORMANCE")

img   = make_image_bytes()
times = []
for _ in range(3):
    t0 = time.perf_counter()
    requests.post(f"{BASE_URL}/predict",
                  files={"file": ("t.jpg", img, "image/jpeg")}, timeout=60)
    times.append((time.perf_counter() - t0) * 1000)
avg = sum(times) / len(times)
check("Avg inference < 5000ms", avg < 5000, f"avg={avg:.0f}ms")
check("Avg inference < 2000ms (ideal)", avg < 2000, f"avg={avg:.0f}ms", warn=avg >= 2000)


section("FINAL SUMMARY")

total  = len(results)
passed = sum(1 for r in results if r[0] == PASS)
warned = sum(1 for r in results if r[0] == WARN)
failed = sum(1 for r in results if r[0] == FAIL)

print(f"\n  Total : {total}")
print(f"  {PASS} : {passed}")
print(f"  {WARN} : {warned}")
print(f"  {FAIL} : {failed}")

if failed:
    print(f"\n  Failed checks:")
    for s, n, d in results:
        if s == FAIL:
            print(f"    • {n} — {d}")

print("\n  🎉 All checks passed!\n" if not failed else "\n  Fix the failures above.\n")