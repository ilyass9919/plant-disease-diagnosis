"""
export_for_retraining.py — updated for Supabase + Cloudinary
"""
import json, os, sys, time, requests
from pathlib import Path
from datetime import datetime

SUPABASE_URL          = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY          = os.getenv("SUPABASE_KEY", "")
AGRONOMIST_PASSWORD   = os.getenv("AGRONOMIST_PASSWORD", "")
EXPORT_DIR            = Path("retrain_export")
USE_CLOUD             = bool(SUPABASE_URL and SUPABASE_KEY)


def export_for_retraining():
    print("=" * 60)
    print("LeafScan — Export Reviewed Data for Retraining")
    print("=" * 60)

    EXPORT_DIR.mkdir(exist_ok=True)
    export_images_dir = EXPORT_DIR / "images"
    export_images_dir.mkdir(exist_ok=True)

    # Fetch reviewed records 
    if USE_CLOUD:
        print("\nFetching reviewed predictions from Supabase...")
        from supabase import create_client
        db = create_client(SUPABASE_URL, SUPABASE_KEY)
        result = (
            db.table("predictions")
            .select("*")
            .not_.is_("human_review", "null")
            .execute()
        )
        reviewed = result.data or []
        all_result = db.table("predictions").select("id").execute()
        total = len(all_result.data or [])
    else:
        print("\nUsing local predictions.jsonl...")
        log_path = Path(os.getenv("PREDICTIONS_LOG", "data/predictions.jsonl"))
        if not log_path.exists():
            print("No predictions.jsonl found.")
            sys.exit(1)
        all_records = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
        reviewed    = [r for r in all_records if r.get("human_review") is not None]
        total       = len(all_records)

    print(f"\nTotal predictions : {total}")
    print(f"Reviewed          : {len(reviewed)}")

    if not reviewed:
        print("\nNo reviewed predictions found.")
        sys.exit(1)

    # Export predictions.jsonl 
    export_log = EXPORT_DIR / "predictions.jsonl"
    export_log.write_text("\n".join(json.dumps(r) for r in reviewed) + "\n")
    print(f"\nExported {len(reviewed)} reviewed records → {export_log}")

    # Download images 
    print("\nDownloading images...")
    copied = missing = 0

    for record in reviewed:
        pred_id   = record["id"]
        image_url = record.get("image_url")
        dst_path  = export_images_dir / f"{pred_id}.jpg"

        if dst_path.exists():
            copied += 1
            continue

        if image_url and image_url.startswith("http"):
            # Download from Cloudinary
            try:
                r = requests.get(image_url, timeout=30)
                if r.status_code == 200:
                    dst_path.write_bytes(r.content)
                    copied += 1
                    time.sleep(0.1)
                else:
                    missing += 1
            except Exception as e:
                print(f"  ⚠️  Could not download {pred_id}: {e}")
                missing += 1
        else:
            # Try local fallback
            local = Path("data/images") / f"{pred_id}.jpg"
            if local.exists():
                import shutil
                shutil.copy2(local, dst_path)
                copied += 1
            else:
                missing += 1

    print(f"  Downloaded : {copied}")
    print(f"  Missing    : {missing}")

    # Summary 
    from collections import Counter
    class_counts  = Counter(
        r["human_review"]["correct_class"]
        for r in reviewed
        if r.get("human_review", {}).get("correct_class")
    )
    correct_count = sum(1 for r in reviewed if r.get("human_review", {}).get("model_was_correct"))

    print(f"\nCorrections by class:")
    for cls, cnt in sorted(class_counts.items()):
        print(f"  {cls:<40} {cnt:>4}  {'█' * cnt}")

    print(f"\nModel was correct  : {correct_count} / {len(reviewed)}")
    print(f"Model was wrong    : {len(reviewed) - correct_count} / {len(reviewed)}")

    summary = {
        "exported_at"      : datetime.utcnow().isoformat(),
        "total_predictions": total,
        "reviewed"         : len(reviewed),
        "images_downloaded": copied,
        "images_missing"   : missing,
        "class_distribution": dict(class_counts),
        "ready_for_retrain": copied >= 50,
    }
    (EXPORT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'=' * 60}")
    if copied < 50:
        print(f"Only {copied} images — need at least 50 to retrain.")
    else:
        print(f"{copied} images ready for retraining.")
        print(f"\nNext steps:")
        print(f"  1. Upload retrain_export/ to Kaggle dataset")
        print(f"  2. Run retrain_from_corrections.ipynb on Kaggle GPU")
        print(f"  3. Download new model → update Render + HuggingFace")

    print(f"\nExport saved to: {EXPORT_DIR.absolute()}")


if __name__ == "__main__":
    export_for_retraining()