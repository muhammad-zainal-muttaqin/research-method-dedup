"""
One-shot dataset setup untuk reproduksi di device baru.

Unduh Brand-New-Dataset-YOLO/images/ dari HuggingFace (~2.3 GB, 3993 .jpg).
Labels, JSON GT, split files sudah ter-track di git — hanya images yang
gitignored (heavy binary).

Idempotent: kalau images/ sudah lengkap, skip download dan langsung verify.

Run dari workspace root:
    python scripts/setup_dataset.py

Exit 0 kalau dataset siap pakai. Exit 1 kalau verifikasi gagal.
"""

import sys
from pathlib import Path

BASE         = Path(__file__).resolve().parent.parent
DATASET_ROOT = BASE / "Brand-New-Dataset-YOLO"
IMG_DIR      = DATASET_ROOT / "images"
LABEL_DIR    = DATASET_ROOT / "labels"
JSON_DIR     = DATASET_ROOT / "json"
SPLIT_FILES  = ["train.txt", "val.txt", "test.txt"]

HF_REPO       = "ULM-DS-Lab/OilPalm-MultiView-BunchCount-YOLO"
EXPECTED_IMG  = 3992
EXPECTED_LBL  = 3992
EXPECTED_JSON = 953


def count_files(directory: Path, pattern: str) -> int:
    if not directory.exists():
        return 0
    return sum(1 for _ in directory.glob(pattern))


def verify_layout() -> tuple[bool, dict]:
    counts = {
        "images": count_files(IMG_DIR, "*.jpg"),
        "labels": count_files(LABEL_DIR, "*.txt"),
        "json":   count_files(JSON_DIR, "*.json"),
    }
    splits_present = all((DATASET_ROOT / s).exists() for s in SPLIT_FILES)
    manifest_present = (DATASET_ROOT / "split_manifest.csv").exists()

    ok = (
        counts["images"] == EXPECTED_IMG
        and counts["labels"] == EXPECTED_LBL
        and counts["json"]   == EXPECTED_JSON
        and splits_present
        and manifest_present
    )
    return ok, {**counts, "splits_present": splits_present, "manifest_present": manifest_present}


def download_from_hf() -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[ERROR] huggingface_hub tidak terinstall.")
        print("        Jalankan: pip install huggingface_hub")
        sys.exit(1)

    print(f"[INFO] Unduh {HF_REPO} -> {DATASET_ROOT} ...")
    snapshot_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        local_dir=str(DATASET_ROOT),
        local_dir_use_symlinks=False,
    )
    print("[INFO] Unduh selesai.")


def main() -> int:
    print(f"[INFO] Dataset root: {DATASET_ROOT}")

    ok, counts = verify_layout()
    if ok:
        print("[OK] Dataset sudah lengkap, skip download.")
        print(f"     images={counts['images']}  labels={counts['labels']}  json={counts['json']}")
        return 0

    print("[INFO] Dataset belum lengkap:")
    for k, v in counts.items():
        print(f"       {k}: {v}")

    img_count = counts["images"]
    if img_count < EXPECTED_IMG:
        download_from_hf()
    else:
        print("[WARN] Images sudah lengkap tapi labels/json/splits kurang.")
        print("       Cek apakah git repo ter-clone penuh (labels & json ter-track).")

    print("[INFO] Verifikasi ulang ...")
    ok, counts = verify_layout()
    if ok:
        print("[OK] Dataset siap pakai.")
        print(f"     images={counts['images']}  labels={counts['labels']}  json={counts['json']}")
        return 0

    print("[FAIL] Verifikasi gagal setelah download:")
    for k, v in counts.items():
        print(f"       {k}: {v}")
    print(f"       Expected images={EXPECTED_IMG} labels={EXPECTED_LBL} json={EXPECTED_JSON}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
