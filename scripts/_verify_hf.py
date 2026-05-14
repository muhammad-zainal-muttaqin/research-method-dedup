"""
Verifikasi dan sinkronisasi dataset lokal vs HuggingFace.
HF = ground truth. File yang berbeda SHA256 akan di-overwrite dari HF.
Kemudian download images/ jika belum ada.

Run dari workspace root:
    python scripts/_verify_hf.py
"""

import hashlib
import os
import sys
from pathlib import Path

HF_TOKEN  = "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"
HF_REPO   = "ULM-DS-Lab/OilPalm-MultiView-BunchCount-YOLO"
REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET   = REPO_ROOT / "Brand-New-Dataset-YOLO"

# File/folder yang perlu diverifikasi (bukan images — itu didownload terpisah)
VERIFY_PATTERNS = ["json/", "labels/", "train.txt", "val.txt", "test.txt",
                   "split_manifest.csv", "data.yaml", "croissant.json",
                   "data/ground_truth.parquet"]

EXPECTED_IMAGES = 3992


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    try:
        from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    except ImportError:
        print("[ERROR] huggingface_hub tidak terinstall. Jalankan: pip install huggingface_hub")
        sys.exit(1)

    api = HfApi(token=HF_TOKEN)

    print(f"[INFO] Mengambil daftar file dari HF repo: {HF_REPO}")
    print("       (satu API call — list_repo_tree recursive)")

    # Satu API call untuk semua file + metadata (termasuk sha256)
    all_files = list(api.list_repo_tree(
        repo_id=HF_REPO,
        repo_type="dataset",
        recursive=True,
        expand=True,  # include lfs info + sha256
    ))

    print(f"[INFO] Total file di HF: {len(all_files)}")

    # Build map: relative_path -> sha256
    hf_map = {}
    for item in all_files:
        if hasattr(item, "path") and hasattr(item, "lfs") and item.lfs:
            hf_map[item.path] = item.lfs.sha256
        elif hasattr(item, "path") and hasattr(item, "blob_id"):
            hf_map[item.path] = item.blob_id  # blob sha for small files

    # Pisahkan images dari non-images
    image_files = [p for p in hf_map if p.startswith("images/")]
    non_image_files = [p for p in hf_map if not p.startswith("images/")]
    print(f"[INFO] File non-images: {len(non_image_files)} | Images: {len(image_files)}")

    # --- Verifikasi file non-images ---
    print("\n[STEP 1] Verifikasi file non-images vs lokal ...")
    same, different, missing, skipped = 0, [], [], 0

    for rel_path in sorted(non_image_files):
        local_path = DATASET / rel_path
        if not local_path.exists():
            missing.append(rel_path)
            continue

        hf_sha = hf_map.get(rel_path)
        if not hf_sha:
            skipped += 1
            continue

        local_sha = sha256_file(local_path)
        if local_sha.lower() == hf_sha.lower():
            same += 1
        else:
            different.append(rel_path)

    print(f"  Sama   : {same}")
    print(f"  Berbeda: {len(different)}")
    print(f"  Missing: {len(missing)}")
    print(f"  Skip   : {skipped}")

    files_to_download = different + missing
    if files_to_download:
        print(f"\n[STEP 2] Download {len(files_to_download)} file yang berbeda/missing dari HF ...")
        for rel_path in files_to_download:
            print(f"  -> {rel_path}")
            local_dest = DATASET / rel_path
            local_dest.parent.mkdir(parents=True, exist_ok=True)
            hf_hub_download(
                repo_id=HF_REPO,
                repo_type="dataset",
                filename=rel_path,
                token=HF_TOKEN,
                local_dir=str(DATASET),
                local_dir_use_symlinks=False,
            )
        print(f"[OK] {len(files_to_download)} file di-overwrite dari HF.")
    else:
        print("\n[OK] Semua file non-images identik dengan HF.")

    # --- Download images jika belum ada ---
    print("\n[STEP 3] Cek images/ ...")
    img_dir = DATASET / "images"
    existing_imgs = list(img_dir.glob("*.jpg")) if img_dir.exists() else []
    print(f"  Images lokal: {len(existing_imgs)} / {EXPECTED_IMAGES}")

    if len(existing_imgs) < EXPECTED_IMAGES:
        print(f"[INFO] Download images dari HF (~2.3 GB, {EXPECTED_IMAGES} file) ...")
        print("       snapshot_download hanya download file yang belum ada (idempotent).")
        snapshot_download(
            repo_id=HF_REPO,
            repo_type="dataset",
            local_dir=str(DATASET),
            allow_patterns=["images/*"],
            token=HF_TOKEN,
            local_dir_use_symlinks=False,
        )
        # Verifikasi ulang
        existing_imgs = list(img_dir.glob("*.jpg"))
        if len(existing_imgs) >= EXPECTED_IMAGES:
            print(f"[OK] Images selesai: {len(existing_imgs)} file.")
        else:
            print(f"[WARN] Images masih kurang: {len(existing_imgs)} / {EXPECTED_IMAGES}")
    else:
        print(f"[OK] Images sudah lengkap: {len(existing_imgs)} file.")

    print("\n[DONE] Verifikasi dan sinkronisasi selesai.")


if __name__ == "__main__":
    main()
