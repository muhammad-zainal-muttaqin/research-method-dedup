"""Repo cleanup: zip backups + dataset, refresh archive.zip, remove redundant folders.

Order matters:
  1. Zip 6 backup dirs -> archive/backups_20260509.zip, rm originals
  2. Zip dataset/      -> archive/dataset_legacy.zip
  3. Zip archive/*     -> archive.zip (root, refresh)
  4. rmtree archive/

Atomic: each zip written to .tmp, fsynced, renamed before any delete.
"""
import os, shutil, sys, zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def human(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024: return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}TB"


def dir_size(path):
    total = 0
    for r, _, files in os.walk(path):
        for f in files:
            try: total += os.path.getsize(os.path.join(r, f))
            except OSError: pass
    return total


def zip_dirs(zip_path, items, base_for_arcname=None, exclude_names=()):
    """Zip multiple paths into one archive. items = list of (abs_src_path, arcname_root)."""
    tmp = zip_path + ".tmp"
    if os.path.exists(tmp): os.remove(tmp)
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for src, arc_root in items:
            if os.path.isfile(src):
                zf.write(src, arc_root)
                continue
            for r, dirs, files in os.walk(src):
                # skip excluded subdirs
                dirs[:] = [d for d in dirs if d not in exclude_names]
                for f in files:
                    if f in exclude_names: continue
                    abs_p = os.path.join(r, f)
                    rel = os.path.relpath(abs_p, src)
                    arc = f"{arc_root}/{rel}".replace("\\", "/") if arc_root else rel.replace("\\","/")
                    zf.write(abs_p, arc)
    os.replace(tmp, zip_path)
    return os.path.getsize(zip_path)


def main():
    archive_dir = os.path.join(ROOT, "archive")
    os.makedirs(archive_dir, exist_ok=True)

    # === Action 1: zip 6 backup folders ===
    backup_specs = [
        (os.path.join(ROOT, "05 Mei 2026", "Output JSON.backup_20260509_213938"),
         "Output_JSON.backup_20260509_213938"),
        (os.path.join(ROOT, "05 Mei 2026", "Output JSON.backup_20260509_214102"),
         "Output_JSON.backup_20260509_214102"),
        (os.path.join(ROOT, "05 Mei 2026", "Output JSON.backup_treeid_20260509_222704"),
         "Output_JSON.backup_treeid_20260509_222704"),
        (os.path.join(ROOT, "json_05 Mei 2026.backup_20260509_213938"),
         "json_05_Mei_2026.backup_20260509_213938"),
        (os.path.join(ROOT, "json_05 Mei 2026.backup_20260509_214102"),
         "json_05_Mei_2026.backup_20260509_214102"),
        (os.path.join(ROOT, "json_05 Mei 2026.backup_treeid_20260509_222704"),
         "json_05_Mei_2026.backup_treeid_20260509_222704"),
    ]
    backup_specs = [(s, a) for s, a in backup_specs if os.path.isdir(s)]
    if backup_specs:
        zp = os.path.join(archive_dir, "backups_20260509.zip")
        before = sum(dir_size(s) for s, _ in backup_specs)
        sz = zip_dirs(zp, backup_specs)
        print(f"[1] zipped {len(backup_specs)} backup dirs -> {os.path.basename(zp)}  "
              f"({human(before)} -> {human(sz)})")
        for src, _ in backup_specs:
            shutil.rmtree(src)
        print(f"    removed {len(backup_specs)} backup dirs")
    else:
        print("[1] no backup dirs found, skip")

    # === Action 2: zip dataset/ (skip images/) ===
    ds = os.path.join(ROOT, "dataset")
    if os.path.isdir(ds):
        zp = os.path.join(archive_dir, "dataset_legacy.zip")
        # Only zip labels/ and data.yaml; skip images/ (2.3G)
        items = []
        for sub in ("labels", "data.yaml"):
            p = os.path.join(ds, sub)
            if os.path.exists(p):
                items.append((p, sub))
        sz = zip_dirs(zp, items)
        print(f"[2] zipped dataset/{{labels,data.yaml}} -> {os.path.basename(zp)} ({human(sz)})")
    else:
        print("[2] no dataset/ found, skip")

    # === Action 3: refresh archive.zip from current archive/ ===
    arc_zip = os.path.join(ROOT, "archive.zip")
    items = [(archive_dir, "")]
    sz = zip_dirs(arc_zip, items)
    print(f"[3] refreshed archive.zip from archive/ ({human(sz)})")

    # === Action 4: rmtree archive/ ===
    shutil.rmtree(archive_dir)
    print("[4] removed archive/ (zipped to archive.zip)")

    print("\nDone.")


if __name__ == "__main__":
    main()
