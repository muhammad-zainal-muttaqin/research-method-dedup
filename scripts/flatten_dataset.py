"""Flatten images/ and labels/ from split subdirs into single flat dirs."""
import json
import shutil
from pathlib import Path

BASE = Path(__file__).parent.parent / "Brand-New-Dataset-YOLO"
SPLITS = ["train", "val", "test"]

# Move images
img_root = BASE / "images"
moved_imgs = 0
for split in SPLITS:
    split_dir = img_root / split
    for f in split_dir.glob("*.jpg"):
        dest = img_root / f.name
        if not dest.exists():
            shutil.move(str(f), str(dest))
            moved_imgs += 1
    # remove old metadata.jsonl in split dir
    old_meta = split_dir / "metadata.jsonl"
    if old_meta.exists():
        old_meta.unlink()
print(f"Moved {moved_imgs} images to images/")

# Move labels
lbl_root = BASE / "labels"
moved_lbls = 0
for split in SPLITS:
    split_dir = lbl_root / split
    for f in split_dir.glob("*.txt"):
        dest = lbl_root / f.name
        if not dest.exists():
            shutil.move(str(f), str(dest))
            moved_lbls += 1
print(f"Moved {moved_lbls} labels to labels/")

# Remove empty split dirs
for folder in [img_root, lbl_root]:
    for split in SPLITS:
        split_dir = folder / split
        if split_dir.exists():
            try:
                split_dir.rmdir()
                print(f"Removed {split_dir}")
            except OSError:
                remaining = list(split_dir.iterdir())
                print(f"WARNING: {split_dir} not empty: {remaining}")

# Regenerate flat metadata.jsonl
rows = []
for f in sorted(img_root.glob("*.jpg")):
    stem = f.stem
    tree_id, side = stem.rsplit("_", 1)
    rows.append({"file_name": f.name, "tree_id": tree_id, "side": int(side)})
meta_out = img_root / "metadata.jsonl"
with open(meta_out, "w", encoding="utf-8") as fp:
    for row in rows:
        fp.write(json.dumps(row) + "\n")
print(f"Written {len(rows)} entries -> images/metadata.jsonl")
