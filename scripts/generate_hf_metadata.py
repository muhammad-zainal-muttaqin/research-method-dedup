"""Generate metadata.jsonl for each split — enables HF ImageFolder Dataset Viewer."""
import json
from pathlib import Path

BASE = Path(__file__).parent.parent / "Brand-New-Dataset-YOLO"

for split in ["train", "val", "test"]:
    img_dir = BASE / "images" / split
    rows = []
    for f in sorted(img_dir.glob("*.jpg")):
        stem = f.stem  # e.g. DAMIMAS_A21B_0001_1
        tree_id, side = stem.rsplit("_", 1)
        rows.append({
            "file_name": f.name,
            "tree_id": tree_id,
            "side": int(side),
            "split": split,
        })
    out = img_dir / "metadata.jsonl"
    with open(out, "w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row) + "\n")
    print(f"{split}: {len(rows)} entries -> {out}")
