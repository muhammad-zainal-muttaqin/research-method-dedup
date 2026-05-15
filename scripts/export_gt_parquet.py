"""Convert 953 JSON GT files → Parquet for HF Data Studio SQL queries."""
import json
import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent.parent / "Brand-New-Dataset-YOLO"
JSON_DIR = BASE / "json"
OUT_DIR = BASE / "data"
OUT_DIR.mkdir(exist_ok=True)

rows = []
for f in sorted(JSON_DIR.glob("*.json")):
    data = json.loads(f.read_text(encoding="utf-8-sig"))
    s = data.get("summary", {})
    by_class = s.get("by_class", {})
    meta = data.get("metadata", {})
    rows.append({
        "tree_id": data.get("tree_id", f.stem),
        "split": data.get("split", ""),
        "varietas": meta.get("varietas", ""),
        "num_sides": len(data.get("images", {})),
        "total_unique_bunches": s.get("total_unique_bunches", 0),
        "B1": by_class.get("B1", 0),
        "B2": by_class.get("B2", 0),
        "B3": by_class.get("B3", 0),
        "B4": by_class.get("B4", 0),
        "total_detections": s.get("total_detections", 0),
        "duplicates_linked": s.get("duplicates_linked", 0),
    })

df = pd.DataFrame(rows)
out = OUT_DIR / "ground_truth.parquet"
df.to_parquet(out, index=False)
print(f"Exported {len(df)} trees -> {out}")
print(df[["varietas", "split", "total_unique_bunches", "B1", "B2", "B3", "B4"]].describe().round(2))
