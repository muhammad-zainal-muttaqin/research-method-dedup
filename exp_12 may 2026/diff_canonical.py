"""Find which tree differs between harness load and canonical benchmark for M01."""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

canon = pd.read_csv(BASE / "reports" / "dedup_brand_new_953" / "per_tree.csv")
here = pd.read_csv(BASE / "exp_12 may 2026" / "out" / "per_tree_step01_baseline.csv")
NAMES = ["B1", "B2", "B3", "B4"]
cols = [f"M01_selector_b2b3_{c}" for c in NAMES]
m = pd.merge(
    canon[["tree_id"] + cols],
    here[["tree_id"] + cols],
    on="tree_id",
    suffixes=("_canon", "_here"),
)
for c in NAMES:
    m[f"d_{c}"] = m[f"M01_selector_b2b3_{c}_canon"] - m[f"M01_selector_b2b3_{c}_here"]
diff = m[(m["d_B1"] != 0) | (m["d_B2"] != 0) | (m["d_B3"] != 0) | (m["d_B4"] != 0)]
print(f"Diff trees: {len(diff)}")
print(diff.head(30).to_string(index=False))
