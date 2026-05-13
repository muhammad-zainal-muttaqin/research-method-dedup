"""
Step 4 — calibrate a side-aware dedup factor on the train split only, then
evaluate on val and test.

Hypothesis: the per-class duplication factor scales with the number of photo
sides taken around the tree. M01's adaptive_corrected uses only n_total and
clips at 1.45, which under-divides for 8-side trees.

Calibration rule:
  base_factor[n_sides][class] = median(naive[class] / gt[class]) over training
                                trees with that many sides and gt>0.

Applied prediction:
  unique[class] = round(naive[class] / base_factor[n_sides][class])

This is a deterministic table lookup — no gradient, no embedding, no learned
threshold. Parameters live in a CSV checked into this exp folder and computed
from the train split only.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(EXP))

from harness import NAMES, OUT_DIR, load_trees, naive_count


def calibrate(train_trees) -> pd.DataFrame:
    """Per (n_sides, class) median(naive/gt) on training trees with gt>0."""
    bucket = defaultdict(lambda: defaultdict(list))
    for t in train_trees:
        n_sides = len({d["side_index"] for d in t.dets}) if t.dets else 0
        naive = naive_count(t.dets)
        for c in NAMES:
            if t.gt[c] > 0:
                bucket[n_sides][c].append(naive[c] / t.gt[c])
    rows = []
    for n_sides, by_c in sorted(bucket.items()):
        for c in NAMES:
            vals = by_c.get(c, [])
            if not vals:
                continue
            rows.append({"n_sides": n_sides, "class": c, "n": len(vals),
                         "median_ratio": float(np.median(vals)),
                         "mean_ratio": float(np.mean(vals))})
    return pd.DataFrame(rows)


def main():
    trees = load_trees()
    train = [t for t in trees if t.split == "train"]
    val = [t for t in trees if t.split == "val"]
    test = [t for t in trees if t.split == "test"]
    other = [t for t in trees if t.split not in ("train", "val", "test")]
    print(f"split sizes: train={len(train)} val={len(val)} test={len(test)} unknown={len(other)}")

    table = calibrate(train)
    table.to_csv(OUT_DIR / "side_factor_table.csv", index=False)
    print("\nCalibrated side factor table (median naive/gt on TRAIN only):")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
