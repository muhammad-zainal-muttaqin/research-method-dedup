"""
Build a 3-D divisor table (n_sides, class, naive-count bucket) calibrated on
TRAIN only. Cells below MIN_SUPPORT fall back to the (n_sides, class) median.

The output CSV is the only parameter introduced — frozen, train-only,
deterministic. No learning, no gradient.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(EXP))

from harness import NAMES, OUT_DIR, load_trees, naive_count
from methods import n_sides_observed


# Bucket edges chosen for cleanliness only (powers-of-two-ish); calibrated on
# train. NO bin boundary was tuned on val/test.
COUNT_BINS = [0, 3, 6, 10, 15, 25, 1000]
MIN_SUPPORT = 12  # cells below this fall back to the 2-D (n_sides, class) median


def bucket_index(n: int) -> int:
    for i in range(len(COUNT_BINS) - 1):
        if COUNT_BINS[i] < n <= COUNT_BINS[i + 1]:
            return i
    return len(COUNT_BINS) - 2


def main() -> None:
    trees = load_trees()
    rows = []
    for t in trees:
        if t.split != "train":
            continue
        ns = n_sides_observed(t.dets)
        naive = naive_count(t.dets)
        for c in NAMES:
            if t.gt[c] > 0 and naive[c] > 0:
                rows.append({
                    "n_sides": ns,
                    "class": c,
                    "naive_bucket": bucket_index(naive[c]),
                    "ratio": naive[c] / t.gt[c],
                })
    df = pd.DataFrame(rows)

    # 3-D
    g3 = df.groupby(["n_sides", "class", "naive_bucket"])["ratio"].agg(["median", "mean", "count"]).reset_index()
    g3.to_csv(OUT_DIR / "divisor_3d.csv", index=False)

    # 2-D fallback
    g2 = df.groupby(["n_sides", "class"])["ratio"].agg(["median", "mean", "count"]).reset_index()
    g2.to_csv(OUT_DIR / "divisor_2d.csv", index=False)

    print("3-D table preview (only well-supported cells):")
    print(g3[g3["count"] >= MIN_SUPPORT].to_string(index=False))
    print("\n2-D fallback table:")
    print(g2.to_string(index=False))


if __name__ == "__main__":
    main()
