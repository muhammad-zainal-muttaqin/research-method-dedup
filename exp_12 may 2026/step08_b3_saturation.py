"""
Investigate whether B3 dup-rate depends on B3 saturation per tree, on the
TRAIN split only. The hypothesis: when B3 is the dominant class, each side
captures more distinct B3 bunches, so naive/gt B3 ratio is smaller.

If a clean monotonic relation exists, we can refine the B3 divisor with a
density-aware lookup. Calibration data: train trees only. Any cutoff must
generalize to val+test without re-tuning.
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


def main() -> None:
    trees = load_trees()
    rows = []
    for t in trees:
        if t.split != "train":
            continue
        dets = t.dets
        ns = n_sides_observed(dets)
        naive = naive_count(dets)
        n_total = len(dets)
        for c in NAMES:
            if t.gt[c] > 0 and naive[c] > 0:
                rows.append({
                    "tree_id": t.tree_id,
                    "n_sides": ns,
                    "n_total": n_total,
                    "class": c,
                    "naive": naive[c],
                    "gt": t.gt[c],
                    "ratio": naive[c] / t.gt[c],
                    "class_frac": naive[c] / max(n_total, 1),
                })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "step08_per_class_ratios.csv", index=False)

    for ns_target in (4, 8):
        sub = df[df["n_sides"] == ns_target]
        print(f"\n=== n_sides={ns_target}  (n={len(sub)}) ===")
        for c in NAMES:
            cc = sub[sub["class"] == c]
            if len(cc) < 5:
                continue
            print(f"  {c}: n={len(cc)}, "
                  f"overall median={cc['ratio'].median():.3f}, mean={cc['ratio'].mean():.3f}")
            # Bin by class_frac
            cc = cc.copy()
            cc["frac_bin"] = pd.cut(cc["class_frac"], bins=[0, 0.2, 0.4, 0.55, 0.7, 0.85, 1.01])
            for b, g in cc.groupby("frac_bin"):
                if len(g) < 5:
                    continue
                print(f"    frac{b}: n={len(g)} median_ratio={g['ratio'].median():.3f} "
                      f"mean_ratio={g['ratio'].mean():.3f}")
            # Bin by naive count
            cc["count_bin"] = pd.cut(cc["naive"], bins=[0, 3, 6, 10, 15, 25, 60])
            for b, g in cc.groupby("count_bin"):
                if len(g) < 5:
                    continue
                print(f"    naive{b}: n={len(g)} median_ratio={g['ratio'].median():.3f} "
                      f"mean_ratio={g['ratio'].mean():.3f}")


if __name__ == "__main__":
    main()
