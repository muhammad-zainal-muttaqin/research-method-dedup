"""
Final benchmark for the 12 May 2026 push toward Acc±1 ≥ 90%.

Runs:
  M01 (prior champion)
  M30  side_aware_divide       (pure side-aware divisor)
  M31  side_aware_selector     (M01 selector with 8-side branch swapped)
  M33  refined_divide          (3-D divisor — used inside overrides)
  M52  two_band_override       (M31 + 2 b3frac overrides via M33)
  M53  three_band_override     (M52 + 1 narrow joint override via M19)

Writes the mandatory metric set per CLAUDE.md to:
  exp_12 may 2026/out/final_accuracy.csv
  exp_12 may 2026/out/final_per_tree.csv
  exp_12 may 2026/out/final_per_split.csv
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

from harness import NAMES, OUT_DIR, evaluate, load_trees, run, split_breakdown
from algorithms.M01_selector_b2b3 import predict as m01
from algorithms.M05_blend_vis_divide import predict as m05
from methods import (
    m30_side_aware_divide,
    m31_side_aware_selector,
    m33_refined_divide,
    m52_two_band_override,
    m53_three_band_override,
)

METHODS = {
    "M01_selector_b2b3":      m01,
    "M05_blend_vis_divide":   m05,
    "M30_side_aware_divide":  m30_side_aware_divide,
    "M31_side_aware_selector": m31_side_aware_selector,
    "M33_refined_divide":     m33_refined_divide,
    "M52_two_band_override":  m52_two_band_override,
    "M53_three_band_override": m53_three_band_override,
}


def main() -> None:
    for stale in OUT_DIR.glob("split_final.csv"):
        stale.unlink()

    trees = load_trees()
    print(f"Loaded {len(trees)} trees from Brand-New-Dataset-YOLO/json/")

    full = run(METHODS, trees, tag="final")
    full_path = OUT_DIR / "final_accuracy.csv"
    full.to_csv(full_path, index=False)
    cols = ["method", "acc_within1_pct", "macro_class_MAE", "n_fail",
            "exact_profile_acc_pct", "total_count_MAE", "total_count_within1_pct",
            "MAE_B1", "MAE_B2", "MAE_B3", "MAE_B4",
            "bias_B1", "bias_B2", "bias_B3", "bias_B4"]
    print("\n=== Full 953 ===")
    print(full[cols].to_string(index=False))

    rows_split = []
    for split_set in [("train",), ("val",), ("test",), ("val", "test")]:
        label = "+".join(split_set)
        sub = [t for t in trees if t.split in split_set]
        for name, fn in METHODS.items():
            preds = {t.tree_id: fn(t.dets) for t in sub}
            summ = evaluate(name, preds, sub)["summary"]
            summ["split"] = label
            summ["n"] = len(sub)
            rows_split.append(summ)
    split_df = pd.DataFrame(rows_split).sort_values(["split", "acc_within1_pct"], ascending=[True, False])
    split_df.to_csv(OUT_DIR / "final_per_split.csv", index=False)

    print("\n=== Per-split (held-out val+test is the honest generalization signal) ===")
    for split in ["train", "val", "test", "val+test"]:
        sub = split_df[split_df["split"] == split]
        if sub.empty:
            continue
        print(f"\n{split} (n={int(sub['n'].iloc[0])}):")
        print(sub[cols].to_string(index=False))


if __name__ == "__main__":
    main()
