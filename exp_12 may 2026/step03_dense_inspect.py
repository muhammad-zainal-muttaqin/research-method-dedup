"""Inspect M01 behaviour on dense trees (n_dets > 40)."""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(EXP))

from harness import NAMES, OUT_DIR, load_trees, naive_count, max_per_side

df = pd.read_csv(OUT_DIR / "step02_all_trees.csv")
dense = df[df["n_dets"] > 25].sort_values("n_dets")

print(f"Trees with n_dets>25: {len(dense)}  within1: {dense['M01_within1'].sum()}")
cols = ["tree_id", "split", "n_dets", "n_sides_observed", "b3frac_naive",
        "naive_B1", "naive_B2", "naive_B3", "naive_B4",
        "gt_B1", "gt_B2", "gt_B3", "gt_B4",
        "M01_B1", "M01_B2", "M01_B3", "M01_B4",
        "M01_err_B1", "M01_err_B2", "M01_err_B3", "M01_err_B4",
        "M06_within1", "M07_within1", "M01_within1"]
print(dense[cols].to_string(index=False, max_colwidth=20))

# stats for very dense
vdense = df[df["n_dets"] > 40]
print(f"\n--- Very dense (>40 dets), n={len(vdense)} ---")
print(f"M01 bias B1={vdense['M01_err_B1'].mean():+.2f}  B2={vdense['M01_err_B2'].mean():+.2f}  "
      f"B3={vdense['M01_err_B3'].mean():+.2f}  B4={vdense['M01_err_B4'].mean():+.2f}")
print(f"Naive vs GT total ratio mean: "
      f"{(vdense[[f'naive_{c}' for c in NAMES]].sum(axis=1) / vdense[[f'gt_{c}' for c in NAMES]].sum(axis=1)).mean():.3f}")
print(f"M01 vs GT total ratio mean: "
      f"{(vdense[[f'M01_{c}' for c in NAMES]].sum(axis=1) / vdense[[f'gt_{c}' for c in NAMES]].sum(axis=1)).mean():.3f}")

# divisor analysis: for each dense tree, naive/gt by class
print("\nPer-class naive/gt ratio in dense (>40) trees:")
for c in NAMES:
    ratios = vdense[f"naive_{c}"] / vdense[f"gt_{c}"].replace(0, np.nan)
    print(f"  {c}: mean={ratios.mean():.3f} med={ratios.median():.3f} n={ratios.notna().sum()}")

# 25-40 bucket
print("\nPer-class naive/gt ratio in 25<n_dets<=40 trees:")
mid = df[(df["n_dets"] > 25) & (df["n_dets"] <= 40)]
for c in NAMES:
    ratios = mid[f"naive_{c}"] / mid[f"gt_{c}"].replace(0, np.nan)
    print(f"  {c}: mean={ratios.mean():.3f} med={ratios.median():.3f} n={ratios.notna().sum()}")
