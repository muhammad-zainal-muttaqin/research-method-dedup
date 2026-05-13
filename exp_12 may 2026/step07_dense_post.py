"""Inspect remaining 8-side / >40-dets failures under M31."""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(EXP))

df = pd.read_csv(EXP / "out" / "step06_all.csv")
fail = df[~df["M31_within1"]].copy()
dense = fail[fail["n_sides"] == 8].copy()
print(f"M31 fails @ 8 sides: {len(dense)}")
cols = ["tree_id", "split", "n_dets", "b3frac",
        "naive_B1", "naive_B2", "naive_B3", "naive_B4",
        "gt_B1", "gt_B2", "gt_B3", "gt_B4",
        "M31_B1", "M31_B2", "M31_B3", "M31_B4",
        "M31_err_B1", "M31_err_B2", "M31_err_B3", "M31_err_B4",
        "M30_sa_within1", "vis_within1", "adp_within1", "gmb_within1", "med_within1", "sid_within1"]
print(dense[cols].to_string(index=False, max_colwidth=22))

print("\n--- M31 fails in mid-density (16-25 dets), 4-side ---")
mid = fail[(fail["n_sides"] == 4) & (fail["n_dets"].between(16, 25))]
print(f"n={len(mid)}")
print(mid[cols].head(30).to_string(index=False, max_colwidth=22))
