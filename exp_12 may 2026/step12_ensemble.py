"""
Build per-class median / weighted ensembles over the side-aware family + a
few proven divisor estimators. Goal: keep M31's wins while picking up some
of the M33/M19/M30/M16 unique recoveries.

Ensembles are deterministic, no learning.
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

from harness import NAMES, OUT_DIR, evaluate, load_trees, run
from algorithms.M01_selector_b2b3 import predict as m01
from algorithms.M19_divide_adaptive import predict as m19
from algorithms.M16_boost_b2b4 import predict as m16
from methods import (
    m30_side_aware_divide, m31_side_aware_selector, m33_refined_divide,
    n_sides_observed, naive_count, max_per_side,
)


def _floor(dets, est):
    return {c: max(est[c], max_per_side(dets, c)) for c in NAMES}


def median_three(dets, fns):
    preds = [fn(dets) for fn in fns]
    out = {c: int(sorted([p[c] for p in preds])[1]) for c in NAMES}
    return _floor(dets, out)


def median_five(dets, fns):
    preds = [fn(dets) for fn in fns]
    out = {c: int(sorted([p[c] for p in preds])[2]) for c in NAMES}
    return _floor(dets, out)


def m35_median3(dets):
    return median_three(dets, [m31_side_aware_selector, m33_refined_divide, m30_side_aware_divide])


def m36_median3_b(dets):
    return median_three(dets, [m31_side_aware_selector, m33_refined_divide, m19])


def m37_median5(dets):
    return median_five(dets, [m31_side_aware_selector, m33_refined_divide, m30_side_aware_divide, m19, m01])


def m38_mean_round(dets):
    # round((M31+M33)/2)
    a = m31_side_aware_selector(dets)
    b = m33_refined_divide(dets)
    out = {c: int(round((a[c] + b[c]) / 2.0)) for c in NAMES}
    return _floor(dets, out)


def m39_min_floor(dets):
    # min(M31, M33) with max_per_side floor — counter B3 overcount on 4-side
    a = m31_side_aware_selector(dets)
    b = m33_refined_divide(dets)
    out = {c: min(a[c], b[c]) for c in NAMES}
    return _floor(dets, out)


def m40_max_floor(dets):
    a = m31_side_aware_selector(dets)
    b = m33_refined_divide(dets)
    out = {c: max(a[c], b[c]) for c in NAMES}
    return out


METHODS = {
    "M01": m01,
    "M31_side_aware_selector": m31_side_aware_selector,
    "M33_refined_divide": m33_refined_divide,
    "M35_median3_M31_M33_M30": m35_median3,
    "M36_median3_M31_M33_M19": m36_median3_b,
    "M37_median5": m37_median5,
    "M38_mean_round": m38_mean_round,
    "M39_min_floor": m39_min_floor,
    "M40_max_floor": m40_max_floor,
}


def main() -> None:
    split_csv = OUT_DIR / "split_step12.csv"
    if split_csv.exists():
        split_csv.unlink()
    trees = load_trees()
    summary = run(METHODS, trees, tag="step12")
    cols = ["method", "acc_within1_pct", "macro_class_MAE", "n_fail",
            "exact_profile_acc_pct", "total_count_MAE", "total_count_within1_pct",
            "MAE_B1", "MAE_B2", "MAE_B3", "MAE_B4"]
    print("Full 953:")
    print(summary[cols].to_string(index=False))

    print("\nHeld-out (val+test):")
    holdout = [t for t in trees if t.split in ("val", "test")]
    rows = []
    for name, fn in METHODS.items():
        preds = {t.tree_id: fn(t.dets) for t in holdout}
        rows.append(evaluate(name, preds, holdout)["summary"])
    df_h = pd.DataFrame(rows).sort_values("acc_within1_pct", ascending=False)
    print(df_h[cols].to_string(index=False))


if __name__ == "__main__":
    main()
