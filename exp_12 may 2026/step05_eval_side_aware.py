"""Step 5 — evaluate the new side-aware methods against M01 baseline on 953."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(EXP))

from harness import NAMES, OUT_DIR, evaluate, load_trees, run, split_breakdown

from algorithms.M01_selector_b2b3 import predict as m01
from algorithms.M03_blend_geometric import predict as m03
from algorithms.M05_blend_vis_divide import predict as m05
from algorithms.M06_weight_visibility import predict as m06
from methods import m30_side_aware_divide, m31_side_aware_selector, m32_side_aware_b2b3


METHODS = {
    "M01_selector_b2b3":   m01,
    "M03_blend_geometric": m03,
    "M05_blend_vis_divide": m05,
    "M06_weight_visibility": m06,
    "M30_side_aware_divide": m30_side_aware_divide,
    "M31_side_aware_selector": m31_side_aware_selector,
    "M32_side_aware_b2b3": m32_side_aware_b2b3,
}


def main() -> None:
    # purge prior split csv to avoid append-stacking
    split_csv = OUT_DIR / "split_step05.csv"
    if split_csv.exists():
        split_csv.unlink()

    trees = load_trees()
    summary = run(METHODS, trees, tag="step05")
    cols = ["method", "acc_within1_pct", "macro_class_MAE", "n_fail",
            "exact_profile_acc_pct", "total_count_MAE", "total_count_within1_pct",
            "MAE_B1", "MAE_B2", "MAE_B3", "MAE_B4",
            "bias_B1", "bias_B2", "bias_B3", "bias_B4"]
    print(summary[cols].to_string(index=False))

    # also evaluate ONLY on val+test (held-out from side-factor calibration)
    print("\n\nHeld-out (val+test) evaluation:")
    holdout = [t for t in trees if t.split in ("val", "test")]
    print(f"Holdout n={len(holdout)}")
    rows = []
    for name, fn in METHODS.items():
        preds = {t.tree_id: fn(t.dets) for t in holdout}
        res = evaluate(name, preds, holdout)
        rows.append(res["summary"])
    pd.DataFrame(rows).sort_values("acc_within1_pct", ascending=False)[cols].to_string()
    df_h = pd.DataFrame(rows).sort_values("acc_within1_pct", ascending=False)
    print(df_h[cols].to_string(index=False))


if __name__ == "__main__":
    main()
