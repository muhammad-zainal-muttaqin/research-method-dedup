"""Evaluate M33/M34 alongside prior champions."""

from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(EXP))

from harness import NAMES, OUT_DIR, evaluate, load_trees, run
from algorithms.M01_selector_b2b3 import predict as m01
from methods import (
    m30_side_aware_divide,
    m31_side_aware_selector,
    m32_side_aware_b2b3,
    m33_refined_divide,
    m34_refined_selector,
)


METHODS = {
    "M01_selector_b2b3":        m01,
    "M30_side_aware_divide":    m30_side_aware_divide,
    "M31_side_aware_selector":  m31_side_aware_selector,
    "M32_side_aware_b2b3":      m32_side_aware_b2b3,
    "M33_refined_divide":       m33_refined_divide,
    "M34_refined_selector":     m34_refined_selector,
}


def main() -> None:
    split_csv = OUT_DIR / "split_step10.csv"
    if split_csv.exists():
        split_csv.unlink()
    trees = load_trees()
    summary = run(METHODS, trees, tag="step10")
    cols = ["method", "acc_within1_pct", "macro_class_MAE", "n_fail",
            "exact_profile_acc_pct", "total_count_MAE", "total_count_within1_pct",
            "MAE_B1", "MAE_B2", "MAE_B3", "MAE_B4"]
    print("Full 953:")
    print(summary[cols].to_string(index=False))

    print("\nHeld-out (val+test only):")
    holdout = [t for t in trees if t.split in ("val", "test")]
    rows = []
    for name, fn in METHODS.items():
        preds = {t.tree_id: fn(t.dets) for t in holdout}
        rows.append(evaluate(name, preds, holdout)["summary"])
    df_h = pd.DataFrame(rows).sort_values("acc_within1_pct", ascending=False)
    print(df_h[cols].to_string(index=False))


if __name__ == "__main__":
    main()
