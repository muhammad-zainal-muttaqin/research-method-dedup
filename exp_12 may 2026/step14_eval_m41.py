"""Evaluate M41 (B3 saturation-aware) vs M31."""
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
from methods import m31_side_aware_selector, m41_b3frac_divisor


METHODS = {
    "M01": m01,
    "M31_side_aware_selector": m31_side_aware_selector,
    "M41_b3frac_divisor": m41_b3frac_divisor,
}


def main() -> None:
    split_csv = OUT_DIR / "split_step14.csv"
    if split_csv.exists():
        split_csv.unlink()
    trees = load_trees()
    summary = run(METHODS, trees, tag="step14")
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

    print("\nTRAIN-only:")
    train = [t for t in trees if t.split == "train"]
    rows = []
    for name, fn in METHODS.items():
        preds = {t.tree_id: fn(t.dets) for t in train}
        rows.append(evaluate(name, preds, train)["summary"])
    df_t = pd.DataFrame(rows).sort_values("acc_within1_pct", ascending=False)
    print(df_t[cols].to_string(index=False))


if __name__ == "__main__":
    main()
