"""Evaluate M51 (gmb branch → refined 3-D) on full + splits."""
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
    m31_side_aware_selector, m50_m31_with_m33_override, m51_gmb_replaced,
    m52_two_band_override, m53_three_band_override,
)


METHODS = {
    "M01": m01,
    "M31_side_aware_selector": m31_side_aware_selector,
    "M52_two_band_override": m52_two_band_override,
    "M53_three_band_override": m53_three_band_override,
}


def main() -> None:
    split_csv = OUT_DIR / "split_step18.csv"
    if split_csv.exists():
        split_csv.unlink()
    trees = load_trees()
    summary = run(METHODS, trees, tag="step18")
    cols = ["method", "acc_within1_pct", "macro_class_MAE", "n_fail",
            "exact_profile_acc_pct", "total_count_MAE", "total_count_within1_pct",
            "MAE_B1", "MAE_B2", "MAE_B3", "MAE_B4",
            "bias_B1", "bias_B2", "bias_B3", "bias_B4"]
    print("Full 953:")
    print(summary[cols].to_string(index=False))

    for split_set in [("train",), ("val", "test"), ("val",), ("test",)]:
        label = "+".join(split_set)
        sub = [t for t in trees if t.split in split_set]
        rows = []
        for name, fn in METHODS.items():
            preds = {t.tree_id: fn(t.dets) for t in sub}
            rows.append(evaluate(name, preds, sub)["summary"])
        df_h = pd.DataFrame(rows).sort_values("acc_within1_pct", ascending=False)
        print(f"\n{label} (n={len(sub)}):")
        print(df_h[cols].to_string(index=False))


if __name__ == "__main__":
    main()
