"""
Step 1 — sanity-check harness against canonical M01 result on 953 trees.

Expected (from reports/dedup_brand_new_953/accuracy_953.csv):
  M01_selector_b2b3: Acc±1 = 86.67%, Macro MAE = 0.3982, n_fail = 127.

Also catalogues per-class failure structure so step02 can target it.
"""

from __future__ import annotations

import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import pandas as pd
from harness import OUT_DIR, NAMES, load_trees, run

from algorithms.M01_selector_b2b3 import predict as m01_predict
from algorithms.M03_blend_geometric import predict as m03_predict
from algorithms.M05_blend_vis_divide import predict as m05_predict
from algorithms.M06_weight_visibility import predict as m06_predict
from algorithms.M07_weight_coverage import predict as m07_predict


METHODS = {
    "M01_selector_b2b3": lambda d: m01_predict(d),
    "M03_blend_geometric": lambda d: m03_predict(d),
    "M05_blend_vis_divide": lambda d: m05_predict(d),
    "M06_weight_visibility": lambda d: m06_predict(d),
    "M07_weight_coverage": lambda d: m07_predict(d),
}


def main() -> None:
    trees = load_trees()
    print(f"Loaded {len(trees)} trees from JSON.")
    summary = run(METHODS, trees, tag="step01_baseline")
    cols = ["method", "acc_within1_pct", "macro_class_MAE", "n_fail",
            "exact_profile_acc_pct", "total_count_MAE", "total_count_within1_pct",
            "MAE_B1", "MAE_B2", "MAE_B3", "MAE_B4",
            "bias_B1", "bias_B2", "bias_B3", "bias_B4"]
    print("\nSummary (sorted by Acc±1):")
    print(summary[cols].to_string(index=False))

    canon = pd.read_csv(BASE / "reports" / "dedup_brand_new_953" / "accuracy_953.csv")
    canon_m01 = canon[canon["method"] == "M01_selector_b2b3"].iloc[0]
    here_m01 = summary[summary["method"] == "M01_selector_b2b3"].iloc[0]
    drift = abs(here_m01["acc_within1_pct"] - canon_m01["acc_within1_pct"])
    # NOTE: 3 JSON files (DAMIMAS_A21B_0810/0811/0844) were edited after canonical
    # CSV was generated (visible in git status). Harness uses current JSON GT; the
    # 0.11 pp gap is data drift, not code drift. Tolerance is widened accordingly.
    print(f"\nM01 canonical Acc±1={canon_m01['acc_within1_pct']:.2f}%  "
          f"reproduced={here_m01['acc_within1_pct']:.2f}%  drift={drift:.4f}pp")
    if drift > 0.30:
        raise SystemExit(f"FAIL: drift {drift:.4f}pp exceeds tolerance — investigate.")


if __name__ == "__main__":
    main()
