"""
Run ALL dedup methods on Brand-New-Dataset-YOLO (953 trees, full JSON GT).

This is the canonical 953-tree benchmark — supersedes 228/478/727/882-tree runs.
Source: Brand-New-Dataset-YOLO/json/ (created 2026-05-09).
All trees have JSON GT, so accuracy computed for ALL 953 trees.

Output: reports/dedup_brand_new_953/
- per_tree.csv          all method predictions per tree
- accuracy_953.csv      Acc±1, MAE, mean_total_err, n_fail per method
- totals.csv            grand totals + dedup_ratio_vs_naive
- mean_per_tree.csv     mean/median bunch counts per method
"""

from __future__ import annotations

import json
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
JSON_DIR = BASE / "Brand-New-Dataset-YOLO" / "json"
OUT_DIR = BASE / "reports" / "dedup_brand_new_953"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NAMES = ["B1", "B2", "B3", "B4"]

# Make scripts/ importable so we can reuse method registry
sys.path.insert(0, str(BASE / "scripts"))

# Reuse method definitions from dedup_all_953
import dedup_all_953 as base


def load_brand_new_trees() -> Dict[str, dict]:
    """Load all 953 trees from Brand-New-Dataset-YOLO/json/."""
    trees = {}
    for jp in sorted(JSON_DIR.glob("*.json")):
        data = json.loads(jp.read_text(encoding="utf-8-sig"))
        tree_id = data.get("tree_name", data.get("tree_id", jp.stem))
        gt = {c: data["summary"]["by_class"].get(c, 0) for c in NAMES}
        dets = []
        for side, sd in data["images"].items():
            si = sd.get("side_index", int(side.replace("sisi_", "")) - 1)
            for ann in sd.get("annotations", []):
                if "bbox_yolo" in ann:
                    dets.append(base._parse_det(ann, side, si))
        trees[tree_id] = {
            "dets": dets,
            "gt": gt,
            "split": data.get("split", "unknown"),
            "source": "json",
        }
    return trees


def _within1(pred, gt):
    return all(abs(pred.get(c, 0) - gt.get(c, 0)) <= 1 for c in NAMES)


def _mae(pred, gt):
    return float(np.mean([abs(pred.get(c, 0) - gt.get(c, 0)) for c in NAMES]))


def main():
    print("Loading v6 params...")
    base._load_v6_params()

    print(f"Loading 953 trees from {JSON_DIR}...")
    trees = load_brand_new_trees()
    print(f"Total trees loaded: {len(trees)}")

    method_names = list(base.METHOD_GROUPS.keys())

    # ── per-tree predictions ──────────────────────────────────
    rows = []
    for tree_id, info in sorted(trees.items()):
        dets = info["dets"]
        row = {
            "tree_id": tree_id,
            "split": info["split"],
            "n_dets": len(dets),
            "n_sides": len(set(d["side_index"] for d in dets)),
        }
        for mname, mfunc in base.METHOD_GROUPS.items():
            try:
                pred = mfunc(dets)
            except Exception as e:
                pred = {c: -1 for c in NAMES}
                row[f"_err_{mname}"] = str(e)
            for c in NAMES:
                row[f"{mname}_{c}"] = pred.get(c, 0)
            row[f"{mname}_total"] = sum(pred.get(c, 0) for c in NAMES)
        rows.append(row)

    per_tree_df = pd.DataFrame(rows)
    per_tree_df.to_csv(OUT_DIR / "per_tree.csv", index=False)
    print(f"Per-tree CSV saved: {len(per_tree_df)} rows.")

    # ── accuracy on all 953 trees ─────────────────────────────
    acc_rows = []
    for mname in method_names:
        within1_list, mae_list, errsum_list = [], [], []
        per_class_mae = {c: [] for c in NAMES}
        per_class_err = {c: [] for c in NAMES}
        exact_profile_list = []
        total_count_err_list = []
        total_count_within1_list = []
        for tree_id in per_tree_df["tree_id"]:
            gt = trees[tree_id]["gt"]
            pred = {c: int(per_tree_df.loc[per_tree_df["tree_id"] == tree_id, f"{mname}_{c}"].iloc[0]) for c in NAMES}
            within1_list.append(_within1(pred, gt))
            mae_list.append(_mae(pred, gt))
            errsum_list.append(sum(abs(pred.get(c, 0) - gt.get(c, 0)) for c in NAMES))

            # Per-class metrics
            for c in NAMES:
                per_class_mae[c].append(abs(pred.get(c, 0) - gt.get(c, 0)))
                per_class_err[c].append(pred.get(c, 0) - gt.get(c, 0))

            # Exact-profile accuracy
            exact_profile_list.append(all(pred.get(c, 0) == gt.get(c, 0) for c in NAMES))

            # Total count metrics
            total_pred = sum(pred.get(c, 0) for c in NAMES)
            total_gt = sum(gt.get(c, 0) for c in NAMES)
            total_count_err_list.append(abs(total_pred - total_gt))
            total_count_within1_list.append(abs(total_pred - total_gt) <= 1)

        acc_rows.append({
            "method": mname,
            "acc_within1_pct": round(np.mean(within1_list) * 100, 2),
            "MAE": round(np.mean(mae_list), 4),
            "mean_total_err": round(np.mean(errsum_list), 4),
            "n_fail": int(sum(1 for x in within1_list if not x)),
            # Mandatory metrics
            "MAE_B1": round(np.mean(per_class_mae["B1"]), 4),
            "MAE_B2": round(np.mean(per_class_mae["B2"]), 4),
            "MAE_B3": round(np.mean(per_class_mae["B3"]), 4),
            "MAE_B4": round(np.mean(per_class_mae["B4"]), 4),
            "macro_class_MAE": round(np.mean([np.mean(per_class_mae[c]) for c in NAMES]), 4),
            "exact_profile_acc": round(np.mean(exact_profile_list) * 100, 2),
            "total_count_MAE": round(np.mean(total_count_err_list), 4),
            "total_count_within1_pct": round(np.mean(total_count_within1_list) * 100, 2),
            "mean_error_B1": round(np.mean(per_class_err["B1"]), 4),
            "mean_error_B2": round(np.mean(per_class_err["B2"]), 4),
            "mean_error_B3": round(np.mean(per_class_err["B3"]), 4),
            "mean_error_B4": round(np.mean(per_class_err["B4"]), 4),
        })

    acc_df = pd.DataFrame(acc_rows).sort_values("acc_within1_pct", ascending=False)
    acc_df.to_csv(OUT_DIR / "accuracy_953.csv", index=False)

    # ── aggregate counts ──────────────────────────────────────
    agg_rows = []
    for mname in method_names:
        totals = {c: int(per_tree_df[f"{mname}_{c}"].sum()) for c in NAMES}
        grand = sum(totals.values())
        naive_grand = int(per_tree_df[[f"M29_baseline_naive_sum_{c}" for c in NAMES]].sum().sum())
        agg_rows.append({
            "method": mname,
            **{c: totals[c] for c in NAMES},
            "total": grand,
            "dedup_ratio_vs_naive": round(grand / naive_grand, 4) if naive_grand else None,
        })
    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv(OUT_DIR / "totals.csv", index=False)

    # ── per-method mean total per tree ────────────────────────
    mean_rows = []
    for mname in method_names:
        mean_rows.append({
            "method": mname,
            "mean_total_per_tree": round(per_tree_df[f"{mname}_total"].mean(), 3),
            "median_total_per_tree": round(per_tree_df[f"{mname}_total"].median(), 3),
            **{f"mean_{c}": round(per_tree_df[f"{mname}_{c}"].mean(), 3) for c in NAMES},
        })
    mean_df = pd.DataFrame(mean_rows)
    mean_df.to_csv(OUT_DIR / "mean_per_tree.csv", index=False)

    # ── print summary ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"ACCURACY on {len(trees)} Brand-New-Dataset-YOLO trees (Acc ±1)")
    print("=" * 70)
    print(acc_df.to_string(index=False))

    print("\n" + "=" * 70)
    print(f"AGGREGATE COUNTS — all {len(trees)} trees")
    print("=" * 70)
    print(agg_df.to_string(index=False))

    print(f"\nAll outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
