"""
Iteration 2 — characterise the 133 trees where floor_clamped_hybrid fails.

Goal: identify the dominant failure mode so iter3 can design targeted fixes.

For each failing tree, log:
- per-class error (pred - gt), so over/under direction is preserved
- naive total, n_sides, total_dets
- which class is the breaking class (|err| >= 2)

Output: exp_10 May 2026/iter2_failures.csv + iter2_summary.md
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
JSON_DIR = BASE / "Brand-New-Dataset-YOLO" / "json"
OUT_DIR = BASE / "exp_10 May 2026"

sys.path.insert(0, str(BASE / "scripts"))
import dedup_all_953 as base  # noqa: E402

NAMES = ["B1", "B2", "B3", "B4"]


def _max_per_side_count(dets, c):
    cd = [d for d in dets if d["class"] == c]
    return max(Counter(d["side_index"] for d in cd).values()) if cd else 0


def floor_clamped_hybrid(dets):
    base_pred = base.hybrid_vis_corr(dets)
    return {c: max(base_pred[c], _max_per_side_count(dets, c)) for c in NAMES}


def load_953_trees():
    trees = {}
    for jp in sorted(JSON_DIR.glob("*.json")):
        data = json.loads(jp.read_text(encoding="utf-8"))
        tree_id = data.get("tree_name", data.get("tree_id", jp.stem))
        gt = {c: data["summary"]["by_class"].get(c, 0) for c in NAMES}
        dets = []
        for side, sd in data["images"].items():
            si = sd.get("side_index", int(side.replace("sisi_", "")) - 1)
            for ann in sd.get("annotations", []):
                if "bbox_yolo" in ann:
                    dets.append(base._parse_det(ann, side, si))
        trees[tree_id] = {"dets": dets, "gt": gt}
    return trees


def main():
    base._load_v6_params()
    trees = load_953_trees()

    failures = []
    for tid, info in trees.items():
        dets = info["dets"]
        gt = info["gt"]
        pred = floor_clamped_hybrid(dets)
        err = {c: pred[c] - gt[c] for c in NAMES}
        if any(abs(err[c]) > 1 for c in NAMES):
            naive = base.naive_count(dets)
            failures.append({
                "tree_id": tid,
                "n_dets": len(dets),
                "n_sides": len(set(d["side_index"] for d in dets)),
                **{f"gt_{c}": gt[c] for c in NAMES},
                **{f"pred_{c}": pred[c] for c in NAMES},
                **{f"err_{c}": err[c] for c in NAMES},
                **{f"naive_{c}": naive[c] for c in NAMES},
                "abs_total_err": sum(abs(err[c]) for c in NAMES),
                "breaking_classes": ",".join(c for c in NAMES if abs(err[c]) > 1),
            })

    df = pd.DataFrame(failures).sort_values("abs_total_err", ascending=False)
    out_csv = OUT_DIR / "iter2_failures.csv"
    df.to_csv(out_csv, index=False)
    print(f"Failures: {len(df)} trees")
    print(f"CSV: {out_csv}")

    # Aggregate failure pattern
    print("\nBreaking-class frequency (|err|>1):")
    for c in NAMES:
        n_break = int((df[f"err_{c}"].abs() > 1).sum())
        n_over = int((df[f"err_{c}"] > 1).sum())
        n_under = int((df[f"err_{c}"] < -1).sum())
        print(f"  {c}: total={n_break}  over_pred={n_over}  under_pred={n_under}")

    print("\nMean signed error per class (failures only):")
    for c in NAMES:
        m = df[f"err_{c}"].mean()
        print(f"  {c}: {m:+.3f}")

    print("\nFailures by abs_total_err:")
    print(df["abs_total_err"].value_counts().sort_index().to_string())

    print("\nMost common breaking-class combos:")
    print(df["breaking_classes"].value_counts().head(10).to_string())

    print("\nDistribution of n_dets in failures:")
    print(df["n_dets"].describe().to_string())

    # Failure classification by direction
    print("\nDirection split:")
    over_only = df[(df[[f"err_{c}" for c in NAMES]] > 1).any(axis=1) & ~(df[[f"err_{c}" for c in NAMES]] < -1).any(axis=1)]
    under_only = df[(df[[f"err_{c}" for c in NAMES]] < -1).any(axis=1) & ~(df[[f"err_{c}" for c in NAMES]] > 1).any(axis=1)]
    mixed = df[(df[[f"err_{c}" for c in NAMES]] > 1).any(axis=1) & (df[[f"err_{c}" for c in NAMES]] < -1).any(axis=1)]
    print(f"  over_pred only:  {len(over_only)}")
    print(f"  under_pred only: {len(under_only)}")
    print(f"  mixed:           {len(mixed)}")


if __name__ == "__main__":
    main()
