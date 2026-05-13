"""
Step 2 — profile the 126 M01 failures to find exploitable structure.

Outputs (in exp_12 may 2026/out/):
  - failures_M01.csv    one row per failed tree, with error signature + features
  - failure_summary.txt  cluster counts, class-error histogram, conditional acc
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(EXP))

from harness import NAMES, OUT_DIR, load_trees, max_per_side, naive_count
from algorithms.M01_selector_b2b3 import predict as m01_predict
from algorithms.M06_weight_visibility import predict as m06_predict
from algorithms.M07_weight_coverage import predict as m07_predict
from algorithms.M03_blend_geometric import predict as m03_predict
from algorithms.M05_blend_vis_divide import predict as m05_predict


CANDIDATES = {
    "M01": m01_predict,
    "M03": m03_predict,
    "M05": m05_predict,
    "M06": m06_predict,
    "M07": m07_predict,
}


def main() -> None:
    trees = load_trees()
    rows = []
    for t in trees:
        dets = t.dets
        n_total = len(dets)
        naive = naive_count(dets)
        row = {
            "tree_id": t.tree_id,
            "split": t.split,
            "n_dets": n_total,
            "n_sides_observed": len({d["side_index"] for d in dets}) if dets else 0,
        }
        # baseline features
        for c in NAMES:
            row[f"naive_{c}"] = naive[c]
            row[f"gt_{c}"] = t.gt[c]
            row[f"maxside_{c}"] = max_per_side(dets, c)
        row["b3frac_naive"] = naive["B3"] / max(n_total, 1)
        row["b4frac_naive"] = naive["B4"] / max(n_total, 1)
        # bbox stats
        if dets:
            areas = np.array([d["area_norm"] for d in dets])
            row["area_mean"] = float(areas.mean())
            row["area_p90"] = float(np.percentile(areas, 90))
            row["area_p10"] = float(np.percentile(areas, 10))
        else:
            row["area_mean"] = row["area_p90"] = row["area_p10"] = 0.0
        # per-method predictions
        for mname, mfunc in CANDIDATES.items():
            pred = mfunc(dets)
            for c in NAMES:
                row[f"{mname}_{c}"] = pred[c]
                row[f"{mname}_err_{c}"] = pred[c] - t.gt[c]
            row[f"{mname}_within1"] = all(abs(pred[c] - t.gt[c]) <= 1 for c in NAMES)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "step02_all_trees.csv", index=False)

    fail = df[~df["M01_within1"]].copy()
    fail.to_csv(OUT_DIR / "failures_M01.csv", index=False)

    # ─── reports ───
    lines = []
    lines.append(f"Trees: {len(df)}  M01 within±1: {df['M01_within1'].sum()}  fail: {len(fail)}")
    lines.append(f"M01 Acc±1: {df['M01_within1'].mean()*100:.2f}%")
    lines.append("")

    # which class contributes the most failures (abs err >= 2)
    lines.append("Per-class |err|>=2 share among failures:")
    for c in NAMES:
        big_err = (fail[f"M01_err_{c}"].abs() >= 2).sum()
        lines.append(f"  {c}: {big_err} trees (bias_sum={fail[f'M01_err_{c}'].sum():+d})")
    lines.append("")

    # class-only-fail counts: which single class makes a tree fail
    def sole_class(r):
        bad = [c for c in NAMES if abs(r[f"M01_err_{c}"]) > 1]
        return bad[0] if len(bad) == 1 else "multi" if len(bad) > 1 else "none"
    fail["sole_class"] = fail.apply(sole_class, axis=1)
    lines.append("Sole-class failure breakdown:")
    for k, v in fail["sole_class"].value_counts().items():
        lines.append(f"  {k}: {v}")
    lines.append("")

    # recoverability: per-tree best-of-N among candidates
    fail["any_candidate_within1"] = fail[[f"{m}_within1" for m in CANDIDATES]].any(axis=1)
    lines.append(f"M01 failures recoverable by SOME existing candidate: {fail['any_candidate_within1'].sum()}/{len(fail)}")

    # which candidate recovers M01-fails
    lines.append("Per-candidate recovery rate on M01 failures:")
    for m in CANDIDATES:
        n = fail[f"{m}_within1"].sum()
        lines.append(f"  {m}: {n}")
    lines.append("")

    # detection-density buckets
    lines.append("M01 Acc±1 by detection-density bucket:")
    df["nbucket"] = pd.cut(df["n_dets"], bins=[0, 8, 16, 25, 40, 80, 1000])
    for k, sub in df.groupby("nbucket"):
        lines.append(f"  {k}: {sub['M01_within1'].mean()*100:.2f}%  (n={len(sub)})")
    lines.append("")

    # B3-fraction buckets
    lines.append("M01 Acc±1 by b3frac_naive bucket:")
    df["b3bucket"] = pd.cut(df["b3frac_naive"], bins=[0, 0.1, 0.3, 0.45, 0.6, 0.75, 1.0])
    for k, sub in df.groupby("b3bucket"):
        lines.append(f"  {k}: {sub['M01_within1'].mean()*100:.2f}%  (n={len(sub)})")

    print("\n".join(lines))
    (OUT_DIR / "failure_summary.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
