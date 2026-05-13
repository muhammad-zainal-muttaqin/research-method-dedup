"""Profile remaining M31 failures and look for recoverable subsets."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(EXP))

from harness import NAMES, OUT_DIR, load_trees, naive_count, max_per_side
from methods import (
    m31_side_aware_selector, m30_side_aware_divide,
    visibility_count, side_coverage, geometric_mean_blend, median3_floor,
    adaptive_corrected_M01, n_sides_observed,
)


def main() -> None:
    trees = load_trees()
    rows = []
    for t in trees:
        dets = t.dets
        ns = n_sides_observed(dets)
        n_total = len(dets)
        naive = naive_count(dets)
        pred = m31_side_aware_selector(dets)
        within = all(abs(pred[c] - t.gt[c]) <= 1 for c in NAMES)
        # alternates
        alts = {
            "M30_sa": m30_side_aware_divide(dets),
            "vis": visibility_count(dets),
            "sid": side_coverage(dets),
            "gmb": geometric_mean_blend(dets),
            "med": median3_floor(dets),
            "adp": adaptive_corrected_M01(dets),
        }
        row = {
            "tree_id": t.tree_id,
            "split": t.split,
            "n_dets": n_total,
            "n_sides": ns,
            "b3frac": naive["B3"] / max(n_total, 1),
            **{f"naive_{c}": naive[c] for c in NAMES},
            **{f"gt_{c}": t.gt[c] for c in NAMES},
            **{f"M31_{c}": pred[c] for c in NAMES},
            **{f"M31_err_{c}": pred[c] - t.gt[c] for c in NAMES},
            "M31_within1": within,
        }
        for name, p in alts.items():
            row[f"{name}_within1"] = all(abs(p[c] - t.gt[c]) <= 1 for c in NAMES)
            for c in NAMES:
                row[f"{name}_{c}"] = p[c]
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "step06_all.csv", index=False)
    fail = df[~df["M31_within1"]].copy()
    fail.to_csv(OUT_DIR / "step06_failures.csv", index=False)
    print(f"M31 fail: {len(fail)} (Acc±1 {df['M31_within1'].mean()*100:.2f}%)")

    print("\nSole-class failures (only this class breaks ±1):")
    def sole_class(r):
        bad = [c for c in NAMES if abs(r[f"M31_err_{c}"]) > 1]
        return bad[0] if len(bad) == 1 else "multi" if len(bad) > 1 else "none"
    fail["sole_class"] = fail.apply(sole_class, axis=1)
    print(fail["sole_class"].value_counts().to_string())

    print("\nPer-class |err|>=2 counts in failures, with bias direction:")
    for c in NAMES:
        big = fail[fail[f"M31_err_{c}"].abs() >= 2]
        pos = (big[f"M31_err_{c}"] > 0).sum()
        neg = (big[f"M31_err_{c}"] < 0).sum()
        print(f"  {c}: {len(big)} trees, overcount {pos} undercount {neg} (bias_sum={big[f'M31_err_{c}'].sum():+d})")

    print("\nRecovery rate of alternates on M31 failures:")
    alts = ["M30_sa", "vis", "sid", "gmb", "med", "adp"]
    for a in alts:
        n = fail[f"{a}_within1"].sum()
        print(f"  {a}: {n}")
    fail["any_alt"] = fail[[f"{a}_within1" for a in alts]].any(axis=1)
    print(f"At-least-one alternate fixes: {fail['any_alt'].sum()}")

    print("\nM31 Acc±1 by n_sides:")
    for k, sub in df.groupby("n_sides"):
        print(f"  {k}: {sub['M31_within1'].mean()*100:.2f}%  (n={len(sub)}, fail={(~sub['M31_within1']).sum()})")

    print("\nM31 Acc±1 by n_dets bucket:")
    df["nbucket"] = pd.cut(df["n_dets"], bins=[0, 8, 16, 25, 40, 80, 1000])
    for k, sub in df.groupby("nbucket"):
        print(f"  {k}: {sub['M31_within1'].mean()*100:.2f}%  (n={len(sub)}, fail={(~sub['M31_within1']).sum()})")

    print("\nM31 Acc±1 by b3frac bucket:")
    df["b3b"] = pd.cut(df["b3frac"], bins=[0, 0.1, 0.3, 0.45, 0.6, 0.75, 1.0])
    for k, sub in df.groupby("b3b"):
        print(f"  {k}: {sub['M31_within1'].mean()*100:.2f}%  (n={len(sub)}, fail={(~sub['M31_within1']).sum()})")


if __name__ == "__main__":
    main()
