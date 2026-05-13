"""
Verify regime overrides on held-out split before deciding.

Hypothesis from step13 (on full 953, BIASED — must verify on holdout):
  (n_sides=4, b3frac in (0.3, 0.45]): M16 91.09 vs M31 89.60
  (n_sides=4, b3frac in (0.45, 0.6]): M33 92.83 vs M31 91.63

Now split by train vs val+test. If the relative ordering holds on val+test
with no parameter tuning, the regime cut is real, not data artifact.
"""

from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(EXP))

from harness import NAMES, load_trees, naive_count
from methods import n_sides_observed, m31_side_aware_selector, m33_refined_divide
from algorithms.M16_boost_b2b4 import predict as m16


def within1(p, g):
    return all(abs(p[c] - g[c]) <= 1 for c in NAMES)


def main():
    trees = load_trees()
    rows = []
    for t in trees:
        n_total = len(t.dets)
        ns = n_sides_observed(t.dets)
        naive = naive_count(t.dets)
        b3frac = naive["B3"] / max(n_total, 1)
        rows.append({
            "tree_id": t.tree_id,
            "split": t.split,
            "ns": ns,
            "b3frac": b3frac,
            "M31_ok": int(within1(m31_side_aware_selector(t.dets), t.gt)),
            "M33_ok": int(within1(m33_refined_divide(t.dets), t.gt)),
            "M16_ok": int(within1(m16(t.dets), t.gt)),
        })
    df = pd.DataFrame(rows)
    for split_set in [("train",), ("val", "test")]:
        sub_all = df[df["split"].isin(split_set)]
        label = "TRAIN" if split_set == ("train",) else "VAL+TEST"
        print(f"\n=== {label} (n={len(sub_all)}) ===")
        for low, high in [(0.30, 0.45), (0.45, 0.60), (0.60, 0.75)]:
            sub = sub_all[(sub_all["ns"] == 4) & (sub_all["b3frac"] > low) & (sub_all["b3frac"] <= high)]
            n = len(sub)
            if n == 0:
                continue
            print(f"  b3frac ({low}, {high}], n={n}: "
                  f"M31={sub['M31_ok'].mean()*100:.2f}%  M33={sub['M33_ok'].mean()*100:.2f}%  M16={sub['M16_ok'].mean()*100:.2f}%")


if __name__ == "__main__":
    main()
