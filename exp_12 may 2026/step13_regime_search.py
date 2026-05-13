"""Look for a clean regime where an alternative method beats M31 robustly."""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(EXP))

from harness import NAMES, OUT_DIR, load_trees, naive_count
from methods import n_sides_observed
from algorithms.M01_selector_b2b3 import predict as m01
from algorithms.M19_divide_adaptive import predict as m19
from algorithms.M16_boost_b2b4 import predict as m16
from methods import m30_side_aware_divide, m31_side_aware_selector, m33_refined_divide


def within1(p, g):
    return all(abs(p[c] - g[c]) <= 1 for c in NAMES)


def main() -> None:
    trees = load_trees()
    rows = []
    for t in trees:
        n_total = len(t.dets)
        ns = n_sides_observed(t.dets)
        naive = naive_count(t.dets)
        b3frac = naive["B3"] / max(n_total, 1)
        b4frac = naive["B4"] / max(n_total, 1)
        b1frac = naive["B1"] / max(n_total, 1)
        rows.append({
            "tree_id": t.tree_id,
            "split": t.split,
            "n_total": n_total,
            "n_sides": ns,
            "b1frac": b1frac, "b3frac": b3frac, "b4frac": b4frac,
            "naive_B4": naive["B4"],
            "M31": int(within1(m31_side_aware_selector(t.dets), t.gt)),
            "M33": int(within1(m33_refined_divide(t.dets), t.gt)),
            "M30": int(within1(m30_side_aware_divide(t.dets), t.gt)),
            "M19": int(within1(m19(t.dets), t.gt)),
            "M16": int(within1(m16(t.dets), t.gt)),
            "M01": int(within1(m01(t.dets), t.gt)),
        })
    df = pd.DataFrame(rows)
    print("Overall:")
    for m in ["M31", "M33", "M30", "M19", "M16", "M01"]:
        print(f"  {m}: {df[m].mean()*100:.2f}%")

    print("\n--- by n_sides ---")
    for k, sub in df.groupby("n_sides"):
        print(f"n_sides={k} (n={len(sub)}):")
        for m in ["M31", "M33", "M30", "M19", "M16", "M01"]:
            print(f"  {m}: {sub[m].mean()*100:.2f}%")

    print("\n--- by b3frac bucket (n_sides==4 only) ---")
    sub4 = df[df["n_sides"] == 4].copy()
    sub4["b3b"] = pd.cut(sub4["b3frac"], bins=[0, 0.1, 0.3, 0.45, 0.6, 0.75, 1.0])
    for k, g in sub4.groupby("b3b"):
        print(f"b3frac{k} n={len(g)}:")
        for m in ["M31", "M33", "M30", "M19", "M16", "M01"]:
            print(f"  {m}: {g[m].mean()*100:.2f}%")

    print("\n--- by n_total bucket (n_sides==4) ---")
    sub4["nb"] = pd.cut(sub4["n_total"], bins=[0, 8, 16, 25, 40, 1000])
    for k, g in sub4.groupby("nb"):
        print(f"n_total{k} n={len(g)}:")
        for m in ["M31", "M33", "M30", "M19", "M16", "M01"]:
            print(f"  {m}: {g[m].mean()*100:.2f}%")


if __name__ == "__main__":
    main()
