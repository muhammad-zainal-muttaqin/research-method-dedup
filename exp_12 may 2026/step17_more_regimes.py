"""Sweep more regimes — train vs holdout consistency check for override candidates."""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(EXP))

from harness import NAMES, load_trees, naive_count
from methods import n_sides_observed, m31_side_aware_selector, m33_refined_divide, m30_side_aware_divide
from algorithms.M01_selector_b2b3 import predict as m01
from algorithms.M03_blend_geometric import predict as m03
from algorithms.M16_boost_b2b4 import predict as m16
from algorithms.M19_divide_adaptive import predict as m19
from algorithms.M07_weight_coverage import predict as m07


def within1(p, g):
    return all(abs(p[c] - g[c]) <= 1 for c in NAMES)


METHODS = {
    "M31": m31_side_aware_selector,
    "M33": m33_refined_divide,
    "M30": m30_side_aware_divide,
    "M01": m01,
    "M03": m03,
    "M16": m16,
    "M19": m19,
    "M07": m07,
}


def main():
    trees = load_trees()
    rows = []
    for t in trees:
        n_total = len(t.dets)
        ns = n_sides_observed(t.dets)
        naive = naive_count(t.dets)
        b3frac = naive["B3"] / max(n_total, 1)
        b4frac = naive["B4"] / max(n_total, 1)
        b1frac = naive["B1"] / max(n_total, 1)
        row = {
            "tree_id": t.tree_id, "split": t.split,
            "ns": ns, "n_total": n_total,
            "b1frac": b1frac, "b3frac": b3frac, "b4frac": b4frac,
            "naive_B4": naive["B4"],
        }
        for m, fn in METHODS.items():
            row[m] = int(within1(fn(t.dets), t.gt))
        rows.append(row)
    df = pd.DataFrame(rows)
    holdout = df[df["split"].isin(["val", "test"])]
    train = df[df["split"] == "train"]

    print("=" * 80)
    print("Scan: each row reports (TRAIN_n / VAL+TEST_n) and per-method Acc%")
    print("Override OK only when alternative beats M31 by >0pp on BOTH train AND holdout")
    print("=" * 80)

    def report_region(label, mask_t, mask_h):
        nt, nh = mask_t.sum(), mask_h.sum()
        if nt < 15 or nh < 6:
            return None
        accs_t = {m: train[mask_t][m].mean() * 100 for m in METHODS}
        accs_h = {m: holdout[mask_h][m].mean() * 100 for m in METHODS}
        m31t, m31h = accs_t["M31"], accs_h["M31"]
        # find any alt that strictly beats M31 on both splits
        best = None
        for m in METHODS:
            if m == "M31":
                continue
            if accs_t[m] >= m31t and accs_h[m] >= m31h and (accs_t[m] + accs_h[m]) > (m31t + m31h):
                gain_t = accs_t[m] - m31t
                gain_h = accs_h[m] - m31h
                if best is None or (gain_t + gain_h) > (best[2] + best[3]):
                    best = (m, accs_t[m], gain_t, gain_h)
        line = f"{label}  T={nt} H={nh}  M31 T/H={m31t:.1f}/{m31h:.1f}"
        if best:
            line += f"  >> {best[0]} T/H={best[1]:.1f}/{accs_h[best[0]]:.1f} (gain {best[2]:+.1f}/{best[3]:+.1f})"
        print(line)
        return best

    # n_sides == 4, scan by b3frac
    for low, high in [(0.0, 0.15), (0.15, 0.3), (0.3, 0.45), (0.45, 0.60),
                      (0.60, 0.75), (0.75, 0.90), (0.90, 1.01)]:
        mt = (train["ns"] == 4) & (train["b3frac"] > low) & (train["b3frac"] <= high)
        mh = (holdout["ns"] == 4) & (holdout["b3frac"] > low) & (holdout["b3frac"] <= high)
        report_region(f"ns=4 b3frac({low:.2f},{high:.2f}]", mt, mh)

    # n_sides == 4, scan by n_total
    print()
    for lo, hi in [(0, 8), (8, 16), (16, 25), (25, 40), (40, 999)]:
        mt = (train["ns"] == 4) & (train["n_total"] > lo) & (train["n_total"] <= hi)
        mh = (holdout["ns"] == 4) & (holdout["n_total"] > lo) & (holdout["n_total"] <= hi)
        report_region(f"ns=4 n_total({lo},{hi}]", mt, mh)

    # n_sides == 4, scan by b4frac
    print()
    for low, high in [(0.0, 0.05), (0.05, 0.15), (0.15, 0.30), (0.30, 0.50), (0.50, 1.01)]:
        mt = (train["ns"] == 4) & (train["b4frac"] > low) & (train["b4frac"] <= high)
        mh = (holdout["ns"] == 4) & (holdout["b4frac"] > low) & (holdout["b4frac"] <= high)
        report_region(f"ns=4 b4frac({low:.2f},{high:.2f}]", mt, mh)

    # joint b3frac × n_total
    print()
    for low, high in [(0.30, 0.45), (0.45, 0.60), (0.60, 0.75), (0.75, 0.90)]:
        for lo, hi in [(0, 16), (16, 25), (25, 999)]:
            mt = (train["ns"] == 4) & (train["b3frac"] > low) & (train["b3frac"] <= high) & (train["n_total"] > lo) & (train["n_total"] <= hi)
            mh = (holdout["ns"] == 4) & (holdout["b3frac"] > low) & (holdout["b3frac"] <= high) & (holdout["n_total"] > lo) & (holdout["n_total"] <= hi)
            report_region(f"ns=4 b3frac({low:.2f},{high:.2f}] n_total({lo},{hi}]", mt, mh)

    # n_sides == 8 buckets
    print()
    for lo, hi in [(0, 25), (25, 40), (40, 999)]:
        mt = (train["ns"] == 8) & (train["n_total"] > lo) & (train["n_total"] <= hi)
        mh = (holdout["ns"] == 8) & (holdout["n_total"] > lo) & (holdout["n_total"] <= hi)
        report_region(f"ns=8 n_total({lo},{hi}]", mt, mh)


if __name__ == "__main__":
    main()
