"""
Oracle analysis — for each tree, ask: is any method in our candidate pool
within ±1? This is an UPPER BOUND on routing-based gains, not a method we
would ship.

Pool: M01 + side-aware-family + existing algorithms/.
"""

from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(EXP))

from harness import NAMES, OUT_DIR, load_trees
from algorithms.M01_selector_b2b3 import predict as m01
from algorithms.M03_blend_geometric import predict as m03
from algorithms.M05_blend_vis_divide import predict as m05
from algorithms.M06_weight_visibility import predict as m06
from algorithms.M07_weight_coverage import predict as m07
from algorithms.M08_divide_density_vis import predict as m08
from algorithms.M09_median_strong5 import predict as m09_raw
from algorithms.M19_divide_adaptive import predict as m19
from algorithms.M16_boost_b2b4 import predict as m16

def m09(dets):
    return m09_raw(dets, {})
from methods import (
    m30_side_aware_divide, m31_side_aware_selector, m33_refined_divide,
    visibility_count, side_coverage, geometric_mean_blend,
    median3_floor, adaptive_corrected_M01,
)

POOL = {
    "M01": m01, "M03": m03, "M05": m05, "M06": m06, "M07": m07, "M08": m08,
    "M16": m16, "M19": m19,
    "M30": m30_side_aware_divide, "M31": m31_side_aware_selector, "M33": m33_refined_divide,
    "vis": visibility_count, "sid": side_coverage, "gmb": geometric_mean_blend,
    "med": median3_floor, "adp": adaptive_corrected_M01,
}


def within1(p, g):
    return all(abs(p[c] - g[c]) <= 1 for c in NAMES)


def main() -> None:
    trees = load_trees()
    rows = []
    for t in trees:
        row = {"tree_id": t.tree_id, "split": t.split}
        wins = []
        for name, fn in POOL.items():
            p = fn(t.dets)
            ok = within1(p, t.gt)
            row[name] = int(ok)
            if ok:
                wins.append(name)
        row["any_pass"] = int(bool(wins))
        row["winners"] = ",".join(wins)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "step11_oracle.csv", index=False)
    print(f"Trees: {len(df)}")
    print(f"Oracle (any method ±1): {df['any_pass'].mean()*100:.2f}%  ({df['any_pass'].sum()}/{len(df)})")
    print(f"M31 alone: {df['M31'].mean()*100:.2f}%")
    print(f"M01 alone: {df['M01'].mean()*100:.2f}%")
    print("\nMethods that uniquely recover at least one tree (none of the others do):")
    cols = [c for c in POOL.keys()]
    unique_save = {}
    for m in cols:
        # trees where m wins but no other does
        mask = (df[m] == 1)
        others = [c for c in cols if c != m]
        none_others = (df[others].sum(axis=1) == 0)
        unique_save[m] = int((mask & none_others).sum())
    for k, v in sorted(unique_save.items(), key=lambda kv: -kv[1]):
        if v > 0:
            print(f"  {k}: {v}")

    print("\nM31 fail trees, ordered by which method recovers them:")
    m31_fail = df[df["M31"] == 0]
    print(f"M31 fails: {len(m31_fail)}  oracle-recoverable: {m31_fail['any_pass'].sum()}")
    print("\nWinning method counts among M31 failures:")
    recovered = m31_fail[m31_fail["any_pass"] == 1]
    counts = pd.Series([w for ws in recovered["winners"] for w in ws.split(",")]).value_counts()
    print(counts.to_string())


if __name__ == "__main__":
    main()
