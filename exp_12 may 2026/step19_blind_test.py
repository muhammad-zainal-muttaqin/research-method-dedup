"""
Strict held-out test protocol.

Selection split: TRAIN + VAL only (n=775)
Blind split:     TEST only (n=166) — never inspected during cut selection

Procedure:
  1. Scan all candidate override cuts (same families as step17).
  2. Adopt only those where alternative beats M31 on BOTH train AND val
     (bilateral gain, no test inspection).
  3. Compose final method (M60_blind_strict) from the adopted cuts.
  4. Evaluate on TEST ALONE for the first time.

Outputs:
  out/blind_candidate_scan.csv  — all candidate cuts + gains on TRAIN/VAL
  out/blind_adopted_cuts.csv    — cuts that pass bilateral-gain filter
  out/blind_final_summary.csv   — M01, M31, M52, M53, M60 on test alone
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(EXP))

from harness import NAMES, OUT_DIR, evaluate, load_trees, naive_count
from methods import (
    n_sides_observed,
    m31_side_aware_selector,
    m33_refined_divide,
    m30_side_aware_divide,
    m52_two_band_override,
    m53_three_band_override,
)
from algorithms.M01_selector_b2b3 import predict as m01
from algorithms.M03_blend_geometric import predict as m03
from algorithms.M16_boost_b2b4 import predict as m16
from algorithms.M19_divide_adaptive import predict as m19
from algorithms.M07_weight_coverage import predict as m07


CANDIDATE_METHODS = {
    "M31": m31_side_aware_selector,
    "M33": m33_refined_divide,
    "M30": m30_side_aware_divide,
    "M01": m01,
    "M03": m03,
    "M16": m16,
    "M19": m19,
    "M07": m07,
}


def within1(p, g):
    return all(abs(p[c] - g[c]) <= 1 for c in NAMES)


def build_feature_table(trees):
    rows = []
    for t in trees:
        n_total = len(t.dets)
        ns = n_sides_observed(t.dets)
        naive = naive_count(t.dets)
        b3frac = naive["B3"] / max(n_total, 1)
        b4frac = naive["B4"] / max(n_total, 1)
        row = {
            "tree_id": t.tree_id, "split": t.split,
            "ns": ns, "n_total": n_total,
            "b3frac": b3frac, "b4frac": b4frac,
            "b1": naive["B1"], "b4": naive["B4"],
        }
        for m, fn in CANDIDATE_METHODS.items():
            row[m] = int(within1(fn(t.dets), t.gt))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    trees = load_trees()
    df = build_feature_table(trees)

    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]
    test = df[df["split"] == "test"]
    print(f"Splits: train={len(train)} val={len(val)} test={len(test)} "
          f"(blind on test, n={len(test)})")

    # ────────── candidate cut universe (same families as step17) ──────────
    candidates: List[Tuple[str, Callable[[pd.DataFrame], pd.Series]]] = []

    def add(label, predicate):
        candidates.append((label, predicate))

    # b3frac single-dim, ns=4
    for low, high in [(0.0, 0.15), (0.15, 0.30), (0.30, 0.45), (0.45, 0.60),
                      (0.60, 0.75), (0.75, 0.90), (0.90, 1.01)]:
        add(
            f"ns=4 b3frac({low:.2f},{high:.2f}]",
            (lambda lo, hi: lambda d: (d["ns"] == 4) & (d["b3frac"] > lo) & (d["b3frac"] <= hi))(low, high),
        )

    # n_total single-dim, ns=4
    for lo, hi in [(0, 8), (8, 16), (16, 25), (25, 40), (40, 999)]:
        add(
            f"ns=4 n_total({lo},{hi}]",
            (lambda l, h: lambda d: (d["ns"] == 4) & (d["n_total"] > l) & (d["n_total"] <= h))(lo, hi),
        )

    # b4frac single-dim, ns=4
    for low, high in [(0.0, 0.05), (0.05, 0.15), (0.15, 0.30), (0.30, 0.50), (0.50, 1.01)]:
        add(
            f"ns=4 b4frac({low:.2f},{high:.2f}]",
            (lambda lo, hi: lambda d: (d["ns"] == 4) & (d["b4frac"] > lo) & (d["b4frac"] <= hi))(low, high),
        )

    # joint b3frac × n_total, ns=4
    for low, high in [(0.30, 0.45), (0.45, 0.60), (0.60, 0.75), (0.75, 0.90)]:
        for lo, hi in [(0, 16), (16, 25), (25, 999)]:
            add(
                f"ns=4 b3frac({low:.2f},{high:.2f}] n_total({lo},{hi}]",
                (lambda lo3, hi3, l, h: lambda d:
                    (d["ns"] == 4) & (d["b3frac"] > lo3) & (d["b3frac"] <= hi3)
                    & (d["n_total"] > l) & (d["n_total"] <= h))(low, high, lo, hi),
            )

    # ns=8 buckets
    for lo, hi in [(0, 25), (25, 40), (40, 999)]:
        add(
            f"ns=8 n_total({lo},{hi}]",
            (lambda l, h: lambda d: (d["ns"] == 8) & (d["n_total"] > l) & (d["n_total"] <= h))(lo, hi),
        )

    # ────────── scan: gains on TRAIN and VAL only ──────────
    scan_rows = []
    for label, pred in candidates:
        mt = pred(train)
        mv = pred(val)
        nt, nv = int(mt.sum()), int(mv.sum())
        if nt < 15 or nv < 6:
            continue
        accs_t = {m: float(train[mt][m].mean() * 100) for m in CANDIDATE_METHODS}
        accs_v = {m: float(val[mv][m].mean() * 100) for m in CANDIDATE_METHODS}
        m31t, m31v = accs_t["M31"], accs_v["M31"]
        best = None
        for m in CANDIDATE_METHODS:
            if m == "M31":
                continue
            # strict: must STRICTLY beat M31 on at least one split AND not lose on the other
            if accs_t[m] >= m31t and accs_v[m] >= m31v and (accs_t[m] + accs_v[m]) > (m31t + m31v):
                gain_t = accs_t[m] - m31t
                gain_v = accs_v[m] - m31v
                if best is None or (gain_t + gain_v) > (best["gain_sum"]):
                    best = {
                        "method": m,
                        "acc_t": accs_t[m], "acc_v": accs_v[m],
                        "gain_t": gain_t, "gain_v": gain_v,
                        "gain_sum": gain_t + gain_v,
                    }
        scan_rows.append({
            "cut": label, "n_train": nt, "n_val": nv,
            "m31_train": m31t, "m31_val": m31v,
            "best_method": best["method"] if best else "",
            "best_acc_train": best["acc_t"] if best else None,
            "best_acc_val": best["acc_v"] if best else None,
            "gain_train": best["gain_t"] if best else None,
            "gain_val": best["gain_v"] if best else None,
        })
    scan_df = pd.DataFrame(scan_rows)
    scan_df.to_csv(OUT_DIR / "blind_candidate_scan.csv", index=False)

    adopted = scan_df[scan_df["best_method"] != ""].copy()
    # Sort by combined gain — pick top non-overlapping in a moment.
    adopted["combined_gain"] = adopted["gain_train"] + adopted["gain_val"]
    adopted = adopted.sort_values("combined_gain", ascending=False)
    adopted.to_csv(OUT_DIR / "blind_adopted_cuts.csv", index=False)

    print("\nCandidates passing bilateral-gain filter (train AND val both >= M31):")
    print(adopted[["cut", "best_method", "n_train", "n_val",
                   "gain_train", "gain_val", "combined_gain"]].to_string(index=False))

    # ────────── compose M60 from adopted cuts ──────────
    # Adopt cuts greedily by combined gain. Disjoint by construction since each
    # cut is a distinct (b3frac, n_total, ns) bucket; overlaps resolved by
    # first-match (greedy order).
    adopted_list = adopted.to_dict("records")

    def make_override(cut_label, method_name):
        """Parse cut_label into a tree-level predicate + method callable."""
        fn = CANDIDATE_METHODS[method_name]
        return cut_label, method_name, fn

    overrides = [make_override(r["cut"], r["best_method"]) for r in adopted_list]

    # Tree-level dispatcher built from labels (string parsing) — we re-derive
    # features per tree at evaluation time so the predicate matches the scan.
    def m60(dets):
        if not dets:
            return {c: 0 for c in NAMES}
        ns = n_sides_observed(dets)
        n_total = len(dets)
        naive = naive_count(dets)
        b3frac = naive["B3"] / max(n_total, 1)
        b4frac = naive["B4"] / max(n_total, 1)

        for cut_label, _method_name, fn in overrides:
            if _match(cut_label, ns, n_total, b3frac, b4frac):
                return fn(dets)
        return m31_side_aware_selector(dets)

    # ────────── evaluate on test alone (BLIND) ──────────
    METHODS = {
        "M01_selector_b2b3":       m01,
        "M31_side_aware_selector": m31_side_aware_selector,
        "M52_two_band_override":   m52_two_band_override,
        "M53_three_band_override": m53_three_band_override,
        "M60_blind_strict":        m60,
    }

    print(f"\n=== TEST alone (n={len(test)}) — first time test data seen ===")
    rows = []
    sub_test = [t for t in trees if t.split == "test"]
    for name, fn in METHODS.items():
        preds = {t.tree_id: fn(t.dets) for t in sub_test}
        summ = evaluate(name, preds, sub_test)["summary"]
        summ["split"] = "test"
        summ["n"] = len(sub_test)
        rows.append(summ)
    test_df = pd.DataFrame(rows).sort_values("acc_within1_pct", ascending=False)
    cols = ["method", "acc_within1_pct", "macro_class_MAE", "n_fail",
            "exact_profile_acc_pct", "total_count_MAE", "total_count_within1_pct",
            "MAE_B1", "MAE_B2", "MAE_B3", "MAE_B4",
            "bias_B1", "bias_B2", "bias_B3", "bias_B4"]
    print(test_df[cols].to_string(index=False))
    test_df.to_csv(OUT_DIR / "blind_final_summary.csv", index=False)

    # Also report full 953 + per-split summary for transparency.
    print(f"\n=== Full 953 (audit) ===")
    rows = []
    for name, fn in METHODS.items():
        preds = {t.tree_id: fn(t.dets) for t in trees}
        summ = evaluate(name, preds, trees)["summary"]
        summ["split"] = "full"
        summ["n"] = len(trees)
        rows.append(summ)
    full_df = pd.DataFrame(rows).sort_values("acc_within1_pct", ascending=False)
    print(full_df[cols].to_string(index=False))

    print(f"\n=== Per-split for M60 ===")
    for label in ["train", "val", "test", "val+test"]:
        if label == "val+test":
            sub = [t for t in trees if t.split in ("val", "test")]
        else:
            sub = [t for t in trees if t.split == label]
        preds = {t.tree_id: m60(t.dets) for t in sub}
        summ = evaluate("M60_blind_strict", preds, sub)["summary"]
        print(f"  {label} (n={len(sub)}): acc±1 = {summ['acc_within1_pct']:.2f}%  "
              f"macro_MAE = {summ['macro_class_MAE']:.3f}")


def _match(label: str, ns: int, n_total: int, b3frac: float, b4frac: float) -> bool:
    """Re-evaluate the predicate captured in the cut label string."""
    parts = label.split()
    # ns clause
    if not parts[0].startswith("ns="):
        return False
    if int(parts[0].split("=")[1]) != ns:
        return False
    # parse each subsequent feature clause
    for seg in parts[1:]:
        # seg looks like  "b3frac(0.30,0.45]"  or  "n_total(16,25]"
        key, rng = seg.split("(", 1)
        lo_s, hi_s = rng.rstrip("]").split(",")
        lo = float(lo_s)
        hi = float(hi_s)
        if key == "b3frac":
            v = b3frac
        elif key == "b4frac":
            v = b4frac
        elif key == "n_total":
            v = n_total
        else:
            return False
        if not (lo < v <= hi):
            return False
    return True


if __name__ == "__main__":
    main()
