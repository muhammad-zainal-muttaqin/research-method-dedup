"""
step20_pure_physics.py — purely algorithmic push toward Acc±1 ≥ 90%

All methods in this file satisfy RULES.txt: no GT-derived constants, no
divisor tables, no BASE_FACTORS, no thresholds chosen by scanning accuracy.

Physical model used:
  Hemisphere visibility: a fruit bunch grows on one side of the trunk and is
  visible from approximately half the cameras that can physically see it.
  Therefore: expected_appearances ≈ active_cameras_for_class / 2

Observable inputs (all GT-free):
  - n_sides       : number of distinct side_index values
  - active_sides_c: number of sides with at least one detection of class c
  - naive_c       : raw detection count for class c
  - x_norm        : horizontal normalised position of each bbox
  - max_per_side_c: maximum detections of class c on any single side
                    (physical floor: unique count cannot be less than this)

Gaussian sigma=0.3 is listed as an ALLOWED generic spatial prior in RULES.txt.
"""

from __future__ import annotations

import sys
from collections import Counter
from math import exp, sqrt
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
EXP  = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(EXP))

from harness import NAMES, OUT_DIR, evaluate, load_trees, run, split_breakdown

# ─────────────────────────────────────────────────────────────────
# Shared primitives
# ─────────────────────────────────────────────────────────────────

def _naive(dets: list) -> dict:
    c = Counter(d["class"] for d in dets)
    return {cl: int(c.get(cl, 0)) for cl in NAMES}


def _n_sides(dets: list) -> int:
    return len({d["side_index"] for d in dets}) if dets else 0


def _active_sides(dets: list, cl: str) -> int:
    return len({d["side_index"] for d in dets if d["class"] == cl})


def _max_per_side(dets: list, cl: str) -> int:
    cd = [d for d in dets if d["class"] == cl]
    return int(max(Counter(d["side_index"] for d in cd).values())) if cd else 0


_ALPHA = 1.0
_SIGMA = 0.3  # generic spatial prior, RULES.txt §ALLOWED


def _gauss_weight(x: float) -> float:
    return 1.0 / (1.0 + _ALPHA * exp(-((x - 0.5) ** 2) / (2.0 * _SIGMA ** 2)))


def _vis(dets: list, cl: str) -> float:
    """Gaussian visibility sum for class cl (raw, before any n_sides scaling)."""
    return sum(_gauss_weight(d["x_norm"]) for d in dets if d["class"] == cl)


# ─────────────────────────────────────────────────────────────────
# M70 — n_sides-scaled Gaussian visibility
# ─────────────────────────────────────────────────────────────────
#
# Physical argument:
#   The Gaussian visibility estimator (M06) was designed for trees with 4
#   camera sides.  With 8 sides, each bunch appears in twice as many frames,
#   so the raw visibility sum is ~2× the true unique count.  Dividing by
#   n_sides/4 (= multiplying by 4/n_sides) normalises back to the 4-side
#   reference frame.
#
#   For n_sides < 4 the scale stays at 1 (no upward inflation): trees with
#   fewer cameras have less duplication, and the Gaussian floor + max_per_side
#   floor are adequate.
#
#   No GT constant anywhere: 4 is the structural reference (360°/90° = 4
#   cameras standard protocol, observable from the data itself).

def m70_vis_nscaled(dets: list) -> dict:
    if not dets:
        return {c: 0 for c in NAMES}
    ns    = _n_sides(dets)
    scale = 4.0 / max(ns, 4)          # 1.0 for ns≤4, 0.5 for ns=8
    out   = {}
    for c in NAMES:
        raw = _vis(dets, c)
        est = round(raw * scale)
        out[c] = max(est, _max_per_side(dets, c))
    return out


# ─────────────────────────────────────────────────────────────────
# M71 — per-class active-sides hemisphere divisor
# ─────────────────────────────────────────────────────────────────
#
# Physical argument:
#   Apply the hemisphere model (divisor = n_active_sides/2) per class rather
#   than globally.  This captures the fact that ripe (B4) bunches are visible
#   from fewer camera angles than unripe ones: fewer active sides → lower
#   divisor → less deduplication, which matches the empirical pattern without
#   any GT calibration.
#
#   Floor ensures divisor ≥ 1 (no "negative dedup" for single-side classes).
#   max_per_side floor is always applied as a hard physical constraint.

def m71_active_hemisphere(dets: list) -> dict:
    if not dets:
        return {c: 0 for c in NAMES}
    naive = _naive(dets)
    out   = {}
    for c in NAMES:
        if naive[c] == 0:
            out[c] = 0
            continue
        act     = _active_sides(dets, c)
        divisor = max(act / 2.0, 1.0)
        est     = round(naive[c] / divisor)
        out[c]  = max(est, _max_per_side(dets, c))
    return out


# ─────────────────────────────────────────────────────────────────
# M72 — geometric mean blend of M70 and M71, with max_per_side floor
# ─────────────────────────────────────────────────────────────────
#
# Rationale: M70 (spatial visibility) and M71 (active-sides hemisphere)
# capture complementary aspects of uniqueness.  Their geometric mean gives
# a balanced estimator that is more conservative than either alone.

def m72_blend_vis_active(dets: list) -> dict:
    if not dets:
        return {c: 0 for c in NAMES}
    a   = m70_vis_nscaled(dets)
    b   = m71_active_hemisphere(dets)
    out = {}
    for c in NAMES:
        v, h = float(a[c]), float(b[c])
        if v == 0 or h == 0:
            blend = (v + h) / 2.0
        else:
            blend = sqrt(v * h)
        out[c] = max(round(blend), _max_per_side(dets, c))
    return out


# ─────────────────────────────────────────────────────────────────
# M73 — visibility with n_sides scale + B2↔B3 reallocation
# ─────────────────────────────────────────────────────────────────
#
# The B2↔B3 correction from M01 is ALLOWED: it does not use any GT-calibrated
# constant.  It only uses the naive B2:B3 ratio to reallocate a fixed joint
# total (B2+B3) estimated by M70.  This addresses the known B2↔B3 visual
# ambiguity without any dataset calibration.

def _b2b3_realloc(pred: dict, dets: list) -> dict:
    """Reallocate pred[B2]+pred[B3] by naive B2:B3 ratio. No GT constants."""
    joint = pred["B2"] + pred["B3"]
    if joint == 0:
        return pred
    b23 = [d for d in dets if d["class"] in ("B2", "B3")]
    if not b23:
        return pred
    n_b3 = sum(1 for d in b23 if d["class"] == "B3")
    n_b2 = len(b23) - n_b3
    if n_b2 + n_b3 == 0:
        return pred
    frac_b3 = n_b3 / (n_b2 + n_b3)
    new_b3  = int(round(joint * frac_b3))
    new_b2  = joint - new_b3
    out = dict(pred)
    out["B2"] = max(new_b2, _max_per_side(dets, "B2"))
    out["B3"] = max(new_b3, _max_per_side(dets, "B3"))
    return out


def m73_vis_nscaled_b2b3(dets: list) -> dict:
    return _b2b3_realloc(m70_vis_nscaled(dets), dets)


def m74_active_hemisphere_b2b3(dets: list) -> dict:
    return _b2b3_realloc(m71_active_hemisphere(dets), dets)


def m75_blend_b2b3(dets: list) -> dict:
    return _b2b3_realloc(m72_blend_vis_active(dets), dets)


# ─────────────────────────────────────────────────────────────────
# M76 — min-conservative blend: take min(M70, M71), floor max_per_side
# ─────────────────────────────────────────────────────────────────
#
# A more conservative estimator: take the smaller of the two independent
# estimates, then apply max_per_side floor.  Reduces overcounting at the cost
# of occasionally undercounting.

def m76_min_conservative(dets: list) -> dict:
    if not dets:
        return {c: 0 for c in NAMES}
    a = m70_vis_nscaled(dets)
    b = m71_active_hemisphere(dets)
    out = {}
    for c in NAMES:
        est    = min(a[c], b[c])
        out[c] = max(est, _max_per_side(dets, c))
    return out


# ─────────────────────────────────────────────────────────────────
# Baseline for comparison (imported, not re-implemented)
# ─────────────────────────────────────────────────────────────────

from algorithms.M01_selector_b2b3 import predict as m01
from algorithms.M06_weight_visibility import predict as m06


# ─────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────

METHODS = {
    "M01_selector_b2b3":         m01,
    "M06_weight_visibility":     m06,
    "M70_vis_nscaled":           m70_vis_nscaled,
    "M71_active_hemisphere":     m71_active_hemisphere,
    "M72_blend_vis_active":      m72_blend_vis_active,
    "M73_vis_nscaled_b2b3":      m73_vis_nscaled_b2b3,
    "M74_active_hemisphere_b2b3": m74_active_hemisphere_b2b3,
    "M75_blend_b2b3":            m75_blend_b2b3,
    "M76_min_conservative":      m76_min_conservative,
}

COLS = [
    "method", "acc_within1_pct", "macro_class_MAE", "n_fail",
    "exact_profile_acc_pct", "total_count_MAE", "total_count_within1_pct",
    "MAE_B1", "MAE_B2", "MAE_B3", "MAE_B4",
    "bias_B1", "bias_B2", "bias_B3", "bias_B4",
]


def main() -> None:
    # purge prior split csv to avoid append-stacking
    split_csv = OUT_DIR / "split_step20.csv"
    if split_csv.exists():
        split_csv.unlink()

    trees = load_trees()
    print(f"Loaded {len(trees)} trees")

    summary = run(METHODS, trees, tag="step20")
    print("\n=== FULL 953-TREE RESULTS ===")
    print(summary[COLS].to_string(index=False))
    print()

    # Per-split breakdown
    print("=== PER-SPLIT BREAKDOWN ===")
    per_tree_df = pd.read_csv(OUT_DIR / "per_tree_step20.csv")
    for split in ("train", "val", "test"):
        sub = [t for t in trees if t.split == split]
        if not sub:
            continue
        print(f"\n--- {split} (n={len(sub)}) ---")
        rows = []
        for name, fn in METHODS.items():
            preds = {t.tree_id: fn(t.dets) for t in sub}
            res   = evaluate(name, preds, sub)
            rows.append(res["summary"])
        df = pd.DataFrame(rows).sort_values("acc_within1_pct", ascending=False)
        print(df[["method", "acc_within1_pct", "macro_class_MAE", "n_fail",
                   "MAE_B4", "bias_B4"]].to_string(index=False))

    # ns=8 vs ns=4 breakdown for the two best new methods
    print("\n=== ns=8 vs ns=4 BREAKDOWN ===")
    ns8_trees  = [t for t in trees if len({d["side_index"] for d in t.dets}) >= 5]
    ns4_trees  = [t for t in trees if len({d["side_index"] for d in t.dets}) < 5]
    print(f"ns=8 trees: {len(ns8_trees)},  ns<=4 trees: {len(ns4_trees)}")

    for label, subset in [("ns=8", ns8_trees), ("ns<=4", ns4_trees)]:
        rows = []
        for name, fn in METHODS.items():
            preds = {t.tree_id: fn(t.dets) for t in subset}
            res   = evaluate(name, preds, subset)
            rows.append(res["summary"])
        df = pd.DataFrame(rows).sort_values("acc_within1_pct", ascending=False)
        print(f"\n{label}:")
        print(df[["method", "acc_within1_pct", "macro_class_MAE",
                   "n_fail", "MAE_B4"]].to_string(index=False))


if __name__ == "__main__":
    main()
