"""
M60_blind_strict — Acc±1 90.24% (953 trees), 91.57% test-blind
================================================================
Full 953-tree canonical benchmark result (2026-05-12).
Cut selection: train+val only. Test never seen during selection.

Architecture
------------
1. Compute tree features: ns, n_total, naive counts, b3frac, b4frac.
2. Greedy first-match through 11 override cuts (selected on train+val).
3. Fallback: M31_side_aware_selector (ns>=5 → 2-D divisor; else trifurc).

Divisor tables (train-only medians)
------------------------------------
Loaded from CSV at first call. Two search paths tried in order:
  1. Same directory as this file  (after copying CSVs alongside the module)
  2. ../exp_12 may 2026/out/      (repo working location)

Interface
---------
    from algorithms.M60_blind_strict import predict
    result = predict(detections)   # {"B1": int, "B2": int, "B3": int, "B4": int}

detections: list[dict] with keys  class (B1-B4), x_norm, y_norm, side_index
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

NAMES = ("B1", "B2", "B3", "B4")

# ── divisor CSV lookup ────────────────────────────────────────────────────────

_HERE = Path(__file__).resolve().parent
_EXP_OUT = _HERE.parent / "exp_12 may 2026" / "out"

def _find_csv(name: str) -> Path:
    local = _HERE / name
    if local.exists():
        return local
    exp = _EXP_OUT / name
    if exp.exists():
        return exp
    raise FileNotFoundError(
        f"{name} not found in {_HERE} or {_EXP_OUT}. "
        "Run exp_12 may 2026/step04_side_factor.py and step09_refined_table.py first."
    )

_MIN_SUPPORT_2D = 20
_MIN_SUPPORT_3D = 12
COUNT_BINS = [0, 3, 6, 10, 15, 25, 1000]
_FALLBACK_2D = {"B1": 2.0, "B2": 2.0, "B3": 1.86, "B4": 1.6}


def _bucket(n: int) -> int:
    for i in range(len(COUNT_BINS) - 1):
        if COUNT_BINS[i] < n <= COUNT_BINS[i + 1]:
            return i
    return len(COUNT_BINS) - 2


@lru_cache(maxsize=1)
def _div2d() -> Dict[int, Dict[str, float]]:
    df = pd.read_csv(_find_csv("divisor_2d.csv"))
    table: Dict[int, Dict[str, float]] = {}
    for _, row in df.iterrows():
        if row["count"] >= _MIN_SUPPORT_2D:
            table.setdefault(int(row["n_sides"]), {})[row["class"]] = float(row["median"])
    return table


@lru_cache(maxsize=1)
def _div3d() -> Dict[int, Dict[str, Dict[int, float]]]:
    df = pd.read_csv(_find_csv("divisor_3d.csv"))
    table: Dict[int, Dict[str, Dict[int, float]]] = {}
    for _, row in df.iterrows():
        if row["count"] >= _MIN_SUPPORT_3D:
            ns = int(row["n_sides"])
            cl = row["class"]
            bk = int(row["naive_bucket"])
            table.setdefault(ns, {}).setdefault(cl, {})[bk] = float(row["median"])
    return table


# ── shared utilities ──────────────────────────────────────────────────────────


def _naive(dets: List[dict]) -> Dict[str, int]:
    c = Counter(d["class"] for d in dets)
    return {cl: int(c.get(cl, 0)) for cl in NAMES}


def _max_per_side(dets: List[dict], cl: str) -> int:
    cd = [d for d in dets if d["class"] == cl]
    return int(max(Counter(d["side_index"] for d in cd).values())) if cd else 0


def _n_sides(dets: List[dict]) -> int:
    return len({d["side_index"] for d in dets}) if dets else 0


# ── divisor estimators ────────────────────────────────────────────────────────


def _side_factor(ns: int, cl: str) -> float:
    """2-D (n_sides, class) median divisor with fallback."""
    cell = _div2d().get(ns, {})
    if cl in cell:
        return cell[cl]
    four = _div2d().get(4, {})
    if cl in four:
        return four[cl] if cl == "B4" else four[cl] * max(1.0, ns / 4.0)
    return _FALLBACK_2D[cl]


def _refined_factor(ns: int, cl: str, naive_cl: int) -> float:
    """3-D (n_sides, class, naive_bucket) divisor; falls back to 2-D."""
    bk = _bucket(naive_cl)
    cell = _div3d().get(ns, {}).get(cl, {})
    return cell[bk] if bk in cell else _side_factor(ns, cl)


def _side_aware_divide(dets: List[dict]) -> Dict[str, int]:
    """M30/M31 base: naive[c] / 2-D median, floored by max_per_side."""
    ns = _n_sides(dets)
    naive = _naive(dets)
    out: Dict[str, int] = {}
    for c in NAMES:
        f = _side_factor(ns, c)
        est = int(round(naive[c] / f)) if f > 0 else naive[c]
        out[c] = max(est, _max_per_side(dets, c))
    return out


def _refined_divide(dets: List[dict]) -> Dict[str, int]:
    """M33: naive[c] / 3-D median, floored by max_per_side."""
    ns = _n_sides(dets)
    naive = _naive(dets)
    out: Dict[str, int] = {}
    for c in NAMES:
        f = _refined_factor(ns, c, naive[c])
        est = int(round(naive[c] / f)) if f > 0 else naive[c]
        out[c] = max(est, _max_per_side(dets, c))
    return out


# ── base component estimators (used by M31 trifurcation) ─────────────────────

_BASE = {"B1": 1.986, "B2": 1.786, "B3": 1.795, "B4": 1.655}


def _adaptive_corrected(dets: List[dict]) -> Dict[str, int]:
    n_total = len(dets)
    scale = float(np.clip(2.05 - 0.014 * n_total, 1.45, 2.10)) / 1.79
    n = _naive(dets)
    return {c: max(0, int(round(n[c] / (_BASE[c] * scale)))) for c in NAMES}


def _visibility_count(dets: List[dict], alpha: float = 1.0, sigma: float = 0.3) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for c in NAMES:
        cd = [d for d in dets if d["class"] == c]
        if not cd:
            out[c] = 0
            continue
        total = sum(
            1.0 / (1.0 + alpha * np.exp(-((d["x_norm"] - 0.5) ** 2) / (2.0 * sigma ** 2)))
            for d in cd
        )
        out[c] = max(0, int(round(total)))
    return out


def _side_coverage(dets: List[dict]) -> Dict[str, int]:
    vis = _visibility_count(dets)
    n = _naive(dets)
    out: Dict[str, int] = {}
    for c in NAMES:
        cd = [d for d in dets if d["class"] == c]
        if not cd:
            out[c] = 0
            continue
        out[c] = min(max(vis[c], _max_per_side(dets, c)), n[c])
    return out


def _geometric_mean_blend(dets: List[dict]) -> Dict[str, int]:
    """M03 base estimator: geometric mean of visibility and adaptive_corrected."""
    v = _visibility_count(dets)
    c = _adaptive_corrected(dets)
    out: Dict[str, int] = {}
    for cl in NAMES:
        if v[cl] == 0 or c[cl] == 0:
            out[cl] = (v[cl] + c[cl]) // 2
        else:
            out[cl] = int(round(np.sqrt(v[cl] * c[cl])))
    return {cl: max(out[cl], _max_per_side(dets, cl)) for cl in NAMES}


def _median3_floor(dets: List[dict]) -> Dict[str, int]:
    a = _visibility_count(dets)
    b = _adaptive_corrected(dets)
    s = _side_coverage(dets)
    out = {cl: sorted([a[cl], b[cl], s[cl]])[1] for cl in NAMES}
    return {cl: max(out[cl], _max_per_side(dets, cl)) for cl in NAMES}


# ── override estimators ───────────────────────────────────────────────────────

_STACK_REF = {"B1": 42.0, "B2": 56.0, "B3": 72.0, "B4": 50.0}
_STACK_COEFF = 0.0008
_CLASS_BOOST = {"B1": 1.0, "B2": 1.10, "B3": 1.0, "B4": 1.08}
_MIN_DUP = 1.10


def _m16_boost_b2b4(dets: List[dict]) -> Dict[str, int]:
    """M16: adaptive_corrected with B2×1.10 / B4×1.08 divisor boost + bracket."""
    n_total = len(dets)
    scale = float(np.clip(2.05 - 0.014 * n_total, 1.45, 2.10)) / 1.79
    naive = _naive(dets)
    # per-class vertical-span density
    stack: Dict[str, float] = {}
    for c in NAMES:
        cd = [d for d in dets if d["class"] == c]
        if not cd:
            stack[c] = 0.0
            continue
        y = [d["y_norm"] for d in cd]
        span = max(y) - min(y) if len(y) > 1 else 0.1
        stack[c] = len(cd) / max(span, 0.05)
    raw: Dict[str, int] = {}
    for c in NAMES:
        nc = naive[c]
        if nc == 0:
            raw[c] = 0
            continue
        stack_extra = 1.0 + _STACK_COEFF * max(0.0, stack[c] - _STACK_REF[c])
        divisor = _BASE[c] * scale * stack_extra * _CLASS_BOOST[c]
        raw[c] = max(0, round(nc / divisor))
    # bracket: floor=max_per_side, ceiling=max(floor, naive/1.10)
    out: Dict[str, int] = {}
    for c in NAMES:
        cd = [d for d in dets if d["class"] == c]
        if not cd:
            out[c] = 0
            continue
        floor = _max_per_side(dets, c)
        ceiling = max(floor, round(naive[c] / _MIN_DUP))
        out[c] = int(np.clip(raw[c], floor, ceiling))
    return out


def _m19_divide_adaptive(dets: List[dict]) -> Dict[str, int]:
    """M19: adaptive_corrected only (no bracket)."""
    return _adaptive_corrected(dets)


def _m07_weight_coverage(dets: List[dict]) -> Dict[str, int]:
    """M07: visibility clamped to [max_per_side, naive]."""
    vis = _visibility_count(dets)
    naive = _naive(dets)
    out: Dict[str, int] = {}
    for c in NAMES:
        cd = [d for d in dets if d["class"] == c]
        if not cd:
            out[c] = 0
            continue
        floor = _max_per_side(dets, c)
        out[c] = min(max(vis[c], floor), naive[c])
    return out


def _m03_blend_geometric(dets: List[dict]) -> Dict[str, int]:
    return _geometric_mean_blend(dets)


# ── M31 fallback selector ─────────────────────────────────────────────────────


def _m31(dets: List[dict]) -> Dict[str, int]:
    ns = _n_sides(dets)
    n_total = len(dets)
    if ns >= 5:
        return _side_aware_divide(dets)
    naive = _naive(dets)
    b3frac = naive["B3"] / max(n_total, 1)
    if b3frac >= 0.60 and n_total >= 25:
        return _median3_floor(dets)
    if naive["B1"] >= 3 and b3frac < 0.45 and naive["B4"] < 10:
        return _adaptive_corrected(dets)
    return _geometric_mean_blend(dets)


# ── M60 override table (11 cuts, greedy first-match, sorted by combined gain) ─
#
#   Each tuple: (ns, feature, lo, hi, [feature2, lo2, hi2], method_fn)
#   where feature ∈ {"b3frac", "b4frac", "n_total"} and range is (lo, hi].
#   Conditions joined with AND; first matching cut wins.
#
# Selection source: step19_blind_test.py on train+val only (test blind).

_OVERRIDES = [
    # cut 1:  ns=4 b3frac(0.75,0.90] n_total(16,25]   → M33  gain: +17.4 / 0.0
    (4, "b3frac", 0.75, 0.90, "n_total", 16,  25,  _refined_divide),
    # cut 2:  ns=4 b3frac(0.45,0.60] n_total(0,16]    → M33  gain: 0.0 / +10.0
    (4, "b3frac", 0.45, 0.60, "n_total",  0,  16,  _refined_divide),
    # cut 3:  ns=4 b3frac(0.30,0.45] n_total(16,25]   → M16  gain: +1.5 / +5.0
    (4, "b3frac", 0.30, 0.45, "n_total", 16,  25,  _m16_boost_b2b4),
    # cut 4:  ns=4 b3frac(0.45,0.60]                  → M33  gain: +0.7 / +5.6
    (4, "b3frac", 0.45, 0.60, None,       None, None, _refined_divide),
    # cut 5:  ns=4 b4frac(0.30,0.50]                  → M33  gain: 0.0 / +6.3
    (4, "b4frac", 0.30, 0.50, None,       None, None, _refined_divide),
    # cut 6:  ns=4 b3frac(0.75,0.90]                  → M33  gain: +3.6 / 0.0
    (4, "b3frac", 0.75, 0.90, None,       None, None, _refined_divide),
    # cut 7:  ns=4 b3frac(0.30,0.45] n_total(0,16]    → M19  gain: +3.4 / 0.0
    (4, "b3frac", 0.30, 0.45, "n_total",  0,  16,  _m19_divide_adaptive),
    # cut 8:  ns=4 b3frac(0.30,0.45] n_total(25,999]  → M07  gain: +3.2 / 0.0
    (4, "b3frac", 0.30, 0.45, "n_total", 25, 999,  _m07_weight_coverage),
    # cut 9:  ns=4 b3frac(0.30,0.45]                  → M16  gain: +2.4 / 0.0
    (4, "b3frac", 0.30, 0.45, None,       None, None, _m16_boost_b2b4),
    # cut 10: ns=4 b4frac(0.05,0.15]                  → M03  gain: +1.7 / 0.0
    (4, "b4frac", 0.05, 0.15, None,       None, None, _m03_blend_geometric),
    # cut 11: ns=4 b3frac(0.45,0.60] n_total(16,25]   → M33  gain: +1.1 / 0.0
    (4, "b3frac", 0.45, 0.60, "n_total", 16,  25,  _refined_divide),
]


def predict(detections: list) -> dict:
    """
    M60_blind_strict prediction.

    Parameters
    ----------
    detections : list[dict]
        Each dict must have: "class" (B1-B4), "x_norm", "y_norm", "side_index".

    Returns
    -------
    dict[str, int]  {"B1": int, "B2": int, "B3": int, "B4": int}
    """
    dets = detections
    if not dets:
        return {c: 0 for c in NAMES}

    ns = _n_sides(dets)
    n_total = len(dets)
    naive = _naive(dets)
    b3frac = naive["B3"] / max(n_total, 1)
    b4frac = naive["B4"] / max(n_total, 1)

    feat = {"b3frac": b3frac, "b4frac": b4frac, "n_total": n_total}

    for row in _OVERRIDES:
        req_ns, f1, lo1, hi1, f2, lo2, hi2, fn = row
        if ns != req_ns:
            continue
        v1 = feat[f1]
        if not (lo1 < v1 <= hi1):
            continue
        if f2 is not None:
            v2 = feat[f2]
            if not (lo2 < v2 <= hi2):
                continue
        return fn(dets)

    return _m31(dets)
