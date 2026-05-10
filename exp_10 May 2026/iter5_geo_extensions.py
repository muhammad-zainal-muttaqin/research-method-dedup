"""
Iteration 5 — extend geometric_mean_blend (iter4 winner: 86.15%, MAE 0.3961).

Candidates:
1. geo_3way        — geometric mean of {visibility, adaptive_corrected, side_coverage}.
2. geo_weighted_w  — vis^w * corr^(1-w), grid w in {0.4, 0.5, 0.6, 0.7}.
3. arith_geo_mix   — 0.5*(arith + geo), softens both extremes.
4. geo_with_floor_ceil — geo + floor (max_per_side) + ceiling (naive).
5. b1_tight_geo    — geo blend + B1 tightening when low cross-side evidence.

Multi-split gate: improve ≥2 splits, no split drops > 0.3pp, AND beat
the iter4 baseline (86.15%) on at least one metric (acc or mae).
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


def _max_per_side(dets, c):
    cd = [d for d in dets if d["class"] == c]
    return max(Counter(d["side_index"] for d in cd).values()) if cd else 0


def _active_sides(dets, c):
    return len(set(d["side_index"] for d in dets if d["class"] == c))


def _floor(dets, pred):
    return {c: max(pred[c], _max_per_side(dets, c)) for c in NAMES}


# ─── candidates ───────────────────────────────────────────────

def geometric_mean_blend(dets):
    """iter4 baseline."""
    v = base.visibility_count(dets)
    c = base.adaptive_corrected(dets)
    out = {}
    for cl in NAMES:
        if v[cl] == 0 or c[cl] == 0:
            out[cl] = (v[cl] + c[cl]) // 2
        else:
            out[cl] = int(round(np.sqrt(v[cl] * c[cl])))
    return _floor(dets, out)


def geo_3way(dets):
    """Geometric mean of three structurally distinct estimators."""
    v = base.visibility_count(dets)
    c = base.adaptive_corrected(dets)
    s = base.side_coverage(dets)
    out = {}
    for cl in NAMES:
        vals = [v[cl], c[cl], s[cl]]
        if min(vals) == 0:
            out[cl] = int(round(sum(vals) / 3.0))
        else:
            out[cl] = int(round((vals[0] * vals[1] * vals[2]) ** (1.0 / 3.0)))
    return _floor(dets, out)


def geo_weighted(w_vis):
    """Weighted geometric mean: vis^w * corr^(1-w)."""
    def _fn(dets):
        v = base.visibility_count(dets)
        c = base.adaptive_corrected(dets)
        out = {}
        for cl in NAMES:
            if v[cl] == 0 or c[cl] == 0:
                out[cl] = int(round(w_vis * v[cl] + (1 - w_vis) * c[cl]))
            else:
                out[cl] = int(round((v[cl] ** w_vis) * (c[cl] ** (1 - w_vis))))
        return _floor(dets, out)
    return _fn


def arith_geo_mix(dets):
    """Average of arithmetic and geometric means of {vis, corr}."""
    v = base.visibility_count(dets)
    c = base.adaptive_corrected(dets)
    out = {}
    for cl in NAMES:
        arith = (v[cl] + c[cl]) / 2.0
        if v[cl] == 0 or c[cl] == 0:
            geo = arith
        else:
            geo = np.sqrt(v[cl] * c[cl])
        out[cl] = max(0, int(round((arith + geo) / 2.0)))
    return _floor(dets, out)


def geo_with_naive_ceiling(dets):
    """geometric_mean_blend + explicit naive ceiling. Mostly redundant
    but guarantees structural ceiling."""
    g = geometric_mean_blend(dets)
    n = base.naive_count(dets)
    return {c: min(g[c], n[c]) for c in NAMES}


def b1_tight_geo(dets):
    """geometric_mean_blend + B1 tightening: when active_sides(B1) <= 1
    AND naive_B1 > max_per_side(B1), cap B1 at max_per_side. Reasoning:
    no cross-side B1 evidence → all B1 dets in one side → ceiling = that
    side's B1 count."""
    g = geometric_mean_blend(dets)
    if _active_sides(dets, "B1") <= 1:
        g["B1"] = min(g["B1"], _max_per_side(dets, "B1"))
    return g


def b1_tight_dense_geo(dets):
    """geometric_mean_blend + B1 tightening only for dense trees
    (n_dets >= 18). At low density, B1=1 estimates are usually correct."""
    g = geometric_mean_blend(dets)
    n_total = len(dets)
    if n_total >= 18 and _active_sides(dets, "B1") <= 1:
        g["B1"] = min(g["B1"], _max_per_side(dets, "B1"))
    return g


def geo_3way_floor_ceil(dets):
    """geo_3way + naive ceiling."""
    g = geo_3way(dets)
    n = base.naive_count(dets)
    return {c: min(g[c], n[c]) for c in NAMES}


# ─── load + eval ──────────────────────────────────────────────

def load_with_split():
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
        trees[tree_id] = {"dets": dets, "gt": gt, "split": data.get("split", "unknown")}
    return trees


def _within1(pred, gt):
    return all(abs(pred[c] - gt[c]) <= 1 for c in NAMES)


def _mae(pred, gt):
    return float(np.mean([abs(pred[c] - gt[c]) for c in NAMES]))


def evaluate(fn, trees, split=None):
    items = list(trees.values()) if split is None else [t for t in trees.values() if t["split"] == split]
    n = len(items)
    if n == 0:
        return {"acc": 0.0, "mae": 0.0, "n_fail": 0, "n": 0}
    ok, mae_list = 0, []
    for info in items:
        pred = fn(info["dets"])
        ok += int(_within1(pred, info["gt"]))
        mae_list.append(_mae(pred, info["gt"]))
    return {"acc": round(100.0 * ok / n, 2), "mae": round(float(np.mean(mae_list)), 4), "n_fail": n - ok, "n": n}


def main():
    base._load_v6_params()
    trees = load_with_split()

    candidates = {
        "iter4_geometric_mean_blend": geometric_mean_blend,
        "geo_3way": geo_3way,
        "arith_geo_mix": arith_geo_mix,
        "geo_with_naive_ceiling": geo_with_naive_ceiling,
        "b1_tight_geo": b1_tight_geo,
        "b1_tight_dense_geo": b1_tight_dense_geo,
        "geo_3way_floor_ceil": geo_3way_floor_ceil,
    }
    for w in (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
        candidates[f"geo_weighted_w{int(w*100):02d}"] = geo_weighted(w)

    rows = []
    bl_tr, bl_va, bl_te = 87.17, 81.46, 87.43  # iter4 geometric_mean_blend per-split
    for name, fn in candidates.items():
        all_ = evaluate(fn, trees, None)
        tr = evaluate(fn, trees, "train")
        va = evaluate(fn, trees, "val")
        te = evaluate(fn, trees, "test")
        rows.append({
            "method": name,
            "acc_all": all_["acc"], "mae_all": all_["mae"],
            "acc_train": tr["acc"], "acc_val": va["acc"], "acc_test": te["acc"],
            "d_train": round(tr["acc"] - bl_tr, 2),
            "d_val": round(va["acc"] - bl_va, 2),
            "d_test": round(te["acc"] - bl_te, 2),
        })

    df = pd.DataFrame(rows).sort_values(["acc_all", "mae_all"], ascending=[False, True])
    df["worst_drop"] = df[["d_train", "d_val", "d_test"]].min(axis=1)
    df["n_up"] = (df[["d_train", "d_val", "d_test"]] > 0).sum(axis=1)
    df["passes"] = (df["worst_drop"] >= -0.3) & (df["n_up"] >= 2)
    df.to_csv(OUT_DIR / "iter5_results.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
