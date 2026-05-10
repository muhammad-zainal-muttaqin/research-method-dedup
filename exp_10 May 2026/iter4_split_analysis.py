"""
Iteration 4 — characterise the val/test/train splits and search for
corrections that improve all three simultaneously.

Goal: understand why val is 5.55pp below train, then find heuristics
that generalise (improve ≥2 splits, no regression > 0.5pp on any split).

Constraint (RULES.txt): no per-tree memorisation; corrections must be
structurally justified and pass multi-split validation.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
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


def floor_clamped_hybrid(dets):
    base_pred = base.hybrid_vis_corr(dets)
    return {c: max(base_pred[c], _max_per_side(dets, c)) for c in NAMES}


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
        varietas = "DAMIMAS" if "DAMIMAS" in tree_id else ("LONSUM" if "LONSUM" in tree_id else "OTHER")
        trees[tree_id] = {
            "dets": dets, "gt": gt,
            "split": data.get("split", "unknown"),
            "varietas": varietas,
        }
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
    return {
        "acc": round(100.0 * ok / n, 2),
        "mae": round(float(np.mean(mae_list)), 4),
        "n_fail": n - ok,
        "n": n,
    }


# ─── characterise splits ──────────────────────────────────────

def split_stats(trees):
    rows = []
    for split in ["train", "val", "test"]:
        ts = [t for t in trees.values() if t["split"] == split]
        if not ts:
            continue
        n_dets = [len(t["dets"]) for t in ts]
        gt_totals = [sum(t["gt"].values()) for t in ts]
        gt_per_class = {c: [t["gt"][c] for t in ts] for c in NAMES}
        var_counts = Counter(t["varietas"] for t in ts)
        rows.append({
            "split": split,
            "n_trees": len(ts),
            "DAMIMAS": var_counts.get("DAMIMAS", 0),
            "LONSUM": var_counts.get("LONSUM", 0),
            "n_dets_mean": round(float(np.mean(n_dets)), 2),
            "n_dets_med": float(np.median(n_dets)),
            "gt_total_mean": round(float(np.mean(gt_totals)), 2),
            "gt_total_med": float(np.median(gt_totals)),
            **{f"gt_{c}_mean": round(float(np.mean(gt_per_class[c])), 3) for c in NAMES},
            **{f"gt_{c}_med": float(np.median(gt_per_class[c])) for c in NAMES},
            "naive_to_gt_ratio_mean": round(float(np.mean([
                len(t["dets"]) / max(sum(t["gt"].values()), 1) for t in ts
            ])), 3),
        })
    return pd.DataFrame(rows)


# ─── candidate corrections ────────────────────────────────────

def hybrid_w(w):
    def _fn(dets):
        v = base.visibility_count(dets)
        c = base.adaptive_corrected(dets)
        out = {cl: max(0, int(round(w * v[cl] + (1 - w) * c[cl]))) for cl in NAMES}
        return {cl: max(out[cl], _max_per_side(dets, cl)) for cl in NAMES}
    return _fn


def density_floor_clamp(dets):
    """Floor clamp + density-aware visibility (n_dets > 25 → tighten sigma).
    Reasoning: dense trees over-count via wide Gaussian — narrower sigma
    reduces the visibility weight on edge detections."""
    n_total = len(dets)
    if n_total > 25:
        v = base.visibility_count(dets, alpha=1.0, sigma=0.25)
    elif n_total < 12:
        v = base.visibility_count(dets, alpha=1.0, sigma=0.35)
    else:
        v = base.visibility_count(dets)
    c = base.adaptive_corrected(dets)
    out = {cl: max(0, int(round(0.6 * v[cl] + 0.4 * c[cl]))) for cl in NAMES}
    return {cl: max(out[cl], _max_per_side(dets, cl)) for cl in NAMES}


def naive_ceiling_clamp(dets):
    """floor_clamped_hybrid + ceiling at naive_count (cannot exceed total
    observations). Structurally trivial but explicit."""
    base_pred = floor_clamped_hybrid(dets)
    n = base.naive_count(dets)
    return {c: min(base_pred[c], n[c]) for c in NAMES}


def geometric_mean_blend(dets):
    """Geometric mean of visibility and adaptive_corrected (instead of
    arithmetic). Structurally: geo mean penalises divergent estimators
    less than arithmetic. Floor-clamped."""
    v = base.visibility_count(dets)
    c = base.adaptive_corrected(dets)
    out = {}
    for cl in NAMES:
        if v[cl] == 0 or c[cl] == 0:
            out[cl] = (v[cl] + c[cl]) // 2
        else:
            out[cl] = int(round(np.sqrt(v[cl] * c[cl])))
    return {cl: max(out[cl], _max_per_side(dets, cl)) for cl in NAMES}


def median3_floor(dets):
    """Per-class median of {visibility, adaptive_corrected, side_coverage}
    + floor clamp. Median of 3 is more robust than arithmetic mean."""
    a = base.visibility_count(dets)
    b = base.adaptive_corrected(dets)
    c_ = base.side_coverage(dets)
    out = {}
    for cl in NAMES:
        vals = sorted([a[cl], b[cl], c_[cl]])
        out[cl] = vals[1]  # median
    return {cl: max(out[cl], _max_per_side(dets, cl)) for cl in NAMES}


def main():
    base._load_v6_params()
    trees = load_with_split()

    # ── split characterisation ────────────────────────────────
    print("=== Split statistics ===")
    stats_df = split_stats(trees)
    print(stats_df.to_string(index=False))
    stats_df.to_csv(OUT_DIR / "iter4_split_stats.csv", index=False)

    candidates = {
        "baseline_floor_clamped_hybrid": floor_clamped_hybrid,
        "density_floor_clamp": density_floor_clamp,
        "naive_ceiling_clamp": naive_ceiling_clamp,
        "geometric_mean_blend": geometric_mean_blend,
        "median3_floor": median3_floor,
    }
    for w in (0.55, 0.60, 0.65, 0.70, 0.75):
        candidates[f"hybrid_w{int(w*100):02d}_floor"] = hybrid_w(w)

    rows = []
    for name, fn in candidates.items():
        all_ = evaluate(fn, trees, None)
        tr = evaluate(fn, trees, "train")
        va = evaluate(fn, trees, "val")
        te = evaluate(fn, trees, "test")
        # multi-split rule: improvement if ≥2 splits up AND no split down >0.5pp
        baseline_tr = 87.01
        baseline_va = 81.46
        baseline_te = 87.43
        baseline_all = 86.04
        delta_tr = round(tr["acc"] - baseline_tr, 2)
        delta_va = round(va["acc"] - baseline_va, 2)
        delta_te = round(te["acc"] - baseline_te, 2)
        ups = sum(d > 0 for d in (delta_tr, delta_va, delta_te))
        worst_drop = min(delta_tr, delta_va, delta_te)
        passes = (ups >= 2) and (worst_drop >= -0.5)
        rows.append({
            "method": name,
            "acc_all": all_["acc"], "mae_all": all_["mae"],
            "acc_train": tr["acc"], "acc_val": va["acc"], "acc_test": te["acc"],
            "d_train": delta_tr, "d_val": delta_va, "d_test": delta_te,
            "n_up": ups, "worst_drop": worst_drop,
            "passes_multi_split": passes,
        })

    df = pd.DataFrame(rows).sort_values("acc_all", ascending=False)
    df.to_csv(OUT_DIR / "iter4_results.csv", index=False)
    print("\n=== Multi-split results (sorted by acc_all) ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
