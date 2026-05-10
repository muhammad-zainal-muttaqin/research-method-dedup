"""
Iteration 1 — ensemble heuristics on 953-tree Brand-New-Dataset-YOLO.

Goal: beat hybrid_vis_corr (86.04% Acc±1, MAE 0.408).
Constraint: 100% deterministic heuristics. No training, no learned weights.
Rule (RULES.txt): no hacks, no workarounds, no monkey patches.

Strategy:
1. Per-class ensemble of established top-5 (median, trimmed mean, weighted mean).
2. Structural floor clamp by max_per_side (lower bound: a class instance seen
   N times in one frame requires at least N unique bunches).
3. Sweep mixing weight w of hybrid_vis_corr to relocate optimum at full scale.

Output: exp_10 May 2026/iter1_results.csv + iter1_report.md.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
JSON_DIR = BASE / "Brand-New-Dataset-YOLO" / "json"
OUT_DIR = BASE / "exp_10 May 2026"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BASE / "scripts"))
import dedup_all_953 as base  # noqa: E402

NAMES = ["B1", "B2", "B3", "B4"]


# ─── candidate methods ────────────────────────────────────────

def _max_per_side_count(dets, c: str) -> int:
    cd = [d for d in dets if d["class"] == c]
    if not cd:
        return 0
    return max(Counter(d["side_index"] for d in cd).values())


def median_top5(dets):
    """Per-class median of five established top performers."""
    estimators = [
        base.visibility_count(dets),
        base.side_coverage(dets),
        base.density_scaled_vis(dets),
        base.adaptive_corrected(dets),
        base.hybrid_vis_corr(dets),
    ]
    return {c: int(median(e[c] for e in estimators)) for c in NAMES}


def trimmed_mean5(dets):
    """Per-class mean of five estimators after dropping the min and max."""
    estimators = [
        base.visibility_count(dets),
        base.side_coverage(dets),
        base.density_scaled_vis(dets),
        base.adaptive_corrected(dets),
        base.hybrid_vis_corr(dets),
    ]
    out = {}
    for c in NAMES:
        vals = sorted(e[c] for e in estimators)
        trimmed = vals[1:-1]  # drop min and max → 3 middle values
        out[c] = max(0, int(round(sum(trimmed) / len(trimmed))))
    return out


def floor_clamped_hybrid(dets):
    """hybrid_vis_corr with structural floor: count cannot fall below the
    maximum number of class instances observed in any single side. A bunch
    cannot be seen twice in one frame, so two B3 in side 1 → at least 2
    distinct bunches."""
    base_pred = base.hybrid_vis_corr(dets)
    return {c: max(base_pred[c], _max_per_side_count(dets, c)) for c in NAMES}


def floor_clamped_vis(dets):
    """visibility with the same structural floor."""
    base_pred = base.visibility_count(dets)
    return {c: max(base_pred[c], _max_per_side_count(dets, c)) for c in NAMES}


def triple_avg(dets):
    """Equal-weight mean of three structurally distinct estimators."""
    a = base.visibility_count(dets)
    b = base.adaptive_corrected(dets)
    c_ = base.side_coverage(dets)
    return {c: max(0, int(round((a[c] + b[c] + c_[c]) / 3.0))) for c in NAMES}


def hybrid_w(w: float):
    """Weighted blend visibility (w) + adaptive_corrected (1-w)."""
    def _fn(dets):
        v = base.visibility_count(dets)
        c = base.adaptive_corrected(dets)
        return {cl: max(0, int(round(w * v[cl] + (1 - w) * c[cl]))) for cl in NAMES}
    _fn.__name__ = f"hybrid_w{int(w * 100):02d}"
    return _fn


CANDIDATES = {
    "baseline_hybrid_vis_corr": base.hybrid_vis_corr,
    "baseline_visibility": base.visibility_count,
    "median_top5": median_top5,
    "trimmed_mean5": trimmed_mean5,
    "floor_clamped_hybrid": floor_clamped_hybrid,
    "floor_clamped_vis": floor_clamped_vis,
    "triple_avg": triple_avg,
}
for _w in (0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80):
    CANDIDATES[f"hybrid_w{int(_w * 100):02d}"] = hybrid_w(_w)


# ─── load + evaluate ──────────────────────────────────────────

def load_953_trees():
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
        trees[tree_id] = {"dets": dets, "gt": gt}
    return trees


def within1(pred, gt):
    return all(abs(pred.get(c, 0) - gt.get(c, 0)) <= 1 for c in NAMES)


def mae(pred, gt):
    return float(np.mean([abs(pred.get(c, 0) - gt.get(c, 0)) for c in NAMES]))


def main():
    base._load_v6_params()
    print(f"Loading trees from {JSON_DIR}...")
    trees = load_953_trees()
    n_trees = len(trees)
    print(f"Loaded {n_trees} trees.")

    rows = []
    for name, fn in CANDIDATES.items():
        ok, mae_list, errs = [], [], []
        for info in trees.values():
            pred = fn(info["dets"])
            ok.append(within1(pred, info["gt"]))
            mae_list.append(mae(pred, info["gt"]))
            errs.append(sum(abs(pred.get(c, 0) - info["gt"].get(c, 0)) for c in NAMES))
        rows.append({
            "method": name,
            "acc_within1_pct": round(100.0 * sum(ok) / n_trees, 2),
            "MAE": round(float(np.mean(mae_list)), 4),
            "mean_total_err": round(float(np.mean(errs)), 4),
            "n_fail": int(n_trees - sum(ok)),
        })

    df = pd.DataFrame(rows).sort_values("acc_within1_pct", ascending=False)
    out_csv = OUT_DIR / "iter1_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nResults written to: {out_csv}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
