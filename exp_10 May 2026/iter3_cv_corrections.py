"""
Iteration 3 — principled per-class corrections with held-out validation.

Findings from iter2:
- B4 systematically under-predicted in dense trees.
- B1, B2 mildly over-predicted.
- B3 bidirectional (irreducible — B2/B3 ambiguity).

Goal: derive per-class divisor adjustments that GENERALISE — not overfit
the 133 failure trees. Strategy: calibrate on `split=train` only, validate
on `split=val` and `split=test`. Report all three, plus full 953 mean.

Constraint (RULES.txt): every adjustment must be:
1. Principled (motivated by structural/statistical reasoning).
2. Tested on held-out data — gain on train and val/test, not just train.
3. Reported honestly — if it overfits, it goes in the report, not in the
   recommendation.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict

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


def floor_clamped_hybrid(dets):
    """Iter1 winner — used as the baseline to beat."""
    base_pred = base.hybrid_vis_corr(dets)
    return {c: max(base_pred[c], _max_per_side(dets, c)) for c in NAMES}


# ─── candidates ───────────────────────────────────────────────

def class_divisor_corrected(dets, divisors):
    """Like adaptive_corrected but with per-class divisor override."""
    n_total = len(dets)
    base_factors = {"B1": 1.986, "B2": 1.786, "B3": 1.795, "B4": 1.655}
    dup_rate = float(np.clip(2.05 - 0.014 * n_total, 1.45, 2.10))
    scale = dup_rate / 1.79
    n = base.naive_count(dets)
    return {
        c: max(0, round(n[c] / (divisors.get(c, base_factors[c]) * scale)))
        for c in NAMES
    }


def hybrid_with_divisors(divisors):
    """hybrid_vis_corr + custom per-class divisors + floor clamp."""
    def _fn(dets):
        v = base.visibility_count(dets)
        c_pred = class_divisor_corrected(dets, divisors)
        out = {cl: max(0, int(round(0.6 * v[cl] + 0.4 * c_pred[cl]))) for cl in NAMES}
        return {cl: max(out[cl], _max_per_side(dets, cl)) for cl in NAMES}
    return _fn


def adaptive_b4_lift(dets):
    """Principled B4 fix: when B4 is observed across multiple sides
    (active_sides_B4 >= 2), lift the floor to ceil(naive_B4 / 1.50)
    instead of letting adaptive_corrected divide by ~1.66.

    Reasoning: B4 (ripe bunch) is rare per tree — typical 0-2 unique.
    When it appears in K>=2 sides, duplication factor is structurally
    bounded below the all-class average (because there are fewer
    unique B4 to compete for cross-side overlap)."""
    base_pred = floor_clamped_hybrid(dets)
    out = dict(base_pred)
    n = base.naive_count(dets)
    if n["B4"] >= 3 and _active_sides(dets, "B4") >= 2:
        lifted = int(np.ceil(n["B4"] / 1.50))
        out["B4"] = max(out["B4"], min(lifted, n["B4"]))
    return out


def b1_concentration_trim(dets):
    """Principled B1 fix: if all B1 detections concentrated in <=1 side,
    cap pred at max_per_side(B1). Reasoning: a class that only appears
    on one side has zero cross-side duplication evidence, so unique
    count = max_per_side, not the inflated visibility-weighted count."""
    base_pred = floor_clamped_hybrid(dets)
    out = dict(base_pred)
    if _active_sides(dets, "B1") <= 1:
        out["B1"] = min(out["B1"], _max_per_side(dets, "B1"))
    return out


def combined_corrections(dets):
    """Stack adaptive_b4_lift + b1_concentration_trim. Both are
    structurally motivated, both should compose."""
    out = adaptive_b4_lift(dets)
    if _active_sides(dets, "B1") <= 1:
        out["B1"] = min(out["B1"], _max_per_side(dets, "B1"))
    if _active_sides(dets, "B2") <= 1:
        out["B2"] = min(out["B2"], _max_per_side(dets, "B2"))
    return out


# ─── load with split info ─────────────────────────────────────

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


def evaluate(fn: Callable, trees, split_filter=None):
    items = trees.values() if split_filter is None else [t for t in trees.values() if t["split"] == split_filter]
    n = len(items)
    if n == 0:
        return {"acc": 0.0, "mae": 0.0, "n_fail": 0, "n": 0}
    ok, mae_list, fails = 0, [], 0
    for info in items:
        pred = fn(info["dets"])
        if _within1(pred, info["gt"]):
            ok += 1
        else:
            fails += 1
        mae_list.append(_mae(pred, info["gt"]))
    return {
        "acc": round(100.0 * ok / n, 2),
        "mae": round(float(np.mean(mae_list)), 4),
        "n_fail": fails,
        "n": n,
    }


# ─── main ─────────────────────────────────────────────────────

def main():
    base._load_v6_params()
    trees = load_with_split()
    splits = defaultdict(int)
    for t in trees.values():
        splits[t["split"]] += 1
    print(f"Total: {len(trees)} trees")
    for s, n in splits.items():
        print(f"  split={s}: {n}")

    candidates = {
        "baseline_floor_clamped_hybrid": floor_clamped_hybrid,
        "iter1_baseline_hybrid_vis_corr": base.hybrid_vis_corr,
        "adaptive_b4_lift": adaptive_b4_lift,
        "b1_concentration_trim": b1_concentration_trim,
        "combined_corrections": combined_corrections,
    }

    # B4 divisor sweep — calibrate on train only, then validate
    for d in (1.35, 1.40, 1.45, 1.50, 1.55, 1.60):
        def make_fn(dv):
            def _fn(dets):
                base_pred = floor_clamped_hybrid(dets)
                out = dict(base_pred)
                n = base.naive_count(dets)
                if n["B4"] >= 3 and _active_sides(dets, "B4") >= 2:
                    lifted = int(np.ceil(n["B4"] / dv))
                    out["B4"] = max(out["B4"], min(lifted, n["B4"]))
                return out
            return _fn
        candidates[f"b4_lift_d{int(d*100):03d}"] = make_fn(d)

    rows = []
    for name, fn in candidates.items():
        all_ = evaluate(fn, trees, None)
        tr = evaluate(fn, trees, "train")
        va = evaluate(fn, trees, "val")
        te = evaluate(fn, trees, "test")
        rows.append({
            "method": name,
            "acc_all": all_["acc"], "mae_all": all_["mae"], "fail_all": all_["n_fail"],
            "acc_train": tr["acc"], "mae_train": tr["mae"], "fail_train": tr["n_fail"],
            "acc_val": va["acc"], "mae_val": va["mae"], "fail_val": va["n_fail"],
            "acc_test": te["acc"], "mae_test": te["mae"], "fail_test": te["n_fail"],
            "n_train": tr["n"], "n_val": va["n"], "n_test": te["n"],
        })

    df = pd.DataFrame(rows).sort_values("acc_all", ascending=False)
    out_csv = OUT_DIR / "iter3_cv_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nCSV: {out_csv}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
