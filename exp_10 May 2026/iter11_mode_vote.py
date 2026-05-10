"""
Iteration 11 — mode-vote ensembles + class-specific routing.

Candidates:
1. mode5 — per class, mode (most common) prediction across 5 estimators
2. median5 — per class, median prediction across 5 estimators
3. trim5 — per class, trim min+max then mean of remaining 3
4. class_specialist — pick best per class independently (using train-only
   leave-one-out style)
5. area_floor_b3 — selector_iter9 with area_clustered as floor for B3 only
6. b2b3_split_med — b2b3_joint_split using median3 as base instead of geo
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


def geometric_mean_blend(dets):
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
    a = base.visibility_count(dets)
    b = base.adaptive_corrected(dets)
    s = base.side_coverage(dets)
    out = {cl: sorted([a[cl], b[cl], s[cl]])[1] for cl in NAMES}
    return {cl: max(out[cl], _max_per_side(dets, cl)) for cl in NAMES}


def selector_iter9_trifurc(dets):
    n_total = len(dets)
    if n_total == 0:
        return geometric_mean_blend(dets)
    naive = base.naive_count(dets)
    b3frac = naive["B3"] / n_total
    if b3frac >= 0.60 and n_total >= 25:
        return median3_floor(dets)
    if naive["B1"] >= 3 and b3frac < 0.45 and naive["B4"] < 10:
        return base.adaptive_corrected(dets)
    return geometric_mean_blend(dets)


def b2b3_joint_split(dets, base_fn=geometric_mean_blend):
    pred = base_fn(dets)
    joint = pred["B2"] + pred["B3"]
    if joint == 0:
        return pred
    b23 = [d for d in dets if d["class"] in ("B2", "B3")]
    if not b23:
        return pred
    n_b3 = sum(1 for d in b23 if d["class"] == "B3")
    n_b2 = sum(1 for d in b23 if d["class"] == "B2")
    if n_b2 + n_b3 == 0:
        return pred
    frac_b3 = n_b3 / (n_b2 + n_b3)
    new_b3 = int(round(joint * frac_b3))
    new_b2 = joint - new_b3
    out = dict(pred)
    out["B2"] = max(new_b2, _max_per_side(dets, "B2"))
    out["B3"] = max(new_b3, _max_per_side(dets, "B3"))
    return out


def _estimators(dets):
    return {
        "vis": base.visibility_count(dets),
        "sid": base.side_coverage(dets),
        "med": median3_floor(dets),
        "geo": geometric_mean_blend(dets),
        "adp": base.adaptive_corrected(dets),
    }


def mode5(dets):
    """Per-class mode (most common); ties broken by lowest value (conservative)."""
    est = _estimators(dets)
    out = {}
    for cl in NAMES:
        vals = [e[cl] for e in est.values()]
        cnt = Counter(vals).most_common()
        max_freq = cnt[0][1]
        candidates = sorted(v for v, c in cnt if c == max_freq)
        out[cl] = candidates[0]
    return {cl: max(out[cl], _max_per_side(dets, cl)) for cl in NAMES}


def median5(dets):
    """Per-class median across 5 estimators."""
    est = _estimators(dets)
    out = {}
    for cl in NAMES:
        vals = sorted(e[cl] for e in est.values())
        out[cl] = vals[len(vals) // 2]
    return {cl: max(out[cl], _max_per_side(dets, cl)) for cl in NAMES}


def trim5(dets):
    """Per-class trimmed mean (drop min+max, mean of middle 3)."""
    est = _estimators(dets)
    out = {}
    for cl in NAMES:
        vals = sorted(e[cl] for e in est.values())
        trimmed = vals[1:-1]
        out[cl] = max(0, int(round(sum(trimmed) / len(trimmed))))
    return {cl: max(out[cl], _max_per_side(dets, cl)) for cl in NAMES}


def b2b3_med_split(dets):
    return b2b3_joint_split(dets, base_fn=median3_floor)


def b2b3_iter9_split(dets):
    return b2b3_joint_split(dets, base_fn=selector_iter9_trifurc)


def selector_with_b2b3(dets):
    """selector_iter9 + b2b3 split correction always applied."""
    pred = selector_iter9_trifurc(dets)
    joint = pred["B2"] + pred["B3"]
    if joint == 0:
        return pred
    b23 = [d for d in dets if d["class"] in ("B2", "B3")]
    if not b23:
        return pred
    n_b3 = sum(1 for d in b23 if d["class"] == "B3")
    n_b2 = sum(1 for d in b23 if d["class"] == "B2")
    if n_b2 + n_b3 == 0:
        return pred
    frac_b3 = n_b3 / (n_b2 + n_b3)
    new_b3 = int(round(joint * frac_b3))
    new_b2 = joint - new_b3
    out = dict(pred)
    out["B2"] = max(new_b2, _max_per_side(dets, "B2"))
    out["B3"] = max(new_b3, _max_per_side(dets, "B3"))
    return out


def class_specialist(dets):
    """Hardcoded per-class best estimator (train-derived):
    B1: adaptive_corrected (winner specialist)
    B2: median3
    B3: geometric_mean_blend
    B4: visibility
    """
    out = {
        "B1": base.adaptive_corrected(dets)["B1"],
        "B2": median3_floor(dets)["B2"],
        "B3": geometric_mean_blend(dets)["B3"],
        "B4": base.visibility_count(dets)["B4"],
    }
    return {cl: max(out[cl], _max_per_side(dets, cl)) for cl in NAMES}


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


def _within1(p, g):
    return all(abs(p[c] - g[c]) <= 1 for c in NAMES)


def _mae(p, g):
    return float(np.mean([abs(p[c] - g[c]) for c in NAMES]))


def evaluate(fn, trees, split=None):
    items = list(trees.values()) if split is None else [t for t in trees.values() if t["split"] == split]
    n = len(items)
    if n == 0:
        return {"acc": 0.0, "mae": 0.0, "n_fail": 0}
    ok, maes = 0, []
    for info in items:
        pred = fn(info["dets"])
        ok += int(_within1(pred, info["gt"]))
        maes.append(_mae(pred, info["gt"]))
    return {"acc": round(100.0 * ok / n, 2), "mae": round(float(np.mean(maes)), 4), "n_fail": n - ok}


def main():
    base._load_v6_params()
    trees = load_with_split()

    candidates = {
        "iter9_baseline": selector_iter9_trifurc,
        "mode5": mode5,
        "median5": median5,
        "trim5": trim5,
        "b2b3_med_split": b2b3_med_split,
        "b2b3_iter9_split": b2b3_iter9_split,
        "selector_with_b2b3": selector_with_b2b3,
        "class_specialist": class_specialist,
    }
    rows = []
    bl = {"all": 86.67, "train": 87.34, "val": 82.58, "test": 88.62}
    for name, fn in candidates.items():
        r = evaluate(fn, trees, None)
        tr = evaluate(fn, trees, "train")
        va = evaluate(fn, trees, "val")
        te = evaluate(fn, trees, "test")
        rows.append({
            "method": name, "acc_all": r["acc"], "mae_all": r["mae"],
            "acc_train": tr["acc"], "acc_val": va["acc"], "acc_test": te["acc"],
            "d_train": round(tr["acc"] - bl["train"], 2),
            "d_val": round(va["acc"] - bl["val"], 2),
            "d_test": round(te["acc"] - bl["test"], 2),
        })
    df = pd.DataFrame(rows)
    df["worst_drop"] = df[["d_train", "d_val", "d_test"]].min(axis=1)
    df = df.sort_values(["acc_all", "mae_all"], ascending=[False, True])
    df.to_csv(OUT_DIR / "iter11_results.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
