"""
Iteration 12 — total-first prediction + class redistribution.

Hypothesis: predicting TOTAL bunches first (robust signal) then splitting
by observation ratio reduces per-class noise.

Variants:
1. total_then_split — predict total via geo_blend total, split by naive ratio
2. total_then_split_iter9 — selector_iter9 total, split by ratio
3. weighted_split — weights from train-only class prior
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


def selector_with_b2b3(dets):
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


def total_then_split(dets, base_fn=geometric_mean_blend):
    """Predict total via base_fn, split by naive class ratio.
    Floor each class by max_per_side."""
    pred_base = base_fn(dets)
    total = sum(pred_base.values())
    naive = base.naive_count(dets)
    naive_total = sum(naive.values())
    if naive_total == 0:
        return pred_base
    out = {}
    remaining = total
    # Distribute by naive ratio, largest first
    for cl in sorted(NAMES, key=lambda c: -naive[c]):
        if cl == NAMES[-1]:
            # last: take residual
            out[cl] = max(0, remaining)
        else:
            share = int(round(total * naive[cl] / naive_total))
            out[cl] = share
            remaining -= share
    # Floor by max_per_side
    return {cl: max(out[cl], _max_per_side(dets, cl)) for cl in NAMES}


def total_then_split_iter9(dets):
    return total_then_split(dets, base_fn=selector_iter9_trifurc)


def total_then_split_iter9_b2b3(dets):
    return total_then_split(dets, base_fn=selector_with_b2b3)


def total_then_split_med(dets):
    return total_then_split(dets, base_fn=median3_floor)


def avg_split_keep_acc(dets):
    """Hybrid: keep Acc-winning prediction (selector_with_b2b3) but smooth
    per-class via averaging with total_then_split."""
    a = selector_with_b2b3(dets)
    b = total_then_split(dets, base_fn=selector_with_b2b3)
    out = {cl: max(0, int(round((a[cl] + b[cl]) / 2.0))) for cl in NAMES}
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
        "iter11_baseline": selector_with_b2b3,
        "iter9_trifurc": selector_iter9_trifurc,
        "total_then_split_geo": total_then_split,
        "total_then_split_iter9": total_then_split_iter9,
        "total_then_split_iter9_b2b3": total_then_split_iter9_b2b3,
        "total_then_split_med": total_then_split_med,
        "avg_split_keep_acc": avg_split_keep_acc,
    }
    rows = []
    for name, fn in candidates.items():
        r = evaluate(fn, trees, None)
        tr = evaluate(fn, trees, "train")
        va = evaluate(fn, trees, "val")
        te = evaluate(fn, trees, "test")
        rows.append({
            "method": name, "acc_all": r["acc"], "mae_all": r["mae"],
            "acc_train": tr["acc"], "acc_val": va["acc"], "acc_test": te["acc"],
        })
    df = pd.DataFrame(rows).sort_values(["acc_all", "mae_all"], ascending=[False, True])
    df.to_csv(OUT_DIR / "iter12_results.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
