"""
Iteration 13 — per-class MAE breakdown of current best.

Identify which class contributes most to MAE so we can target
class-specific rounding/correction.
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


def selector_with_b2b3(dets):
    n_total = len(dets)
    if n_total == 0:
        return geometric_mean_blend(dets)
    naive = base.naive_count(dets)
    b3frac = naive["B3"] / n_total
    if b3frac >= 0.60 and n_total >= 25:
        pred = median3_floor(dets)
    elif naive["B1"] >= 3 and b3frac < 0.45 and naive["B4"] < 10:
        pred = base.adaptive_corrected(dets)
    else:
        pred = geometric_mean_blend(dets)
    # b2b3 split
    joint = pred["B2"] + pred["B3"]
    if joint == 0:
        return {cl: max(pred[cl], _max_per_side(dets, cl)) for cl in NAMES}
    b23 = [d for d in dets if d["class"] in ("B2", "B3")]
    if not b23:
        return {cl: max(pred[cl], _max_per_side(dets, cl)) for cl in NAMES}
    n_b3 = sum(1 for d in b23 if d["class"] == "B3")
    n_b2 = sum(1 for d in b23 if d["class"] == "B2")
    if n_b2 + n_b3 == 0:
        return {cl: max(pred[cl], _max_per_side(dets, cl)) for cl in NAMES}
    frac_b3 = n_b3 / (n_b2 + n_b3)
    new_b3 = int(round(joint * frac_b3))
    new_b2 = joint - new_b3
    out = dict(pred)
    out["B2"] = new_b2
    out["B3"] = new_b3
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


def main():
    base._load_v6_params()
    trees = load_with_split()
    rows = []
    for tid, info in trees.items():
        pred = selector_with_b2b3(info["dets"])
        gt = info["gt"]
        rows.append({
            "tree_id": tid, "split": info["split"],
            **{f"err_{c}": pred[c] - gt[c] for c in NAMES},
            **{f"abs_{c}": abs(pred[c] - gt[c]) for c in NAMES},
        })
    df = pd.DataFrame(rows)

    print("=== Per-class MAE ===")
    for c in NAMES:
        m = df[f"abs_{c}"].mean()
        m_err = df[f"err_{c}"].mean()
        print(f"  {c}: |err| mean = {m:.4f}  signed mean = {m_err:+.4f}")

    print("\n=== Total MAE (current) ===")
    total_mae = df[[f"abs_{c}" for c in NAMES]].mean(axis=1).mean()
    print(f"  {total_mae:.4f}")

    print("\n=== Distribution of |err| per class ===")
    for c in NAMES:
        dist = df[f"abs_{c}"].value_counts().sort_index()
        print(f"\n  {c}:")
        print(dist.to_string())

    # MAE by signed direction per class
    print("\n=== Over-pred vs under-pred per class ===")
    for c in NAMES:
        n_over = int((df[f"err_{c}"] > 0).sum())
        n_under = int((df[f"err_{c}"] < 0).sum())
        n_zero = int((df[f"err_{c}"] == 0).sum())
        print(f"  {c}: over={n_over} ({100*n_over/len(df):.1f}%)  under={n_under} ({100*n_under/len(df):.1f}%)  zero={n_zero}")


if __name__ == "__main__":
    main()
