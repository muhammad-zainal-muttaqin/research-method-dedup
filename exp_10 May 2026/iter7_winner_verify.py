"""Verify the iter7 winner selector with MAE + per-split MAE + rule coverage."""

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


def selector_iter7(dets):
    """Iter7 winner: switch to adaptive_corrected when naive_B1 >= 3
    AND ratio_B3_total < 0.45. Profile-aligned per train-only feature
    means (adapt_only B1=3.25, B3frac=0.44 vs geo_only B1=2.33, B3frac=0.57)."""
    n_total = len(dets)
    naive = base.naive_count(dets)
    ratio_b3 = naive["B3"] / max(n_total, 1)
    if naive["B1"] >= 3 and ratio_b3 < 0.45:
        return base.adaptive_corrected(dets)
    return geometric_mean_blend(dets)


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


def main():
    base._load_v6_params()
    trees = load_with_split()
    rows = []
    for tid, info in trees.items():
        dets = info["dets"]
        gt = info["gt"]
        naive = base.naive_count(dets)
        rule_fires = (naive["B1"] >= 3 and naive["B3"] / max(len(dets), 1) < 0.45)
        pred = selector_iter7(dets)
        rows.append({
            "tree_id": tid, "split": info["split"],
            "rule_fires": int(rule_fires),
            "ok": int(_within1(pred, gt)),
            "mae": _mae(pred, gt),
        })
    df = pd.DataFrame(rows)

    print("=== Selector iter7 — verification ===")
    print(f"\nTotal trees: {len(df)}")
    print(f"Rule fires (use adaptive_corrected): {df['rule_fires'].sum()} ({100*df['rule_fires'].mean():.2f}%)")

    print("\nPer-split Acc+-1 and MAE:")
    for sp in ["train", "val", "test", None]:
        sp_df = df if sp is None else df[df["split"] == sp]
        n = len(sp_df)
        if n == 0:
            continue
        acc = 100.0 * sp_df["ok"].sum() / n
        mae = sp_df["mae"].mean()
        n_fires = sp_df["rule_fires"].sum()
        sp_label = sp if sp else "ALL"
        print(f"  {sp_label:6s}: n={n:4d}  Acc+-1={acc:.2f}%  MAE={mae:.4f}  rule_fires={n_fires}")

    # Compare baseline geo on same split
    print("\nFor reference, geometric_mean_blend baseline:")
    bl_rows = []
    for tid, info in trees.items():
        pred = geometric_mean_blend(info["dets"])
        bl_rows.append({
            "split": info["split"],
            "ok": int(_within1(pred, info["gt"])),
            "mae": _mae(pred, info["gt"]),
        })
    bl = pd.DataFrame(bl_rows)
    for sp in ["train", "val", "test", None]:
        sp_df = bl if sp is None else bl[bl["split"] == sp]
        n = len(sp_df)
        acc = 100.0 * sp_df["ok"].sum() / n
        mae = sp_df["mae"].mean()
        sp_label = sp if sp else "ALL"
        print(f"  {sp_label:6s}: Acc+-1={acc:.2f}%  MAE={mae:.4f}")

    df.to_csv(OUT_DIR / "iter7_verify.csv", index=False)


if __name__ == "__main__":
    main()
