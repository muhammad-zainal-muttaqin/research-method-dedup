"""
Iteration 8 — multi-route selector + remaining-recoverable profile.

Steps:
1. Run iter7 selector + 6 other methods on 953 trees.
2. Find trees iter7 misses BUT a peer passes — these are residual recoverables.
3. Profile features of residual recoverables (TRAIN ONLY) to find a 2nd
   specialist route.
4. Compose multi-route selector and validate held-out.

Constraint: each rule must be 1-2 thresholds, profile-aligned, and pass
multi-split gate (worst_drop >= -0.3pp, all-improvement positive).
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


def selector_iter7(dets):
    n_total = len(dets)
    naive = base.naive_count(dets)
    if n_total > 0 and naive["B1"] >= 3 and (naive["B3"] / n_total) < 0.45:
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


def features(dets):
    n_total = len(dets)
    naive = base.naive_count(dets)
    return {
        "n_dets": n_total,
        "naive_total": sum(naive.values()),
        "naive_B1": naive["B1"],
        "naive_B2": naive["B2"],
        "naive_B3": naive["B3"],
        "naive_B4": naive["B4"],
        "active_B1": _active_sides(dets, "B1"),
        "active_B2": _active_sides(dets, "B2"),
        "active_B3": _active_sides(dets, "B3"),
        "active_B4": _active_sides(dets, "B4"),
        "max_B3": _max_per_side(dets, "B3"),
        "ratio_B3_total": naive["B3"] / max(n_total, 1),
        "ratio_B4_total": naive["B4"] / max(n_total, 1),
        "ratio_B1_total": naive["B1"] / max(n_total, 1),
    }


def main():
    base._load_v6_params()
    trees = load_with_split()

    methods = {
        "selector_iter7": selector_iter7,
        "geometric_mean_blend": geometric_mean_blend,
        "visibility": base.visibility_count,
        "side_coverage": base.side_coverage,
        "median3_floor": median3_floor,
        "density_scaled_vis": base.density_scaled_vis,
    }

    rows = []
    for tid, info in trees.items():
        dets = info["dets"]
        gt = info["gt"]
        feat = features(dets)
        passes = {m: _within1(fn(dets), gt) for m, fn in methods.items()}
        rows.append({
            "tree_id": tid,
            "split": info["split"],
            **feat,
            **{f"pass_{m}": int(p) for m, p in passes.items()},
        })
    df = pd.DataFrame(rows)

    # Identify residual recoverables = iter7 fails but some peer passes
    iter7_fail = df[df["pass_selector_iter7"] == 0].copy()
    peer_cols = [f"pass_{m}" for m in methods if m != "selector_iter7"]
    iter7_fail["any_peer_pass"] = (iter7_fail[peer_cols].sum(axis=1) > 0).astype(int)
    residual = iter7_fail[iter7_fail["any_peer_pass"] == 1].copy()

    print(f"=== iter7 fails: {len(iter7_fail)} ===")
    print(f"  any peer passes: {len(residual)}")
    print(f"  no peer passes (structural hard): {(iter7_fail['any_peer_pass'] == 0).sum()}")

    # Which peer recovers most? (per tree it counts as 1 recovery)
    # but key: which peer to route to per residual tree?
    print("\nPeer recovery distribution among residual trees:")
    for m in methods:
        if m == "selector_iter7":
            continue
        n = int(residual[f"pass_{m}"].sum())
        print(f"  {m:25s}: {n}")

    # Feature profile: residual on TRAIN only
    res_train = residual[residual["split"] == "train"]
    feat_cols = [c for c in df.columns if c not in {"tree_id", "split"} and not c.startswith("pass_") and c != "any_peer_pass"]
    print(f"\n=== Feature medians (TRAIN residual, n={len(res_train)}) ===")
    print(res_train[feat_cols].median().round(2).to_string())

    # Compare to "pass_iter7" trees on train (those iter7 already gets right)
    iter7_pass_train = df[(df["split"] == "train") & (df["pass_selector_iter7"] == 1)]
    print(f"\n=== Feature medians (TRAIN iter7-pass, n={len(iter7_pass_train)}) ===")
    print(iter7_pass_train[feat_cols].median().round(2).to_string())

    # Compare to "fails iter7 AND no peer recovers" (structural hard)
    hard_train = df[(df["split"] == "train") & (df["pass_selector_iter7"] == 0)].merge(
        iter7_fail[iter7_fail["any_peer_pass"] == 0][["tree_id"]], on="tree_id", how="inner"
    )
    print(f"\n=== Feature medians (TRAIN structural hard, n={len(hard_train)}) ===")
    print(hard_train[feat_cols].median().round(2).to_string())

    # Save residual for next step
    residual.to_csv(OUT_DIR / "iter8_residual.csv", index=False)
    df.to_csv(OUT_DIR / "iter8_perfreestructure.csv", index=False)


if __name__ == "__main__":
    main()
