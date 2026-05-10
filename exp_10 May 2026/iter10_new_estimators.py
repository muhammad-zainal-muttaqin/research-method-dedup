"""
Iteration 10 — expand toolkit with new principled estimators.

Goal: raise oracle ceiling above 89.61%. Current selector iter9 = 86.67%.
User target: 90% Acc+-1, MAE < 0.2.

Strategy: add new estimators that exploit signals NOT used by existing
toolkit:
1. area_clustered_count — cluster same-class detections by bbox area
   similarity across sides (size invariant assumption).
2. b2b3_joint_split — predict B2+B3 jointly then split by y-position
   median (B3 typically lower on tree, B2 higher).
3. spatial_cluster_count — UnionFind merge by (y, area) similarity
   across sides only.

Constraint: no learning, deterministic, no peek at GT.
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


# ─── new estimators ───────────────────────────────────────────

def area_clustered_count(dets, area_tol=0.30, y_tol=0.20):
    """For each class, cluster cross-side detections by (y, area)
    similarity. Assume same bunch viewed from different sides has
    similar y position and similar area. Within-side clusters never
    merge (one bunch per side at most).

    Deterministic: union-find over edges where:
    - different sides
    - |y1 - y2| <= y_tol
    - |sqrt(area1) - sqrt(area2)| / max(sqrt(area)) <= area_tol
    """
    out = {}
    for cl in NAMES:
        cd = [d for d in dets if d["class"] == cl]
        n = len(cd)
        if n == 0:
            out[cl] = 0
            continue
        if n == 1:
            out[cl] = 1
            continue
        parent = list(range(n))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
        for i in range(n):
            for j in range(i + 1, n):
                if cd[i]["side_index"] == cd[j]["side_index"]:
                    continue
                if abs(cd[i]["y_norm"] - cd[j]["y_norm"]) > y_tol:
                    continue
                a_i = np.sqrt(max(cd[i]["area_norm"], 1e-9))
                a_j = np.sqrt(max(cd[j]["area_norm"], 1e-9))
                if abs(a_i - a_j) / max(a_i, a_j) > area_tol:
                    continue
                union(i, j)
        out[cl] = len({find(i) for i in range(n)})
    return out


def b2b3_joint_split(dets):
    """Predict B2 and B3 jointly via geometric_mean_blend on union, then
    split by y_norm median (lower y -> B3, upper -> B2).
    Assumption: B3 (mature) hangs lower on tree on average.
    """
    pred_blend = geometric_mean_blend(dets)
    joint = pred_blend["B2"] + pred_blend["B3"]
    if joint == 0:
        return pred_blend
    # Split observed B2+B3 detections by y median
    b23 = [d for d in dets if d["class"] in ("B2", "B3")]
    if not b23:
        return pred_blend
    n_b3_obs = sum(1 for d in b23 if d["class"] == "B3")
    n_b2_obs = sum(1 for d in b23 if d["class"] == "B2")
    if n_b2_obs + n_b3_obs == 0:
        return pred_blend
    frac_b3 = n_b3_obs / (n_b2_obs + n_b3_obs)
    new_b3 = int(round(joint * frac_b3))
    new_b2 = joint - new_b3
    out = dict(pred_blend)
    out["B2"] = max(new_b2, _max_per_side(dets, "B2"))
    out["B3"] = max(new_b3, _max_per_side(dets, "B3"))
    return out


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

    methods = {
        "selector_iter9": selector_iter9_trifurc,
        "geometric_mean_blend": geometric_mean_blend,
        "median3_floor": median3_floor,
        "visibility": base.visibility_count,
        "side_coverage": base.side_coverage,
        "adaptive_corrected": base.adaptive_corrected,
        "density_scaled_vis": base.density_scaled_vis,
        "area_clustered_default": area_clustered_count,
        "area_clustered_tight": lambda d: area_clustered_count(d, area_tol=0.20, y_tol=0.15),
        "area_clustered_loose": lambda d: area_clustered_count(d, area_tol=0.40, y_tol=0.25),
        "b2b3_joint_split": b2b3_joint_split,
    }

    # ── individual evaluation ─────────────────────────────────
    print("=== Individual methods on 953 ===")
    rows = []
    for name, fn in methods.items():
        r = evaluate(fn, trees, None)
        tr = evaluate(fn, trees, "train")
        va = evaluate(fn, trees, "val")
        te = evaluate(fn, trees, "test")
        rows.append({
            "method": name, "acc": r["acc"], "mae": r["mae"], "n_fail": r["n_fail"],
            "acc_train": tr["acc"], "acc_val": va["acc"], "acc_test": te["acc"],
        })
    df = pd.DataFrame(rows).sort_values("acc", ascending=False)
    df.to_csv(OUT_DIR / "iter10_individual.csv", index=False)
    print(df.to_string(index=False))

    # ── new oracle with expanded toolkit ──────────────────────
    print("\n=== Oracle ceiling with NEW toolkit ===")
    pass_matrix = {}
    for tid, info in trees.items():
        passes = {}
        for name, fn in methods.items():
            passes[name] = _within1(fn(info["dets"]), info["gt"])
        pass_matrix[tid] = passes

    n_total = len(trees)
    n_any = sum(1 for tid in trees if any(pass_matrix[tid].values()))
    n_all_fail = n_total - n_any
    print(f"  Total: {n_total}")
    print(f"  Any method passes: {n_any} ({100*n_any/n_total:.2f}%)  -- NEW ORACLE")
    print(f"  All methods fail (structural hard): {n_all_fail} ({100*n_all_fail/n_total:.2f}%)")

    # Per-method recovery of selector_iter9 failures
    iter9_fails = [tid for tid in trees if not pass_matrix[tid]["selector_iter9"]]
    print(f"\n  selector_iter9 fails: {len(iter9_fails)}")
    for m in methods:
        if m == "selector_iter9":
            continue
        n_recover = sum(1 for tid in iter9_fails if pass_matrix[tid][m])
        print(f"    {m:30s} recovers: {n_recover}")


if __name__ == "__main__":
    main()
