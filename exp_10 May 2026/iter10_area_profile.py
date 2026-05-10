"""Profile trees uniquely recovered by area_clustered_tight."""

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


def area_clustered_tight(dets, area_tol=0.20, y_tol=0.15):
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
                ra, rb = find(i), find(j)
                if ra != rb:
                    parent[rb] = ra
        out[cl] = len({find(i) for i in range(n)})
    return out


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


def main():
    base._load_v6_params()
    trees = load_with_split()
    rows = []
    for tid, info in trees.items():
        dets = info["dets"]
        gt = info["gt"]
        sel = selector_iter9_trifurc(dets)
        area = area_clustered_tight(dets)
        gmb = geometric_mean_blend(dets)
        med = median3_floor(dets)
        vis = base.visibility_count(dets)
        adp = base.adaptive_corrected(dets)
        sid = base.side_coverage(dets)
        naive = base.naive_count(dets)
        n_total = len(dets)
        rows.append({
            "tree_id": tid, "split": info["split"],
            "n_dets": n_total, "n_sides": len(set(d["side_index"] for d in dets)),
            "naive_B1": naive["B1"], "naive_B2": naive["B2"],
            "naive_B3": naive["B3"], "naive_B4": naive["B4"],
            "active_B1": _active_sides(dets, "B1"),
            "active_B4": _active_sides(dets, "B4"),
            "max_B3": _max_per_side(dets, "B3"),
            "ratio_B3": naive["B3"] / max(n_total, 1),
            "pass_sel": int(_within1(sel, gt)),
            "pass_area": int(_within1(area, gt)),
            "pass_gmb": int(_within1(gmb, gt)),
            "pass_med": int(_within1(med, gt)),
            "pass_vis": int(_within1(vis, gt)),
            "pass_adp": int(_within1(adp, gt)),
            "pass_sid": int(_within1(sid, gt)),
        })
    df = pd.DataFrame(rows)

    # Trees uniquely recoverable by area_clustered_tight
    sel_fail = df[df["pass_sel"] == 0]
    area_only = sel_fail[
        (sel_fail["pass_area"] == 1)
        & (sel_fail["pass_gmb"] == 0)
        & (sel_fail["pass_med"] == 0)
        & (sel_fail["pass_vis"] == 0)
        & (sel_fail["pass_adp"] == 0)
        & (sel_fail["pass_sid"] == 0)
    ]
    print(f"Trees uniquely recoverable by area_clustered_tight: {len(area_only)}")
    print(area_only[["tree_id", "split", "n_dets", "naive_B1", "naive_B2", "naive_B3", "naive_B4", "ratio_B3"]].to_string(index=False))

    # All trees where area is correct AND selector wrong
    print(f"\n\nAll trees where selector_iter9 wrong AND area_clustered_tight right: {(sel_fail['pass_area']==1).sum()}")
    sel_fail_area_ok = sel_fail[sel_fail["pass_area"] == 1]
    print(sel_fail_area_ok[["tree_id", "split", "n_dets", "naive_B1", "naive_B2", "naive_B3", "naive_B4", "ratio_B3", "pass_gmb", "pass_med", "pass_vis", "pass_adp"]].to_string(index=False))

    # Counter-examples: where area_tight is wrong (so we can't always route)
    n_area_correct = int((df["pass_area"] == 1).sum())
    n_area_wrong = int((df["pass_area"] == 0).sum())
    print(f"\narea_clustered_tight overall: pass {n_area_correct}, fail {n_area_wrong}")


if __name__ == "__main__":
    main()
