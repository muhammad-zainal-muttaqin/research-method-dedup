"""
Iteration 6 — disagreement analysis across top methods.

Question: do all methods fail the same trees (structural ceiling) or
different trees (ensemble disagreement opportunity)?

Steps:
1. Run 7 top methods on 953 trees.
2. For each tree, log which methods pass.
3. Compute oracle ceiling: always pick a passing method per tree.
4. Compute coverage of unions and intersections.
5. Identify "ensemble-recoverable" trees: failing in winner but passing
   in at least one peer.

Output: exp_10 May 2026/iter6_disagreement.csv + iter6_report.md
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


def floor_clamped_hybrid(dets):
    p = base.hybrid_vis_corr(dets)
    return {c: max(p[c], _max_per_side(dets, c)) for c in NAMES}


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


def main():
    base._load_v6_params()
    trees = load_with_split()
    n_total = len(trees)
    print(f"Total: {n_total}")

    methods = {
        "geometric_mean_blend": geometric_mean_blend,
        "floor_clamped_hybrid": floor_clamped_hybrid,
        "visibility": base.visibility_count,
        "side_coverage": base.side_coverage,
        "adaptive_corrected": base.adaptive_corrected,
        "median3_floor": median3_floor,
        "density_scaled_vis": base.density_scaled_vis,
    }

    # Per-tree pass matrix
    rows = []
    pass_matrix = {}  # tid -> {method: bool}
    pred_matrix = {}  # tid -> {method: pred}
    for tid, info in trees.items():
        gt = info["gt"]
        passes = {}
        preds = {}
        for name, fn in methods.items():
            pred = fn(info["dets"])
            passes[name] = _within1(pred, gt)
            preds[name] = pred
        pass_matrix[tid] = passes
        pred_matrix[tid] = preds
        rows.append({
            "tree_id": tid,
            "split": info["split"],
            "n_dets": len(info["dets"]),
            **{f"pass_{m}": int(p) for m, p in passes.items()},
            "n_methods_pass": sum(passes.values()),
            "any_pass": int(any(passes.values())),
            "all_pass": int(all(passes.values())),
        })
    df = pd.DataFrame(rows)

    # ── per-method totals ─────────────────────────────────────
    print("\n=== Per-method Acc+-1 (sanity) ===")
    for m in methods:
        n_pass = int(df[f"pass_{m}"].sum())
        print(f"  {m:25s}: {100*n_pass/n_total:.2f}% ({n_pass}/{n_total})")

    # ── disagreement breakdown ────────────────────────────────
    print("\n=== Disagreement breakdown ===")
    print(f"  Trees where ALL methods pass:  {int(df['all_pass'].sum())} ({100*df['all_pass'].mean():.2f}%)")
    print(f"  Trees where ANY method passes: {int(df['any_pass'].sum())} ({100*df['any_pass'].mean():.2f}%)  -- ORACLE CEILING")
    print(f"  Trees where NO method passes:  {int((1-df['any_pass']).sum())} ({100*(1-df['any_pass']).mean():.2f}%)  -- STRUCTURAL HARD")

    n_total_pass_dist = df["n_methods_pass"].value_counts().sort_index()
    print("\nDistribution of n_methods_pass:")
    print(n_total_pass_dist.to_string())

    # ── ensemble-recoverable: winner fails but at least one peer passes ──
    winner = "geometric_mean_blend"
    losers = df[df[f"pass_{winner}"] == 0]
    recoverable = losers[losers["any_pass"] == 1]
    structural = losers[losers["any_pass"] == 0]
    print(f"\n=== Winner = {winner} ===")
    print(f"  Failures total: {len(losers)}")
    print(f"  Recoverable (some peer passes): {len(recoverable)}")
    print(f"  Structural hard (all fail): {len(structural)}")

    # For recoverable: which peer methods pass them?
    print("\nPeer methods that recover winner failures:")
    for m in methods:
        if m == winner:
            continue
        n_recover = int(((recoverable[f"pass_{m}"] == 1)).sum())
        print(f"  {m:25s}: {n_recover}")

    # Save
    df.to_csv(OUT_DIR / "iter6_disagreement.csv", index=False)
    print(f"\nCSV: iter6_disagreement.csv")

    # Also save the recoverable subset for iter7
    recoverable.to_csv(OUT_DIR / "iter6_recoverable.csv", index=False)
    print(f"CSV: iter6_recoverable.csv ({len(recoverable)} trees)")


if __name__ == "__main__":
    main()
