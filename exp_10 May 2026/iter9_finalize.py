"""
Iteration 9 — finalisation: residual probe + production benchmark.

Steps:
1. Run iter8 winner selector vs all baselines on 953 trees.
2. Investigate the 4 trees iter8 still misses (where peer methods pass).
3. Test if a third route (visibility/median3) can be added safely.
4. Produce final consolidated benchmark CSV.
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


def selector_iter8(dets):
    n_total = len(dets)
    if n_total == 0:
        return geometric_mean_blend(dets)
    naive = base.naive_count(dets)
    b3frac = naive["B3"] / n_total
    if naive["B1"] >= 3 and b3frac < 0.45 and naive["B4"] < 10:
        return base.adaptive_corrected(dets)
    return geometric_mean_blend(dets)


def selector_iter9_trifurc(dets):
    """iter8 + third route to median3_floor when neither geo nor adaptive
    fits. Tested only — not a final commit."""
    n_total = len(dets)
    if n_total == 0:
        return geometric_mean_blend(dets)
    naive = base.naive_count(dets)
    b3frac = naive["B3"] / n_total
    # Third route: when extremely B3-heavy (residual hard zone)
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
        return {"acc": 0.0, "mae": 0.0, "n": 0, "n_fail": 0}
    ok, maes = 0, []
    for info in items:
        pred = fn(info["dets"])
        ok += int(_within1(pred, info["gt"]))
        maes.append(_mae(pred, info["gt"]))
    return {
        "acc": round(100.0 * ok / n, 2),
        "mae": round(float(np.mean(maes)), 4),
        "n": n,
        "n_fail": n - ok,
    }


def main():
    base._load_v6_params()
    trees = load_with_split()

    # ── 1. residual probe of iter8 ────────────────────────────
    iter8_fail_train_no_peer = []
    iter8_fail_with_peer = []
    for tid, info in trees.items():
        dets = info["dets"]
        gt = info["gt"]
        if not _within1(selector_iter8(dets), gt):
            peers_ok = (
                _within1(median3_floor(dets), gt)
                or _within1(base.visibility_count(dets), gt)
                or _within1(base.side_coverage(dets), gt)
            )
            entry = {"tree_id": tid, "split": info["split"], "n_dets": len(dets)}
            if peers_ok:
                iter8_fail_with_peer.append(entry)
            else:
                iter8_fail_train_no_peer.append(entry)
    print(f"=== iter8 fails: total={len(iter8_fail_with_peer) + len(iter8_fail_train_no_peer)}")
    print(f"  with peer pass (recoverable): {len(iter8_fail_with_peer)}")
    print(f"  no peer pass (structural hard): {len(iter8_fail_train_no_peer)}")

    # ── 2. final benchmark ────────────────────────────────────
    final_methods = {
        "selector_iter8 (PRODUCTION)": selector_iter8,
        "selector_iter9_trifurc (TEST)": selector_iter9_trifurc,
        "geometric_mean_blend": geometric_mean_blend,
        "hybrid_vis_corr (orig baseline)": base.hybrid_vis_corr,
        "visibility": base.visibility_count,
        "side_coverage": base.side_coverage,
        "median3_floor": median3_floor,
        "adaptive_corrected": base.adaptive_corrected,
    }
    rows = []
    for name, fn in final_methods.items():
        all_ = evaluate(fn, trees, None)
        tr = evaluate(fn, trees, "train")
        va = evaluate(fn, trees, "val")
        te = evaluate(fn, trees, "test")
        rows.append({
            "method": name,
            "acc_all": all_["acc"], "mae_all": all_["mae"], "n_fail_all": all_["n_fail"],
            "acc_train": tr["acc"], "acc_val": va["acc"], "acc_test": te["acc"],
        })
    df = pd.DataFrame(rows).sort_values("acc_all", ascending=False)
    df.to_csv(OUT_DIR / "iter9_final_benchmark.csv", index=False)
    print("\n=== FINAL BENCHMARK ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
