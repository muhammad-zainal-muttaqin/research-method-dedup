"""
Iteration 8 step 2 — refined selector with B4 cap on iter7 rule.

Hypothesis: residual trees have high naive_B4 (median 9 train) — they
fall into iter7 trigger zone (B1>=3, B3frac<0.45) but adaptive_corrected
is wrong for them (geo_blend is right). Add B4 cap to skip routing
when B4 is dominant.
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


def selector_factory(b4_cap=None, b4_ratio_cap=None):
    """Refined: route to adaptive when B1>=3, B3frac<0.45, AND B4 conditions allow."""
    def _fn(dets):
        n_total = len(dets)
        if n_total == 0:
            return geometric_mean_blend(dets)
        naive = base.naive_count(dets)
        b3frac = naive["B3"] / n_total
        b4frac = naive["B4"] / n_total
        if naive["B1"] >= 3 and b3frac < 0.45:
            if b4_cap is not None and naive["B4"] >= b4_cap:
                return geometric_mean_blend(dets)
            if b4_ratio_cap is not None and b4frac >= b4_ratio_cap:
                return geometric_mean_blend(dets)
            return base.adaptive_corrected(dets)
        return geometric_mean_blend(dets)
    return _fn


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
        return {"acc": 0.0, "mae": 0.0, "n": 0}
    ok, maes = 0, []
    for info in items:
        pred = fn(info["dets"])
        ok += int(_within1(pred, info["gt"]))
        maes.append(_mae(pred, info["gt"]))
    return {"acc": round(100.0 * ok / n, 2), "mae": round(float(np.mean(maes)), 4), "n": n}


def main():
    base._load_v6_params()
    trees = load_with_split()

    candidates = {
        "iter7_baseline": selector_factory(),  # no B4 cap
    }
    for cap in (5, 6, 7, 8, 9, 10, 12):
        candidates[f"iter8_b4cap_{cap}"] = selector_factory(b4_cap=cap)
    for rc in (0.18, 0.20, 0.22, 0.25, 0.28):
        candidates[f"iter8_b4rcap_{int(rc*100):02d}"] = selector_factory(b4_ratio_cap=rc)

    rows = []
    bl = {"all": 86.46, "train": 87.17, "val": 82.02, "test": 88.62}  # iter7
    for name, fn in candidates.items():
        all_ = evaluate(fn, trees, None)
        tr = evaluate(fn, trees, "train")
        va = evaluate(fn, trees, "val")
        te = evaluate(fn, trees, "test")
        rows.append({
            "method": name,
            "acc_all": all_["acc"], "mae_all": all_["mae"],
            "acc_train": tr["acc"], "acc_val": va["acc"], "acc_test": te["acc"],
            "mae_train": tr["mae"], "mae_val": va["mae"], "mae_test": te["mae"],
            "d_all": round(all_["acc"] - bl["all"], 2),
            "d_train": round(tr["acc"] - bl["train"], 2),
            "d_val": round(va["acc"] - bl["val"], 2),
            "d_test": round(te["acc"] - bl["test"], 2),
        })

    df = pd.DataFrame(rows)
    df["worst_drop"] = df[["d_train", "d_val", "d_test"]].min(axis=1)
    df["passes"] = (df["worst_drop"] >= -0.3) & (df["acc_all"] > bl["all"])
    df = df.sort_values(["passes", "acc_all"], ascending=[False, False])
    df.to_csv(OUT_DIR / "iter8_refined_results.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
