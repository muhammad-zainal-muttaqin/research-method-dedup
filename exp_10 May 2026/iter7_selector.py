"""
Iteration 7 — deterministic selector geo vs adaptive_corrected.

Step 1: profile features on TRAIN split only.
Step 2: derive 1-2 feature rules from train.
Step 3: evaluate on val + test held-out.

Constraint: rule must be simple (1-2 thresholds), motivated by feature
distribution, and not regress any split > 0.3pp.
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


def features(dets):
    n_total = len(dets)
    naive = base.naive_count(dets)
    n_sides = len(set(d["side_index"] for d in dets))
    return {
        "n_dets": n_total,
        "n_sides": n_sides,
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
        "max_B4": _max_per_side(dets, "B4"),
        "ratio_B3_total": naive["B3"] / max(n_total, 1),
    }


def main():
    base._load_v6_params()
    trees = load_with_split()

    # ── classify each tree ────────────────────────────────────
    rows = []
    for tid, info in trees.items():
        dets = info["dets"]
        gt = info["gt"]
        pred_g = geometric_mean_blend(dets)
        pred_a = base.adaptive_corrected(dets)
        ok_g = _within1(pred_g, gt)
        ok_a = _within1(pred_a, gt)
        feat = features(dets)
        rows.append({
            "tree_id": tid,
            "split": info["split"],
            **feat,
            "geo_ok": int(ok_g),
            "adapt_ok": int(ok_a),
            "label": (
                "both_ok" if ok_g and ok_a else
                "geo_only" if ok_g and not ok_a else
                "adapt_only" if ok_a and not ok_g else
                "both_fail"
            ),
        })
    df = pd.DataFrame(rows)
    df_train = df[df["split"] == "train"]
    df_val = df[df["split"] == "val"]
    df_test = df[df["split"] == "test"]

    print("=== Class distribution per split ===")
    for sp_name, sp_df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        print(f"\n{sp_name} (n={len(sp_df)}):")
        print(sp_df["label"].value_counts().to_string())

    # ── feature profiling on TRAIN ────────────────────────────
    print("\n=== Feature means by label (TRAIN ONLY) ===")
    feat_cols = [c for c in df.columns if c not in {"tree_id", "split", "geo_ok", "adapt_ok", "label"}]
    print(df_train.groupby("label")[feat_cols].mean().round(2).to_string())

    print("\n=== Feature medians by label (TRAIN ONLY) ===")
    print(df_train.groupby("label")[feat_cols].median().round(2).to_string())

    # ── candidate rules: switch geo -> adaptive when condition holds ──
    # Build several thresholds from train profile, test on all splits.
    rules = {}
    # Rule A: high B3 dominance (B3 fraction > X)
    for thr in (0.30, 0.35, 0.40, 0.45):
        rules[f"R_B3frac_gt_{int(thr*100):02d}"] = lambda f, t=thr: f["ratio_B3_total"] > t
    # Rule B: high naive_total (dense trees)
    for thr in (24, 26, 28, 30):
        rules[f"R_naivetotal_ge_{thr}"] = lambda f, t=thr: f["naive_total"] >= t
    # Rule C: B4 active >= 3
    rules["R_activeB4_ge_3"] = lambda f: f["active_B4"] >= 3
    # Rule D: combined high naive_B3 AND active_B3
    for thr_b3 in (8, 10, 12):
        rules[f"R_naiveB3_ge_{thr_b3}"] = lambda f, t=thr_b3: f["naive_B3"] >= t
    # Rule E: naive_B1 threshold (train medians: adapt_only=4, geo_only=2)
    for thr in (3, 4, 5):
        rules[f"R_naiveB1_ge_{thr}"] = lambda f, t=thr: f["naive_B1"] >= t
    # Rule F: combined naive_B1 high AND active_B4 high (dense w/ B1+B4 presence)
    rules["R_B1ge3_AND_actB4ge3"] = lambda f: f["naive_B1"] >= 3 and f["active_B4"] >= 3
    rules["R_B1ge4_AND_actB4ge3"] = lambda f: f["naive_B1"] >= 4 and f["active_B4"] >= 3
    rules["R_B1ge3_AND_actB4ge4"] = lambda f: f["naive_B1"] >= 3 and f["active_B4"] >= 4
    rules["R_B1ge4_AND_actB4ge4"] = lambda f: f["naive_B1"] >= 4 and f["active_B4"] >= 4
    # Rule G: low B3 ratio (adapt_only had ratio 0.42 median, geo_only 0.58)
    for thr in (0.45, 0.50):
        rules[f"R_B3frac_lt_{int(thr*100):02d}"] = lambda f, t=thr: f["ratio_B3_total"] < t
    # Rule H: combined B1 high AND B3 ratio low
    rules["R_B1ge3_AND_B3frac_lt45"] = lambda f: f["naive_B1"] >= 3 and f["ratio_B3_total"] < 0.45

    # ── evaluate selectors ────────────────────────────────────
    selector_rows = []
    for name, rule in rules.items():
        for sp_name, sp_df in [("all", df), ("train", df_train), ("val", df_val), ("test", df_test)]:
            n = len(sp_df)
            n_pass = 0
            for _, r in sp_df.iterrows():
                fdict = {c: r[c] for c in feat_cols}
                use_adapt = rule(fdict)
                ok = r["adapt_ok"] if use_adapt else r["geo_ok"]
                n_pass += int(ok)
            acc = round(100.0 * n_pass / n, 2) if n else 0.0
            selector_rows.append({
                "rule": name, "split": sp_name, "acc": acc, "n": n,
            })

    sel_df = pd.DataFrame(selector_rows)
    pivot = sel_df.pivot(index="rule", columns="split", values="acc").reset_index()
    pivot = pivot[["rule", "all", "train", "val", "test"]]
    bl = {"all": 86.15, "train": 87.17, "val": 81.46, "test": 87.43}
    for col in ("all", "train", "val", "test"):
        pivot[f"d_{col}"] = (pivot[col] - bl[col]).round(2)
    pivot["worst_drop"] = pivot[["d_train", "d_val", "d_test"]].min(axis=1)
    pivot["n_up"] = (pivot[["d_train", "d_val", "d_test"]] > 0).sum(axis=1)
    pivot["passes"] = (pivot["worst_drop"] >= -0.3) & (pivot["all"] > bl["all"])
    pivot = pivot.sort_values(["passes", "all"], ascending=[False, False])
    pivot.to_csv(OUT_DIR / "iter7_selectors.csv", index=False)
    print("\n=== Selector results (vs baseline geo 86.15%) ===")
    print(pivot.to_string(index=False))


if __name__ == "__main__":
    main()
