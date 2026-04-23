"""
Focused V5 search: find best ensemble & hybrid to break 95%.
Skips slow grid searches; uses best-known params from previous V5 run.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from itertools import combinations

JSON_DIR = Path(r"D:\Work\Assisten Dosen\research-method-dedup\json")
NAMES = ["B1", "B2", "B3", "B4"]


def load_tree_data(json_path):
    data = json.loads(json_path.read_text(encoding="utf-8"))
    tree_id = data.get("tree_name", data.get("tree_id", json_path.stem))
    gt = data["summary"]["by_class"]
    gt_counts = {c: gt.get(c, 0) for c in NAMES}
    all_detections = []
    for side_name, side_data in data["images"].items():
        si = side_data.get("side_index", int(side_name.replace("sisi_", "")) - 1)
        for ann in side_data.get("annotations", []):
            if "bbox_yolo" not in ann:
                continue
            cx, cy, w, h = ann["bbox_yolo"]
            all_detections.append(
                {
                    "class": ann["class_name"],
                    "y_norm": cy,
                    "x_norm": cx,
                    "area_norm": w * h,
                    "side_index": si,
                }
            )
    return tree_id, all_detections, gt_counts


tree_data = []
for jp in sorted(JSON_DIR.glob("*.json")):
    try:
        tree_data.append(load_tree_data(jp))
    except Exception:
        pass


def naive_count(dets):
    c = Counter(d["class"] for d in dets)
    return {k: c.get(k, 0) for k in NAMES}


def corrected_naive(dets, factors=None):
    if factors is None:
        factors = {"B1": 1.986, "B2": 1.786, "B3": 1.795, "B4": 1.655}
    n = naive_count(dets)
    return {c: max(0, round(n[c] / factors[c])) for c in NAMES}


def visibility_count(dets, alpha=1.0, sigma=0.3):
    counts = {}
    for c in NAMES:
        cd = [d for d in dets if d["class"] == c]
        if not cd:
            counts[c] = 0
            continue
        total = sum(
            1.0 / (1.0 + alpha * np.exp(-((d["x_norm"] - 0.5) ** 2) / (2.0 * sigma ** 2)))
            for d in cd
        )
        counts[c] = max(0, int(round(total)))
    return counts


def _tree_density_scale(n_total):
    dup_rate = 2.05 - 0.014 * n_total
    dup_rate = np.clip(dup_rate, 1.45, 2.10)
    return float(dup_rate / 1.79)


def adaptive_corrected_count(dets):
    n_total = len(dets)
    base = {"B1": 1.986, "B2": 1.786, "B3": 1.795, "B4": 1.655}
    scale = _tree_density_scale(n_total)
    factors = {c: base[c] * scale for c in NAMES}
    return corrected_naive(dets, factors)


def density_scaled_visibility(dets):
    n_total = len(dets)
    vis = visibility_count(dets)
    boost = 1.0 + 0.025 * (n_total - 12) / 12.0
    boost = np.clip(boost, 0.92, 1.15)
    return {c: max(0, int(round(vis[c] * boost))) for c in NAMES}


def side_coverage_count(dets):
    vis = visibility_count(dets)
    naive = naive_count(dets)
    counts = {}
    for c in NAMES:
        cd = [d for d in dets if d["class"] == c]
        if not cd:
            counts[c] = 0
            continue
        per_side = Counter(d["side_index"] for d in cd)
        mps = max(per_side.values()) if per_side else 0
        pred = max(vis[c], mps)
        pred = min(pred, naive[c])
        counts[c] = pred
    return counts


def hybrid_vis_corr(dets, w=0.6):
    vis = visibility_count(dets)
    corr = adaptive_corrected_count(dets)
    return {c: max(0, int(round(w * vis[c] + (1 - w) * corr[c]))) for c in NAMES}


def median_ensemble(preds_list):
    return {c: int(round(np.median([p[c] for p in preds_list]))) for c in NAMES}


def trimmed_mean_ensemble(preds_list):
    result = {}
    for c in NAMES:
        vals = sorted([p[c] for p in preds_list])
        if len(vals) <= 2:
            result[c] = int(round(np.mean(vals)))
        else:
            result[c] = int(round(np.mean(vals[1:-1])))
    return result


def evaluate(preds_list):
    rows = []
    for (_, _, gt), pred in zip(tree_data, preds_list):
        err = {c: abs(pred.get(c, 0) - gt[c]) for c in NAMES}
        rows.append(
            {
                "ok": all(e <= 1 for e in err.values()),
                **{f"err_{c}": err[c] for c in NAMES},
            }
        )
    df = pd.DataFrame(rows)
    acc = df["ok"].mean() * 100
    mae = df[[f"err_{c}" for c in NAMES]].mean(axis=1).mean()
    return acc, mae, df


# Precompute predictions for key methods
methods = {
    "naive": naive_count,
    "corrected": corrected_naive,
    "visibility": visibility_count,
    "adaptive_corrected": adaptive_corrected_count,
    "density_scaled_vis": density_scaled_visibility,
    "side_coverage": side_coverage_count,
    "hybrid_vis_corr": hybrid_vis_corr,
    "best_vis_grid": lambda dets: visibility_count(dets, alpha=0.9, sigma=0.45),
}

print("Precomputing method predictions...")
all_preds = {name: [func(dets) for _, dets, _ in tree_data] for name, func in methods.items()}

print("\n=== Individual Method Accuracies ===")
for name in methods:
    acc, mae, _ = evaluate(all_preds[name])
    print(f"{name:25} Acc={acc:.2f}%  MAE={mae:.4f}")

# Focused ensemble search over top methods
pool = [
    "adaptive_corrected",
    "visibility",
    "best_vis_grid",
    "side_coverage",
    "hybrid_vis_corr",
    "corrected",
    "density_scaled_vis",
]

print("\n=== Ensemble Search ===")
best = None
for size in range(3, min(6, len(pool) + 1)):
    for subset in combinations(pool, size):
        subset_preds = [all_preds[name] for name in subset]
        for agg_name, agg_func in [
            ("median", median_ensemble),
            ("trimmed_mean", trimmed_mean_ensemble),
        ]:
            preds = []
            for i in range(len(tree_data)):
                preds.append(agg_func([sp[i] for sp in subset_preds]))
            acc, mae, _ = evaluate(preds)
            if best is None or acc > best["acc"]:
                best = {
                    "acc": acc,
                    "mae": mae,
                    "subset": subset,
                    "agg": agg_name,
                }
                print(
                    f"New best: {acc:.2f}% (MAE {mae:.4f}) with {agg_name} over {subset}"
                )

print(f"\n=== FINAL BEST ENSEMBLE ===")
print(
    f"{best['acc']:.2f}% with {best['agg']} over {best['subset']}"
)

# Try a few hand-crafted hybrids
print("\n=== Hand-Crafted Hybrids ===")

# 1. median(adaptive_corrected, visibility)
hybrid_preds = []
for i in range(len(tree_data)):
    hybrid_preds.append(
        median_ensemble([all_preds["adaptive_corrected"][i], all_preds["visibility"][i]])
    )
acc, mae, _ = evaluate(hybrid_preds)
print(f"median(adaptive_corrected, visibility): {acc:.2f}%  MAE={mae:.4f}")

# 2. median(adaptive_corrected, best_vis_grid, side_coverage)
hybrid_preds = []
for i in range(len(tree_data)):
    hybrid_preds.append(
        median_ensemble(
            [
                all_preds["adaptive_corrected"][i],
                all_preds["best_vis_grid"][i],
                all_preds["side_coverage"][i],
            ]
        )
    )
acc, mae, _ = evaluate(hybrid_preds)
print(f"median(adaptive_corrected, best_vis_grid, side_coverage): {acc:.2f}%  MAE={mae:.4f}")

# 3. adaptive_corrected but clamped to [max_per_side, naive]
def corrected_clamped(dets):
    corr = adaptive_corrected_count(dets)
    naive = naive_count(dets)
    counts = {}
    for c in NAMES:
        cd = [d for d in dets if d["class"] == c]
        if not cd:
            counts[c] = 0
            continue
        per_side = Counter(d["side_index"] for d in cd)
        mps = max(per_side.values()) if per_side else 0
        pred = max(corr[c], mps)
        pred = min(pred, naive[c])
        counts[c] = pred
    return counts

preds = [corrected_clamped(dets) for _, dets, _ in tree_data]
acc, mae, _ = evaluate(preds)
print(f"corrected_clamped: {acc:.2f}%  MAE={mae:.4f}")

# 4. Weighted blend: 0.7 adaptive_corrected + 0.3 visibility
def blend_70_30(dets):
    corr = adaptive_corrected_count(dets)
    vis = visibility_count(dets)
    return {c: max(0, int(round(0.7 * corr[c] + 0.3 * vis[c]))) for c in NAMES}

preds = [blend_70_30(dets) for _, dets, _ in tree_data]
acc, mae, _ = evaluate(preds)
print(f"blend_70_30 (adaptive_corrected, visibility): {acc:.2f}%  MAE={mae:.4f}")

# 5. Per-class switching: use visibility for B3/B4 (overcount issues) and adaptive_corrected for B1/B2
def switch_vis_corr(dets):
    corr = adaptive_corrected_count(dets)
    vis = visibility_count(dets)
    return {
        "B1": corr["B1"],
        "B2": corr["B2"],
        "B3": vis["B3"],
        "B4": vis["B4"],
    }

preds = [switch_vis_corr(dets) for _, dets, _ in tree_data]
acc, mae, _ = evaluate(preds)
print(f"switch_vis_corr (vis for B3/B4, corr for B1/B2): {acc:.2f}%  MAE={mae:.4f}")

# 6. Per-class switching v2: use best_vis_grid for B3/B4
def switch_vis2_corr(dets):
    corr = adaptive_corrected_count(dets)
    vis = visibility_count(dets, alpha=0.9, sigma=0.45)
    return {
        "B1": corr["B1"],
        "B2": corr["B2"],
        "B3": vis["B3"],
        "B4": vis["B4"],
    }

preds = [switch_vis2_corr(dets) for _, dets, _ in tree_data]
acc, mae, _ = evaluate(preds)
print(f"switch_vis2_corr (best_vis_grid for B3/B4, corr for B1/B2): {acc:.2f}%  MAE={mae:.4f}")

# 7. Try adaptive_corrected with gentler density slope
def adaptive_corrected_gentle(dets):
    n_total = len(dets)
    base = {"B1": 1.986, "B2": 1.786, "B3": 1.795, "B4": 1.655}
    dup_rate = 2.02 - 0.010 * n_total
    dup_rate = np.clip(dup_rate, 1.55, 2.10)
    scale = float(dup_rate / 1.79)
    factors = {c: base[c] * scale for c in NAMES}
    return corrected_naive(dets, factors)

preds = [adaptive_corrected_gentle(dets) for _, dets, _ in tree_data]
acc, mae, _ = evaluate(preds)
print(f"adaptive_corrected_gentle: {acc:.2f}%  MAE={mae:.4f}")

# 8. Try per-class density slopes
def per_class_density_corrected(dets):
    n_total = len(dets)
    # Custom per-class dup_rate slopes
    dup = {
        "B1": max(1.85, 2.08 - 0.010 * n_total),
        "B2": max(1.60, 1.95 - 0.015 * n_total),
        "B3": max(1.60, 1.94 - 0.016 * n_total),
        "B4": max(1.45, 1.80 - 0.020 * n_total),
    }
    naive = naive_count(dets)
    return {c: max(0, round(naive[c] / dup[c])) for c in NAMES}

preds = [per_class_density_corrected(dets) for _, dets, _ in tree_data]
acc, mae, _ = evaluate(preds)
print(f"per_class_density_corrected: {acc:.2f}%  MAE={mae:.4f}")

print("\n=== Done ===")
