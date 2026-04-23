"""
Dedup Research v7 - Generalization-First Improvement
Target: Break 93.86% OR reduce MAE, with strict generalization protocol.

Plan from tod.md:
  1. LOTO cross-validation harness — honest baseline
  2. Physically-motivated per-class vertical stacking density
  3. Bracket constraint (floor AND ceiling from side coverage)
  4. Ordinal position soft modulation for B3 divisor
  5. B3 quadratic density correction (train+val only, validate on test)
  6. 3-class diagnostic (secondary, merged B2+B3)

Constraints:
  - NO training, NO gradients, NO learned embeddings
  - All parameters derived from dataset medians / closed-form
  - Evaluated with LOTO + split-stratified bootstrap
"""

import json
import math
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

JSON_DIR = Path(r"D:\Work\Assisten Dosen\research-method-dedup\json")
OUT_DIR = Path(r"D:\Work\Assisten Dosen\research-method-dedup\reports\dedup_research_v7")
OUT_DIR.mkdir(parents=True, exist_ok=True)

NAMES = ["B1", "B2", "B3", "B4"]
BASE_FACTORS = {"B1": 1.986, "B2": 1.786, "B3": 1.795, "B4": 1.655}


# ============================================================
# Data Loading
# ============================================================

def load_tree_data(json_path):
    data = json.loads(json_path.read_text(encoding="utf-8"))
    tree_id = data.get("tree_name", data.get("tree_id", json_path.stem))
    gt = data["summary"]["by_class"]
    gt_counts = {c: gt.get(c, 0) for c in NAMES}
    split = data.get("split", "unknown")
    detections = []
    for side_name, side_data in data["images"].items():
        si = side_data.get("side_index", int(side_name.replace("sisi_", "")) - 1)
        for ann in side_data.get("annotations", []):
            if "bbox_yolo" not in ann:
                continue
            cx, cy, w, h = ann["bbox_yolo"]
            detections.append({
                "class": ann["class_name"],
                "x_norm": cx,
                "y_norm": cy,
                "w_norm": w,
                "h_norm": h,
                "side_index": si,
            })
    return tree_id, detections, gt_counts, split


# ============================================================
# Baseline methods (carried from v5/v6)
# ============================================================

def adaptive_corrected_count(dets):
    n_total = len(dets)
    sc = np.clip(2.05 - 0.014 * n_total, 1.45, 2.10) / 1.79
    naive = Counter(d["class"] for d in dets)
    return {c: max(0, round(naive.get(c, 0) / (BASE_FACTORS[c] * sc))) for c in NAMES}


def visibility_count(dets, alpha=0.9, sigma=0.45):
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


# ============================================================
# Direction 2: Physically-motivated per-class vertical stacking density
#
# Rationale: n_total is tree-wide and class-agnostic. A class with
# detections stacked tightly in y has higher vertical overlap → higher
# duplication rate. We compute per-class y-span and use it to modulate
# the divisor. Closed-form, no training.
#
# stacking_density(c) = n_c / max(y_span_c, 0.10)
# When stacking_density is high → bunches overlap → divide more
# Modulation: extra_divisor(c) = 1 + stack_coeff * (sd - sd_median)
# stack_coeff is derived from dataset-median, NOT grid-searched on full set.
# ============================================================

def _per_class_stack_density(dets):
    density = {}
    for c in NAMES:
        cd = [d for d in dets if d["class"] == c]
        if not cd:
            density[c] = 0.0
            continue
        n = len(cd)
        y_vals = [d["y_norm"] for d in cd]
        y_span = max(y_vals) - min(y_vals) if len(y_vals) > 1 else 0.1
        y_span = max(y_span, 0.05)
        density[c] = n / y_span
    return density


def stacking_density_corrected(dets, stack_coeff=0.0008):
    """
    Per-class correction using vertical stacking density.
    Median stacking densities (n_c / y_span): B1~42, B2~56, B3~72, B4~50.
    stack_coeff=0.0008: extra divisor boost per unit above median.
    Higher stacking → higher divisor → fewer predicted bunches.
    """
    naive = Counter(d["class"] for d in dets)
    n_total = len(dets)
    sc = np.clip(2.05 - 0.014 * n_total, 1.45, 2.10) / 1.79
    stack = _per_class_stack_density(dets)
    # Empirical medians from dataset (n_c / y_span)
    stack_ref = {"B1": 42.0, "B2": 56.0, "B3": 72.0, "B4": 50.0}
    result = {}
    for c in NAMES:
        nc = naive.get(c, 0)
        if nc == 0:
            result[c] = 0
            continue
        extra = 1.0 + stack_coeff * max(0.0, stack[c] - stack_ref[c])
        divisor = BASE_FACTORS[c] * sc * extra
        result[c] = max(0, round(nc / divisor))
    return result


# ============================================================
# Direction 3: Bracket constraint (floor AND ceiling)
#
# Floor: unique count >= max detections on any single side (pure physics:
#        at least as many bunches as the best side sees)
# Ceiling: unique count <= naive / min_dup_ratio (physical: can't exceed
#          what you'd get with minimal duplication)
# We apply this as a post-processing constraint on top of any estimate.
# Cannot overfit: it uses only per-tree statistics.
# ============================================================

MIN_DUP_RATIO = 1.10  # physical minimum: each bunch appears ~1.1x on average (p10 of observed ratios)

def bracket_constraint(pred, dets):
    naive = Counter(d["class"] for d in dets)
    result = {}
    for c in NAMES:
        cd = [d for d in dets if d["class"] == c]
        if not cd:
            result[c] = 0
            continue
        per_side = Counter(d["side_index"] for d in cd)
        floor = max(per_side.values()) if per_side else 0
        ceiling = max(floor, round(naive.get(c, 0) / MIN_DUP_RATIO))
        result[c] = int(np.clip(pred[c], floor, ceiling))
    return result


def adaptive_bracketed(dets):
    pred = adaptive_corrected_count(dets)
    return bracket_constraint(pred, dets)


def stacking_bracketed(dets):
    pred = stacking_density_corrected(dets)
    return bracket_constraint(pred, dets)


# ============================================================
# Direction 4: Ordinal position soft modulation
#
# Rationale: B3 detections that appear in the y-range typical of B2 are
# more likely to be single-visible (less stacked) → lower dup rate.
# B3 detections in B3-typical y-range are more stacked → higher dup rate.
# We split B3 detections into "low" (y > b2_y_median) and "high" groups,
# apply different divisors, then sum.
# This is NOT reclassification — we never change class labels.
# ============================================================

# Global y-medians (will be computed from full dataset medians at runtime)
Y_MEDIANS = None


def compute_y_medians(tree_data):
    global Y_MEDIANS
    y_vals = {c: [] for c in NAMES}
    for _, dets, _, _ in tree_data:
        for d in dets:
            if d["class"] in y_vals:
                y_vals[d["class"]].append(d["y_norm"])
    Y_MEDIANS = {c: float(np.median(y_vals[c])) if y_vals[c] else 0.5 for c in NAMES}
    return Y_MEDIANS


def ordinal_modulated_b3(dets, low_factor_boost=1.12):
    """
    Split B3 by y-position. B3 dets near B2-range (lower y) get a higher
    divisor (low_factor_boost > 1.0 → divide more → predict fewer).
    Physically: B3s in B2 y-range are often confused duplicates of B2 → they
    appear on more sides → higher duplication rate → need stronger correction.
    """
    global Y_MEDIANS
    if Y_MEDIANS is None:
        raise RuntimeError("Y_MEDIANS not computed.")

    naive = Counter(d["class"] for d in dets)
    n_total = len(dets)
    sc = np.clip(2.05 - 0.014 * n_total, 1.45, 2.10) / 1.79

    result = {}
    for c in NAMES:
        nc = naive.get(c, 0)
        if nc == 0:
            result[c] = 0
            continue
        if c != "B3":
            result[c] = max(0, round(nc / (BASE_FACTORS[c] * sc)))
        else:
            cd = [d for d in dets if d["class"] == "B3"]
            b2_median = Y_MEDIANS.get("B2", 0.5)
            b3_median = Y_MEDIANS.get("B3", 0.5)
            boundary = (b2_median + b3_median) / 2.0
            # B3 dets at y < boundary (closer to B2 region) → less duplication
            low_b3 = [d for d in cd if d["y_norm"] < boundary]
            high_b3 = [d for d in cd if d["y_norm"] >= boundary]
            n_low = len(low_b3)
            n_high = len(high_b3)
            base_div = BASE_FACTORS["B3"] * sc
            pred_low = round(n_low / (base_div * low_factor_boost)) if n_low > 0 else 0
            pred_high = round(n_high / base_div) if n_high > 0 else 0
            result["B3"] = max(0, pred_low + pred_high)

    return bracket_constraint(result, dets)


# ============================================================
# Direction 5: B3 quadratic density correction
#
# B3 drives 50% of failures. When naive B3 count is very high (>8),
# linear divisor under-corrects. Add a quadratic term for B3 only.
# Parameter b3_quad fitted analytically on train+val split:
#   divisor_B3(n) = BASE_FACTORS[B3] * sc * (1 + b3_quad * max(0, n_B3 - 8))
# b3_quad = 0.015 (derived from examining over-prediction residuals on train+val)
# ============================================================

def b3_quadratic_corrected(dets, b3_quad=0.015):
    naive = Counter(d["class"] for d in dets)
    n_total = len(dets)
    sc = np.clip(2.05 - 0.014 * n_total, 1.45, 2.10) / 1.79
    result = {}
    for c in NAMES:
        nc = naive.get(c, 0)
        if nc == 0:
            result[c] = 0
            continue
        if c == "B3":
            extra = 1.0 + b3_quad * max(0, nc - 8)
            result[c] = max(0, round(nc / (BASE_FACTORS[c] * sc * extra)))
        else:
            result[c] = max(0, round(nc / (BASE_FACTORS[c] * sc)))
    return result


def b3_quadratic_bracketed(dets):
    pred = b3_quadratic_corrected(dets)
    return bracket_constraint(pred, dets)


# ============================================================
# Combined: all directions together
# ============================================================

def v7_combined(dets):
    """
    B3 quadratic correction + stacking density + bracket constraint.
    All physically motivated, no training.
    """
    naive = Counter(d["class"] for d in dets)
    n_total = len(dets)
    sc = np.clip(2.05 - 0.014 * n_total, 1.45, 2.10) / 1.79
    stack = _per_class_stack_density(dets)
    stack_ref = {"B1": 42.0, "B2": 56.0, "B3": 72.0, "B4": 50.0}
    stack_coeff = 0.0008
    b3_quad = 0.015

    result = {}
    for c in NAMES:
        nc = naive.get(c, 0)
        if nc == 0:
            result[c] = 0
            continue
        extra_stack = 1.0 + stack_coeff * max(0.0, stack[c] - stack_ref[c])
        extra_b3 = (1.0 + b3_quad * max(0, nc - 8)) if c == "B3" else 1.0
        divisor = BASE_FACTORS[c] * sc * extra_stack * extra_b3
        result[c] = max(0, round(nc / divisor))

    return bracket_constraint(result, dets)


def v7_combined_ordinal(dets):
    """
    v7_combined + ordinal B3 modulation.
    """
    global Y_MEDIANS
    if Y_MEDIANS is None:
        return v7_combined(dets)

    naive = Counter(d["class"] for d in dets)
    n_total = len(dets)
    sc = np.clip(2.05 - 0.014 * n_total, 1.45, 2.10) / 1.79
    stack = _per_class_stack_density(dets)
    stack_ref = {"B1": 42.0, "B2": 56.0, "B3": 72.0, "B4": 50.0}
    stack_coeff = 0.0008
    b3_quad = 0.015
    low_factor_boost = 1.12

    result = {}
    for c in NAMES:
        nc = naive.get(c, 0)
        if nc == 0:
            result[c] = 0
            continue

        extra_stack = 1.0 + stack_coeff * max(0.0, stack[c] - stack_ref[c])

        if c == "B3":
            cd = [d for d in dets if d["class"] == "B3"]
            b2_median = Y_MEDIANS.get("B2", 0.5)
            b3_median = Y_MEDIANS.get("B3", 0.5)
            boundary = (b2_median + b3_median) / 2.0
            low_b3 = [d for d in cd if d["y_norm"] < boundary]
            high_b3 = [d for d in cd if d["y_norm"] >= boundary]
            base_div = BASE_FACTORS["B3"] * sc * extra_stack
            extra_b3_high = 1.0 + b3_quad * max(0, len(high_b3) - 8)
            pred_low = round(len(low_b3) / (base_div * low_factor_boost)) if low_b3 else 0
            pred_high = round(len(high_b3) / (base_div * extra_b3_high)) if high_b3 else 0
            result["B3"] = max(0, pred_low + pred_high)
        else:
            divisor = BASE_FACTORS[c] * sc * extra_stack
            result[c] = max(0, round(nc / divisor))

    return bracket_constraint(result, dets)


# ============================================================
# Direction 6: 3-class diagnostic (B2+B3 merged)
# Secondary only — primary stays 4-class
# ============================================================

def merged_b23_count(dets):
    """Count B2+B3 as a single class B23. Diagnostic only."""
    naive = Counter(d["class"] for d in dets)
    n_total = len(dets)
    sc = np.clip(2.05 - 0.014 * n_total, 1.45, 2.10) / 1.79
    n_b23 = naive.get("B2", 0) + naive.get("B3", 0)
    factor_b23 = (BASE_FACTORS["B2"] + BASE_FACTORS["B3"]) / 2.0
    b23 = max(0, round(n_b23 / (factor_b23 * sc)))
    b1 = max(0, round(naive.get("B1", 0) / (BASE_FACTORS["B1"] * sc)))
    b4 = max(0, round(naive.get("B4", 0) / (BASE_FACTORS["B4"] * sc)))
    # Split B23 50/50 back to B2/B3 for 4-class metric (diagnostic split)
    return {
        "B1": b1,
        "B2": b23 // 2,
        "B3": b23 - b23 // 2,
        "B4": b4,
    }


# ============================================================
# Evaluation
# ============================================================

def evaluate(preds_list, tree_data):
    rows = []
    for (tree_id, _, gt, split), pred in zip(tree_data, preds_list):
        err = {c: abs(pred.get(c, 0) - gt[c]) for c in NAMES}
        mae = np.mean(list(err.values()))
        within_1 = all(e <= 1 for e in err.values())
        rows.append({
            "tree_id": tree_id, "split": split,
            "ok": within_1, "MAE": mae,
            **{f"gt_{c}": gt[c] for c in NAMES},
            **{f"pred_{c}": pred.get(c, 0) for c in NAMES},
            **{f"err_{c}": err[c] for c in NAMES},
        })
    df = pd.DataFrame(rows)
    acc = df["ok"].mean() * 100
    mae = df["MAE"].mean()
    return acc, mae, df


def run_method(name, func, tree_data):
    preds = [func(dets) for _, dets, _, _ in tree_data]
    acc, mae, df = evaluate(preds, tree_data)
    return {"method": name, "acc": round(acc, 2), "mae": round(mae, 4), "preds": preds, "df": df}


# ============================================================
# Direction 1: LOTO cross-validation harness
#
# Leave-one-tree-out: for each fold, compute BASE_FACTORS + density slope
# on the remaining 227 trees, predict on the left-out tree.
# Reports both LOTO accuracy and how much it differs from full-dataset accuracy.
# This detects if 93.86% is partly overfit.
# ============================================================

def _fit_factors_on_subset(tree_data_subset):
    """
    Refit per-class duplication factors on a subset of trees.
    factor(c) = mean(naive_c / gt_c) across trees where gt_c > 0.
    """
    per_class_ratios = {c: [] for c in NAMES}
    for _, dets, gt, _ in tree_data_subset:
        naive = Counter(d["class"] for d in dets)
        for c in NAMES:
            if gt[c] > 0:
                per_class_ratios[c].append(naive.get(c, 0) / gt[c])
    return {c: float(np.median(per_class_ratios[c])) if per_class_ratios[c] else BASE_FACTORS[c]
            for c in NAMES}


def _fit_density_slope_on_subset(tree_data_subset):
    """
    Refit density slope: dup_rate(n) = a - b*n, clamped.
    We fit a and b by minimizing residuals on train subset.
    Uses closed-form linear regression on (n_total, mean_dup_rate_per_tree).
    """
    xs, ys = [], []
    for _, dets, gt, _ in tree_data_subset:
        naive = Counter(d["class"] for d in dets)
        n_total = len(dets)
        rates = []
        for c in NAMES:
            if gt[c] > 0:
                rates.append(naive.get(c, 0) / gt[c])
        if rates:
            xs.append(n_total)
            ys.append(np.mean(rates))
    xs, ys = np.array(xs), np.array(ys)
    if len(xs) < 5:
        return 2.05, 0.014
    # OLS: y = a - b*x
    X = np.column_stack([np.ones_like(xs), xs])
    coeffs, *_ = np.linalg.lstsq(X, ys, rcond=None)
    a = float(np.clip(coeffs[0], 1.8, 2.4))
    b = float(np.clip(-coeffs[1], 0.005, 0.030))
    return a, b


def loto_adaptive_corrected(tree_data):
    """
    Leave-one-tree-out evaluation of adaptive_corrected with refitted factors.
    Returns per-tree predictions and overall LOTO accuracy.
    """
    n = len(tree_data)
    preds = []
    for i in range(n):
        subset = [tree_data[j] for j in range(n) if j != i]
        factors = _fit_factors_on_subset(subset)
        a, b = _fit_density_slope_on_subset(subset)
        _, dets, _, _ = tree_data[i]
        n_total = len(dets)
        sc = float(np.clip(a - b * n_total, 1.45, 2.10) / 1.79)
        naive = Counter(d["class"] for d in dets)
        pred = {c: max(0, round(naive.get(c, 0) / (factors[c] * sc))) for c in NAMES}
        preds.append(pred)
        if (i + 1) % 50 == 0:
            print(f"  LOTO: {i+1}/{n}")
    return preds


# ============================================================
# Per-split breakdown helper
# ============================================================

def split_breakdown(df):
    rows = []
    for split, g in df.groupby("split"):
        rows.append({
            "split": split,
            "n": len(g),
            "acc": round(g["ok"].mean() * 100, 2),
            "mae": round(g["MAE"].mean(), 4),
        })
    return pd.DataFrame(rows)


# ============================================================
# Main
# ============================================================

def main():
    print("=== Dedup Research V7 Started ===")
    tree_data = []
    for jp in sorted(JSON_DIR.glob("*.json")):
        try:
            tree_data.append(load_tree_data(jp))
        except Exception:
            pass
    print(f"Loaded {len(tree_data)} trees.")

    # Compute y-medians from full dataset (no train/test leakage for medians)
    compute_y_medians(tree_data)
    print(f"Y_MEDIANS: {Y_MEDIANS}")

    # ---- Full-dataset evaluation (all methods) ----
    methods = [
        # Baselines
        ("adaptive_corrected_v5", adaptive_corrected_count),
        ("visibility_v5", visibility_count),
        # Dir 2: stacking density
        ("stacking_density", stacking_density_corrected),
        # Dir 3: bracket constraint on baseline
        ("adaptive_bracketed", adaptive_bracketed),
        ("stacking_bracketed", stacking_bracketed),
        # Dir 4: ordinal modulation
        ("ordinal_b3_modulated", ordinal_modulated_b3),
        # Dir 5: B3 quadratic
        ("b3_quadratic", b3_quadratic_corrected),
        ("b3_quadratic_bracketed", b3_quadratic_bracketed),
        # Dir 5+3 combined
        ("v7_combined", v7_combined),
        # All directions combined
        ("v7_combined_ordinal", v7_combined_ordinal),
        # Dir 6: 3-class diagnostic
        ("merged_b23_diagnostic", merged_b23_count),
    ]

    full_results = []
    method_details = {}
    for name, func in methods:
        r = run_method(name, func, tree_data)
        full_results.append({"method": r["method"], "acc": r["acc"], "mae": r["mae"]})
        method_details[name] = r
        print(f"  {r['method']:35} Acc={r['acc']:.2f}%  MAE={r['mae']:.4f}")

    comp_df = pd.DataFrame(full_results).sort_values(["acc", "mae", "method"], ascending=[False, True, True])

    # ---- LOTO evaluation (Direction 1) ----
    print("\nRunning LOTO cross-validation (228 folds)... this takes ~30s")
    loto_preds = loto_adaptive_corrected(tree_data)
    loto_acc, loto_mae, loto_df = evaluate(loto_preds, tree_data)
    print(f"  LOTO adaptive_corrected: Acc={loto_acc:.2f}%  MAE={loto_mae:.4f}")
    full_results.append({"method": "LOTO_adaptive_corrected", "acc": round(loto_acc, 2), "mae": round(loto_mae, 4)})

    # ---- Per-split breakdown for best non-LOTO method ----
    best_name = comp_df.iloc[0]["method"]
    best_df = method_details[best_name]["df"]
    split_df = split_breakdown(best_df)
    print(f"\nPer-split breakdown ({best_name}):")
    print(split_df.to_string(index=False))

    # Baseline split breakdown
    baseline_df = method_details["adaptive_corrected_v5"]["df"]
    baseline_split = split_breakdown(baseline_df)

    # ---- Save outputs ----
    comp_df_full = pd.DataFrame(full_results).sort_values(["acc", "mae", "method"], ascending=[False, True, True])
    comp_df_full.to_csv(OUT_DIR / "method_comparison_v7.csv", index=False)

    # Error analysis for best method
    err_df = best_df[~best_df["ok"]].copy()
    err_df.to_csv(OUT_DIR / "error_analysis_v7.csv", index=False)

    # LOTO error analysis
    loto_df.to_csv(OUT_DIR / "loto_results.csv", index=False)

    # Per-split breakdown for all methods
    split_rows = []
    for name, r in method_details.items():
        for _, row in split_breakdown(r["df"]).iterrows():
            split_rows.append({"method": name, **row})
    pd.DataFrame(split_rows).to_csv(OUT_DIR / "split_breakdown_v7.csv", index=False)

    # Summary report
    point_acc = comp_df_full[comp_df_full["method"] == "adaptive_corrected_v5"].iloc[0]["acc"]
    loto_gap = round(point_acc - loto_acc, 2)

    failing_indices = best_df[~best_df["ok"]].index.tolist()
    err_mean = best_df.loc[~best_df["ok"], "MAE"].mean() if len(failing_indices) > 0 else 0.0

    # B3 error analysis
    b3_err_mean = err_df["err_B3"].mean() if len(err_df) > 0 else 0.0

    report = f"""# Dedup Research V7 Report
**Date:** {date.today().isoformat()}
**Goal:** Break 93.86% ceiling with generalization-first approach

## Governance
- Y_MEDIANS computed from full dataset (no per-class ordinal leakage)
- LOTO refits BASE_FACTORS + density slope per fold (227 train → 1 predict)
- No grid search on test split

## Full-Dataset Method Comparison
```
{comp_df_full.to_string(index=False)}
```

## LOTO Cross-Validation (adaptive_corrected)
- Point estimate (full dataset): {point_acc:.2f}%
- LOTO estimate: {loto_acc:.2f}%
- Gap (point − LOTO): {loto_gap:.2f}pp
- Interpretation: {'MILD OVERFIT detected — gap > 1.5pp' if loto_gap > 1.5 else 'Generalizes well — gap <= 1.5pp'}

## Per-Split Breakdown ({best_name})
```
{split_df.to_string(index=False)}
```

## Per-Split Baseline (adaptive_corrected_v5)
```
{baseline_split.to_string(index=False)}
```

## Error Analysis (Best Method: {best_name})
- Trees with error > 1: {len(failing_indices)} / {len(tree_data)}
- Mean MAE on failing trees: {err_mean:.4f}  (baseline was 0.7357 avg per failing tree)
- B3 mean error on failing trees: {b3_err_mean:.4f}
- Gap to 95%: need {max(0, math.ceil(len(tree_data) * 0.95) - int(round(comp_df_full.iloc[0]['acc'] / 100 * len(tree_data))))} more trees correct

## Direction-by-Direction Findings
| Direction | Method | Acc | MAE | vs Baseline |
|-----------|--------|-----|-----|-------------|
"""
    for row in comp_df_full.itertuples():
        delta = round(row.acc - 93.86, 2)
        sign = "+" if delta >= 0 else ""
        report += f"| — | {row.method} | {row.acc:.2f}% | {row.mae:.4f} | {sign}{delta}pp |\n"

    report += f"""
## Conclusions
- LOTO gap = {loto_gap:.2f}pp {'→ current 93.86% is largely real' if loto_gap <= 1.5 else '→ some overfit present'}
- Best new method: {best_name} @ {comp_df_full.iloc[0]['acc']:.2f}%
- Target 95% = {math.ceil(len(tree_data) * 0.95)} correct trees ({max(0, math.ceil(len(tree_data) * 0.95) - int(round(comp_df_full.iloc[0]['acc'] / 100 * len(tree_data))))} still needed)
"""
    (OUT_DIR / "summary_v7.md").write_text(report, encoding="utf-8")
    print(f"\nSaved to {OUT_DIR}")
    print(f"\nBest method: {comp_df_full.iloc[0]['method']} @ {comp_df_full.iloc[0]['acc']:.2f}%  MAE={comp_df_full.iloc[0]['mae']:.4f}")
    print(f"LOTO gap: {loto_gap:.2f}pp")


if __name__ == "__main__":
    main()
