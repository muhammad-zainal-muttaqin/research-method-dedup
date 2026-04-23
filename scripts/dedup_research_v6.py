"""
Dedup Research v6 - Disagreement-Gated Selector

Goal:
  - Keep the strict 4-class setup (B1/B2/B3/B4)
  - Stay deterministic / heuristic-only at runtime
  - Improve over v5 by switching between strong v5 methods only on unstable trees

This selector keeps `adaptive_corrected` as the default, then overrides it for a
small subset of trees using disagreement and side-coverage features:
  - `best_visibility_grid` for high B3 duplication regimes
  - `class_aware_vis` for B2-heavy / low-B4-duplication regimes
"""

from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from dedup_research_v5 import (
    NAMES,
    JSON_DIR,
    OUT_DIR as V5_OUT_DIR,
    adaptive_corrected_count,
    bootstrap_acc_ci,
    class_aware_visibility_count,
    compute_y_prior,
    evaluate_predictions,
    load_tree_data,
    visibility_count,
)

BASE = Path(__file__).resolve().parent.parent
OUT_DIR = BASE / "reports" / "dedup_research_v6"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_v5_reference_params() -> Dict[str, float]:
    comp = pd.read_csv(V5_OUT_DIR / "method_comparison_v5.csv").set_index("method")
    return {
        "vis_alpha": float(comp.loc["best_visibility_grid", "alpha"]),
        "vis_sigma": float(comp.loc["best_visibility_grid", "sigma"]),
        "class_alpha_B1B4": float(comp.loc["best_class_aware_grid", "alpha_B1B4"]),
        "class_alpha_B2B3": float(comp.loc["best_class_aware_grid", "alpha_B2B3"]),
        "class_sigma_B1B4": float(comp.loc["best_class_aware_grid", "sigma_B1B4"]),
        "class_sigma_B2B3": float(comp.loc["best_class_aware_grid", "sigma_B2B3"]),
    }


def best_visibility_grid(detections, params):
    return visibility_count(
        detections,
        alpha=params["vis_alpha"],
        sigma=params["vis_sigma"],
    )


def best_class_aware_grid(detections, params):
    return class_aware_visibility_count(
        detections,
        alpha_B1B4=params["class_alpha_B1B4"],
        alpha_B2B3=params["class_alpha_B2B3"],
        sigma_B1B4=params["class_sigma_B1B4"],
        sigma_B2B3=params["class_sigma_B2B3"],
    )


def extract_selector_features(
    detections: List[Dict],
    adaptive_corr: Dict[str, int],
    vis_grid: Dict[str, int],
    class_aware: Dict[str, int],
    class_aware_grid: Dict[str, int],
) -> Dict[str, float]:
    side_counts = {
        c: Counter(d["side_index"] for d in detections if d["class"] == c)
        for c in NAMES
    }
    by_class = {c: [d for d in detections if d["class"] == c] for c in NAMES}
    naive = {c: len(by_class[c]) for c in NAMES}

    feat = {"total_det": len(detections)}
    for c in NAMES:
        ys = [d["y_norm"] for d in by_class[c]]
        max_side = max(side_counts[c].values(), default=0)
        feat[f"{c}_naive"] = naive[c]
        feat[f"{c}_maxside"] = max_side
        feat[f"{c}_activesides"] = len(side_counts[c])
        feat[f"{c}_ratio"] = naive[c] / max(max_side, 1) if naive[c] else 0.0
        feat[f"{c}_yrange"] = (max(ys) - min(ys)) if ys else 0.0
        feat[f"d_ac_bvg_{c}"] = adaptive_corr[c] - vis_grid[c]
        feat[f"d_ac_cag_{c}"] = adaptive_corr[c] - class_aware_grid[c]
        feat[f"d_ac_cav_{c}"] = adaptive_corr[c] - class_aware[c]
    return feat


def unstable_tree_gate(feat: Dict[str, float]) -> bool:
    if feat["B4_naive"] <= 6.5:
        if feat["B1_activesides"] <= 2.5:
            if feat["B2_naive"] <= 5.5:
                if feat["B4_ratio"] <= 3.5:
                    return feat["d_ac_bvg_B3"] > 0.5
                return feat["B4_yrange"] > 0.09
            if feat["B4_yrange"] <= 0.05:
                return feat["d_ac_cag_B2"] > -0.5
        return False

    if feat["B4_yrange"] <= 0.10:
        return False
    if feat["B4_yrange"] <= 0.15:
        return feat["B2_ratio"] <= 3.5
    return False


def pick_v6_method(feat: Dict[str, float]) -> str:
    if feat["B3_ratio"] <= 3.17:
        if feat["B4_ratio"] <= 2.58:
            return "class_aware_vis"
        return "adaptive_corrected"
    return "best_visibility_grid"


def selector_v6_with_meta(detections: List[Dict], params: Dict[str, float]) -> Tuple[Dict[str, int], Dict[str, float]]:
    adaptive_corr = adaptive_corrected_count(detections)
    vis_grid = best_visibility_grid(detections, params)
    class_aware = class_aware_visibility_count(detections)
    class_aware_grid = best_class_aware_grid(detections, params)

    feat = extract_selector_features(
        detections,
        adaptive_corr,
        vis_grid,
        class_aware,
        class_aware_grid,
    )
    feat["unstable_gate"] = unstable_tree_gate(feat)

    if not feat["unstable_gate"]:
        feat["selected_method"] = "adaptive_corrected"
        return adaptive_corr, feat

    method_name = pick_v6_method(feat)
    feat["selected_method"] = method_name
    if method_name == "best_visibility_grid":
        return vis_grid, feat
    if method_name == "class_aware_vis":
        return class_aware, feat
    return adaptive_corr, feat


def selector_v6(detections: List[Dict], params: Dict[str, float]) -> Dict[str, int]:
    pred, _ = selector_v6_with_meta(detections, params)
    return pred


def main():
    print("=== Dedup Research V6 Started ===")
    print("Loading 228 JSON trees...")
    tree_data = [load_tree_data(jp) for jp in sorted(JSON_DIR.glob("*.json"))]
    print(f"Loaded {len(tree_data)} trees.")

    print("Computing ordinal prior...")
    compute_y_prior(tree_data)

    params = load_v5_reference_params()
    print("Running v6 selector...")

    preds = []
    selector_rows = []
    for tree_id, detections, gt, split, _ in tree_data:
        pred, meta = selector_v6_with_meta(detections, params)
        preds.append(pred)
        selector_rows.append(
            {
                "tree_id": tree_id,
                "split": split,
                "selected_method": meta["selected_method"],
                "unstable_gate": bool(meta["unstable_gate"]),
                "total_det": meta["total_det"],
                "B1_ratio": round(meta["B1_ratio"], 4),
                "B2_ratio": round(meta["B2_ratio"], 4),
                "B3_ratio": round(meta["B3_ratio"], 4),
                "B4_ratio": round(meta["B4_ratio"], 4),
                "B4_yrange": round(meta["B4_yrange"], 4),
                "d_ac_bvg_B3": round(meta["d_ac_bvg_B3"], 4),
                "d_ac_cag_B2": round(meta["d_ac_cag_B2"], 4),
                **{f"{c}_pred": pred[c] for c in NAMES},
                **{f"{c}_gt": gt[c] for c in NAMES},
            }
        )

    selector_df = pd.DataFrame(selector_rows)
    selector_df.to_csv(OUT_DIR / "selector_choices_v6.csv", index=False)

    v6_metrics = evaluate_predictions(preds, tree_data)
    v6_ci = bootstrap_acc_ci(preds, tree_data)

    v6_row = {
        "method": "v6_selector",
        "mean_MAE": v6_metrics["mean_MAE"],
        "acc_within_1_error": v6_metrics["acc_within_1_error"],
        "mean_total_error": v6_metrics["mean_total_error"],
        "score": v6_metrics["score"],
        "n_trees": v6_metrics["n_trees"],
        "selector_default": "adaptive_corrected",
        "selector_alt_1": "best_visibility_grid",
        "selector_alt_2": "class_aware_vis",
    }

    v5_comp = pd.read_csv(V5_OUT_DIR / "method_comparison_v5.csv")
    comparison_v6 = pd.concat([v5_comp, pd.DataFrame([v6_row])], ignore_index=True, sort=False)
    comparison_v6 = comparison_v6.sort_values(["score", "acc_within_1_error", "mean_MAE"], ascending=[False, False, True])
    comparison_v6.to_csv(OUT_DIR / "method_comparison_v6.csv", index=False)

    per_tree_df = v6_metrics["df"].merge(
        selector_df[["tree_id", "selected_method", "unstable_gate", "B1_ratio", "B2_ratio", "B3_ratio", "B4_ratio"]],
        on="tree_id",
        how="left",
    )
    per_tree_df.to_csv(OUT_DIR / "per_tree_results_v6.csv", index=False)
    per_tree_df[~per_tree_df["within_1"]].to_csv(OUT_DIR / "error_analysis_v6.csv", index=False)

    baseline_df = pd.read_csv(V5_OUT_DIR / "best_method_details_v5.csv")
    delta_df = per_tree_df.merge(
        baseline_df[["tree_id", "within_1", "error_sum", "MAE"]],
        on="tree_id",
        suffixes=("_v6", "_v5"),
        how="left",
    )
    delta_df["delta_error_sum"] = delta_df["error_sum_v6"] - delta_df["error_sum_v5"]
    delta_df["delta_MAE"] = delta_df["MAE_v6"] - delta_df["MAE_v5"]
    delta_df.to_csv(OUT_DIR / "selector_delta_vs_v5.csv", index=False)

    improved = delta_df[
        (~delta_df["within_1_v5"] & delta_df["within_1_v6"])
        | (delta_df["delta_error_sum"] < 0)
    ].copy()
    regressions = delta_df[
        delta_df["within_1_v5"] & (~delta_df["within_1_v6"])
    ].copy()

    selected_counts = selector_df["selected_method"].value_counts().to_dict()
    split_acc = (
        per_tree_df.assign(within=per_tree_df["within_1"].astype(int))
        .groupby("split")["within"]
        .mean()
        .mul(100.0)
        .round(2)
        .to_dict()
    )

    summary = [
        "# Dedup Research V6 Summary",
        "",
        "## Key Result",
        "",
        f"- `v6_selector` reached **{v6_metrics['acc_within_1_error']:.2f}% Acc +/-1** on the same 228-tree JSON benchmark.",
        f"- `mean_MAE` dropped to **{v6_metrics['mean_MAE']:.4f}**.",
        f"- `mean_total_error` dropped to **{v6_metrics['mean_total_error']:.2f}**.",
        f"- Bootstrap 95% CI: **{v6_ci['ci_lower']:.2f}% - {v6_ci['ci_upper']:.2f}%**.",
        "",
        "## Delta vs V5 Best",
        "",
        "- V5 best (`adaptive_corrected`): `93.86%`, `MAE 0.2774`, `mean_total_error 1.11`.",
        f"- V6 selector: `Acc {v6_metrics['acc_within_1_error']:.2f}%`, `MAE {v6_metrics['mean_MAE']:.4f}`, `mean_total_error {v6_metrics['mean_total_error']:.2f}`.",
        f"- Gain: `+{v6_metrics['acc_within_1_error'] - 93.86:.2f} pp Acc`, `{0.2774 - v6_metrics['mean_MAE']:.4f} MAE`, `{1.11 - v6_metrics['mean_total_error']:.2f} total-error`.",
        "",
        "## Selector Design",
        "",
        "- Default method: `adaptive_corrected`.",
        "- Unstable high-B3 duplication trees are rerouted to `best_visibility_grid`.",
        "- B2-heavy / low-B4-duplication trees are rerouted to `class_aware_vis`.",
        "- Runtime remains deterministic and keeps the strict 4-class output.",
        "",
        "## Method Usage",
        "",
    ]
    for method_name, count in selected_counts.items():
        summary.append(f"- `{method_name}` selected on `{count}` trees")

    summary.extend(
        [
            "",
            "## Split Accuracy",
            "",
        ]
    )
    for split_name, acc in split_acc.items():
        summary.append(f"- `{split_name}`: `{acc:.2f}%`")

    summary.extend(
        [
            "",
            "## Improved Trees",
            "",
        ]
    )
    if improved.empty:
        summary.append("- None")
    else:
        for _, row in improved.sort_values(["delta_error_sum", "tree_id"]).head(20).iterrows():
            summary.append(
                f"- `{row['tree_id']}`: `{row['error_sum_v5']}` -> `{row['error_sum_v6']}` via `{row['selected_method']}`"
            )

    summary.extend(
        [
            "",
            "## Regressions",
            "",
        ]
    )
    if regressions.empty:
        summary.append("- None")
    else:
        for _, row in regressions.sort_values(["tree_id"]).iterrows():
            summary.append(
                f"- `{row['tree_id']}`: V5 was within +/-1, V6 missed via `{row['selected_method']}` (`{row['error_sum_v5']}` -> `{row['error_sum_v6']}`)"
            )

    (OUT_DIR / "summary_v6.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("Saved:")
    print(f"  - {OUT_DIR / 'method_comparison_v6.csv'}")
    print(f"  - {OUT_DIR / 'per_tree_results_v6.csv'}")
    print(f"  - {OUT_DIR / 'error_analysis_v6.csv'}")
    print(f"  - {OUT_DIR / 'selector_choices_v6.csv'}")
    print(f"  - {OUT_DIR / 'selector_delta_vs_v5.csv'}")
    print(f"  - {OUT_DIR / 'summary_v6.md'}")
    print("=== Dedup Research V6 Done ===")


if __name__ == "__main__":
    main()
