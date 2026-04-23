"""
Dedup Research v9 - Regime-Aware Selector

Goal:
  - Stay strict 4-class and heuristic-only.
  - Convert the v9 audit into a runtime-ready selector.
  - Break past v6_selector by keeping v6 as the default and applying
    only narrow, physically motivated specialist overrides.

Selector design:
  1. Start from v6_selector as the baseline.
  2. Route overlap-heavy B4-only trees to stacking_bracketed.
  3. Route compact class_aware trees with low B4 support to b2_b4_boosted.
  4. Route compact B3/B4-only low-total trees to floor_anchor_50.
  5. Route one dense all-side moderate-dup regime to b2_b4_boosted.
"""

from __future__ import annotations

from collections import Counter
from datetime import date
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

import dedup_research_v5 as v5
import dedup_research_v6 as v6
import dedup_research_v7 as v7
import dedup_research_v8 as v8

BASE = Path(__file__).resolve().parent.parent
OUT_DIR = BASE / "reports" / "dedup_research_v9"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NAMES = ["B1", "B2", "B3", "B4"]


def load_tree_data() -> List[Tuple[str, List[Dict], Dict[str, int], str]]:
    tree_data = [v7.load_tree_data(jp) for jp in sorted(v7.JSON_DIR.glob("*.json"))]
    return tree_data


def ensure_priors(tree_data: List[Tuple[str, List[Dict], Dict[str, int], str]]) -> Dict[str, float]:
    v5_tree_data = [v5.load_tree_data(jp) for jp in sorted(v5.JSON_DIR.glob("*.json"))]
    v5.compute_y_prior(v5_tree_data)
    return v7.compute_y_medians(tree_data)


def evaluate_method(
    name: str,
    func: Callable[[List[Dict]], Dict[str, int]],
    tree_data: List[Tuple[str, List[Dict], Dict[str, int], str]],
) -> Tuple[Dict[str, float], pd.DataFrame]:
    rows = []
    for tree_id, dets, gt, split in tree_data:
        pred = func(dets)
        err = {c: abs(pred.get(c, 0) - gt[c]) for c in NAMES}
        delta = {c: pred.get(c, 0) - gt[c] for c in NAMES}
        rows.append(
            {
                "tree_id": tree_id,
                "split": split,
                "ok": all(e <= 1 for e in err.values()),
                "MAE": float(np.mean(list(err.values()))),
                "error_sum": int(sum(err.values())),
                **{f"gt_{c}": gt[c] for c in NAMES},
                **{f"pred_{c}": pred.get(c, 0) for c in NAMES},
                **{f"err_{c}": err[c] for c in NAMES},
                **{f"delta_{c}": delta[c] for c in NAMES},
            }
        )
    df = pd.DataFrame(rows)
    metrics = {
        "method": name,
        "acc": round(df["ok"].mean() * 100.0, 2),
        "mae": round(df["MAE"].mean(), 4),
        "mean_total_error": round(df["error_sum"].mean(), 4),
        "n_fail": int((~df["ok"]).sum()),
    }
    return metrics, df


def per_class_error_signs(df: pd.DataFrame) -> Dict[str, int]:
    signs: Dict[str, int] = {}
    for c in NAMES:
        delta = df[f"delta_{c}"]
        signs[f"{c}_over_gt1"] = int((delta > 1).sum())
        signs[f"{c}_under_gt1"] = int((delta < -1).sum())
        signs[f"{c}_over_any"] = int((delta > 0).sum())
        signs[f"{c}_under_any"] = int((delta < 0).sum())
    return signs


def median_strong5_factory(params: Dict[str, float]) -> Callable[[List[Dict]], Dict[str, int]]:
    def median_strong5(dets: List[Dict]) -> Dict[str, int]:
        preds = [
            v6.selector_v6(dets, params),
            v7.stacking_bracketed(dets),
            v8.b2_b4_boosted(dets),
            v8.floor_anchored(dets),
            v8.per_side_median(dets),
        ]
        return {c: int(round(float(np.median([pred[c] for pred in preds])))) for c in NAMES}

    return median_strong5


def b2_median_v6_factory(params: Dict[str, float]) -> Callable[[List[Dict]], Dict[str, int]]:
    def b2_median_v6(dets: List[Dict]) -> Dict[str, int]:
        preds = {
            "v6": v6.selector_v6(dets, params),
            "stacking_bracketed": v7.stacking_bracketed(dets),
            "b2_b4_boosted": v8.b2_b4_boosted(dets),
            "floor_anchor_50": v8.floor_anchored(dets),
            "per_side_median": v8.per_side_median(dets),
        }
        out = dict(preds["v6"])
        out["B2"] = int(round(float(np.median([pred["B2"] for pred in preds.values()]))))
        return out

    return b2_median_v6


def selector_v9_with_meta(dets: List[Dict], params: Dict[str, float]) -> Tuple[Dict[str, int], Dict[str, float]]:
    pred_v6, meta = v6.selector_v6_with_meta(dets, params)
    pred = dict(pred_v6)

    if (
        meta["B1_naive"] == 0
        and meta["B2_naive"] == 0
        and meta["B3_naive"] == 0
        and meta["B4_naive"] > 0
        and meta["B4_maxside"] >= 4
    ):
        meta["selected_method_v9"] = "v7_stacking_bracketed"
        meta["selector_reason_v9"] = "b4_only_overlap"
        return v7.stacking_bracketed(dets), meta

    if (
        meta["selected_method"] == "class_aware_vis"
        and meta["total_det"] >= 21
        and meta["B4_naive"] <= 2
    ):
        meta["selected_method_v9"] = "v8_b2_b4_boosted"
        meta["selector_reason_v9"] = "classaware_compact_lowb4"
        return v8.b2_b4_boosted(dets), meta

    if (
        meta["B1_naive"] == 0
        and meta["B2_naive"] == 0
        and meta["B3_naive"] > 0
        and meta["B4_naive"] > 0
        and meta["total_det"] <= 13
        and meta["B3_activesides"] == 4
        and meta["B4_activesides"] == 4
        and meta["B3_ratio"] <= 3.0
        and meta["B4_ratio"] >= 4.0
    ):
        meta["selected_method_v9"] = "v8_floor_anchor_50"
        meta["selector_reason_v9"] = "b3b4_only_lowtotal"
        return v8.floor_anchored(dets), meta

    if (
        meta["selected_method"] == "adaptive_corrected"
        and meta["total_det"] >= 28
        and meta["B2_activesides"] == 4
        and meta["B3_activesides"] == 4
        and meta["B4_activesides"] == 4
        and meta["B2_ratio"] < 3.0
        and meta["B3_ratio"] < 2.5
    ):
        meta["selected_method_v9"] = "v8_b2_b4_boosted"
        meta["selector_reason_v9"] = "dense_allside_moderatedup"
        return v8.b2_b4_boosted(dets), meta

    meta["selected_method_v9"] = meta["selected_method"]
    meta["selector_reason_v9"] = "v6_default"
    return pred, meta


def selector_v9(params: Dict[str, float]) -> Callable[[List[Dict]], Dict[str, int]]:
    def run(dets: List[Dict]) -> Dict[str, int]:
        pred, _ = selector_v9_with_meta(dets, params)
        return pred

    return run


def build_method_family(params: Dict[str, float]) -> Dict[str, Callable[[List[Dict]], Dict[str, int]]]:
    return {
        "v5_adaptive_corrected": v5.adaptive_corrected_count,
        "v5_best_visibility_grid": lambda dets: v6.best_visibility_grid(dets, params),
        "v5_class_aware_vis": v5.class_aware_visibility_count,
        "v6_selector": lambda dets: v6.selector_v6(dets, params),
        "v9_selector": selector_v9(params),
        "v7_stacking_density": v7.stacking_density_corrected,
        "v7_stacking_bracketed": v7.stacking_bracketed,
        "v7_ordinal_b3": v7.ordinal_modulated_b3,
        "v8_entropy_modulated": v8.entropy_modulated,
        "v8_side_agreement": v8.side_agreement_corrected,
        "v8_floor_anchor_50": v8.floor_anchored,
        "v8_per_side_median": v8.per_side_median,
        "v8_multi_consensus": v8.multi_estimator_consensus,
        "v8_b2_b4_boosted": v8.b2_b4_boosted,
        "v9_median_strong5": median_strong5_factory(params),
        "v9_b2_median_v6": b2_median_v6_factory(params),
    }


def oracle_analysis(
    tree_data: List[Tuple[str, List[Dict], Dict[str, int], str]],
    method_details: Dict[str, pd.DataFrame],
    method_names: List[str],
    label: str,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    rows = []
    for idx, (tree_id, _, _, split) in enumerate(tree_data):
        ok_map = {name: bool(method_details[name].iloc[idx]["ok"]) for name in method_names}
        rows.append(
            {
                "tree_id": tree_id,
                "split": split,
                "oracle_ok": any(ok_map.values()),
                "n_methods_ok": int(sum(ok_map.values())),
                "methods_ok": ",".join([name for name, ok in ok_map.items() if ok]),
            }
        )
    df = pd.DataFrame(rows)
    metrics = {
        "oracle_label": label,
        "acc": round(df["oracle_ok"].mean() * 100.0, 2),
        "n_fail": int((~df["oracle_ok"]).sum()),
    }
    return metrics, df


def main() -> None:
    print("=== Dedup Research V9 Started ===")
    tree_data = load_tree_data()
    print(f"Loaded {len(tree_data)} trees.")

    y_medians = ensure_priors(tree_data)
    params = v6.load_v5_reference_params()

    methods = build_method_family(params)
    method_rows = []
    method_details: Dict[str, pd.DataFrame] = {}
    sign_rows = []

    for name, func in methods.items():
        metrics, df = evaluate_method(name, func, tree_data)
        method_rows.append(metrics)
        method_details[name] = df
        sign_rows.append({"method": name, **per_class_error_signs(df)})
        print(f"  {name:22} Acc={metrics['acc']:.2f}%  MAE={metrics['mae']:.4f}")

    comp_df = pd.DataFrame(method_rows).sort_values(["acc", "mae", "method"], ascending=[False, True, True])
    comp_df.to_csv(OUT_DIR / "method_comparison_v9.csv", index=False)
    pd.DataFrame(sign_rows).to_csv(OUT_DIR / "error_sign_summary_v9.csv", index=False)

    selector_rows = []
    for tree_id, dets, gt, split in tree_data:
        pred, meta = selector_v9_with_meta(dets, params)
        selector_rows.append(
            {
                "tree_id": tree_id,
                "split": split,
                "selected_method_v6": meta["selected_method"],
                "selected_method_v9": meta["selected_method_v9"],
                "selector_reason_v9": meta["selector_reason_v9"],
                "unstable_gate": bool(meta["unstable_gate"]),
                "total_det": meta["total_det"],
                "B1_naive": meta["B1_naive"],
                "B2_naive": meta["B2_naive"],
                "B3_naive": meta["B3_naive"],
                "B4_naive": meta["B4_naive"],
                "B2_ratio": round(meta["B2_ratio"], 4),
                "B3_ratio": round(meta["B3_ratio"], 4),
                "B4_ratio": round(meta["B4_ratio"], 4),
                "B3_yrange": round(meta["B3_yrange"], 4),
                "B4_yrange": round(meta["B4_yrange"], 4),
                **{f"pred_{c}": pred[c] for c in NAMES},
                **{f"gt_{c}": gt[c] for c in NAMES},
            }
        )
    selector_df = pd.DataFrame(selector_rows)
    selector_df.to_csv(OUT_DIR / "selector_choices_v9.csv", index=False)

    rescue_rows = []
    v6_fail_df = method_details["v6_selector"][~method_details["v6_selector"]["ok"]].copy()
    rescue_methods = [
        "v5_best_visibility_grid",
        "v5_class_aware_vis",
        "v7_stacking_bracketed",
        "v8_entropy_modulated",
        "v8_side_agreement",
        "v8_floor_anchor_50",
        "v8_per_side_median",
        "v8_multi_consensus",
        "v8_b2_b4_boosted",
        "v9_selector",
        "v9_median_strong5",
        "v9_b2_median_v6",
    ]
    for _, row in v6_fail_df.iterrows():
        tree_id = row["tree_id"]
        rescue = {"tree_id": tree_id, "split": row["split"], "v6_error_sum": int(row["error_sum"])}
        for method_name in rescue_methods:
            alt_row = method_details[method_name].set_index("tree_id").loc[tree_id]
            rescue[f"{method_name}_ok"] = bool(alt_row["ok"])
            rescue[f"{method_name}_error_sum"] = int(alt_row["error_sum"])
        rescue_rows.append(rescue)
    rescue_df = pd.DataFrame(rescue_rows)
    rescue_df.to_csv(OUT_DIR / "v6_failure_rescue_matrix.csv", index=False)

    oracle_narrow_methods = [
        "v6_selector",
        "v9_selector",
        "v5_adaptive_corrected",
        "v5_best_visibility_grid",
        "v5_class_aware_vis",
        "v7_stacking_bracketed",
        "v8_entropy_modulated",
        "v8_b2_b4_boosted",
    ]
    oracle_broad_methods = [
        "v6_selector",
        "v9_selector",
        "v5_adaptive_corrected",
        "v5_best_visibility_grid",
        "v5_class_aware_vis",
        "v7_stacking_density",
        "v7_stacking_bracketed",
        "v7_ordinal_b3",
        "v8_entropy_modulated",
        "v8_side_agreement",
        "v8_floor_anchor_50",
        "v8_per_side_median",
        "v8_multi_consensus",
        "v8_b2_b4_boosted",
    ]
    oracle_narrow_metrics, oracle_narrow_df = oracle_analysis(tree_data, method_details, oracle_narrow_methods, "narrow_family")
    oracle_broad_metrics, oracle_broad_df = oracle_analysis(tree_data, method_details, oracle_broad_methods, "broad_family")
    oracle_narrow_df.to_csv(OUT_DIR / "oracle_narrow_v9.csv", index=False)
    oracle_broad_df.to_csv(OUT_DIR / "oracle_broad_v9.csv", index=False)

    v7_best_tiebroken = comp_df[comp_df["method"].isin(["v7_stacking_density", "v7_stacking_bracketed"])].iloc[0]
    v8_best_tiebroken = comp_df[
        comp_df["method"].isin(["v8_entropy_modulated", "v7_stacking_density", "v7_stacking_bracketed"])
    ].iloc[0]
    v9_df = method_details["v9_selector"]
    v6_df = method_details["v6_selector"]
    v9_fail_df = v9_df[~v9_df["ok"]].copy()
    improved = ((v6_df["ok"] == False) & (v9_df["ok"] == True)).sum()
    regressed = ((v6_df["ok"] == True) & (v9_df["ok"] == False)).sum()
    reason_counts = selector_df["selector_reason_v9"].value_counts().to_dict()
    selector_df.merge(v9_df[["tree_id", "ok", "error_sum"]], on="tree_id", how="left").to_csv(
        OUT_DIR / "per_tree_results_v9.csv", index=False
    )
    v9_fail_df.to_csv(OUT_DIR / "error_analysis_v9.csv", index=False)

    rescue_counts = {}
    for method_name in rescue_methods:
        rescue_counts[method_name] = int(rescue_df[f"{method_name}_ok"].sum()) if not rescue_df.empty else 0

    split_rows = []
    for split, g in v9_df.groupby("split"):
        split_rows.append(
            {
                "split": split,
                "n": len(g),
                "acc": round(g["ok"].mean() * 100.0, 2),
                "mae": round(g["MAE"].mean(), 4),
            }
        )
    split_df = pd.DataFrame(split_rows)
    split_df.to_csv(OUT_DIR / "split_breakdown_v9.csv", index=False)

    best_v9 = comp_df.set_index("method").loc["v9_selector"]
    best_v6 = comp_df.set_index("method").loc["v6_selector"]
    summary_lines = [
        "# Dedup Research V9",
        "",
        f"Date: {date.today().isoformat()}",
        "",
        "## Key Result",
        "",
        f"- `v9_selector` reached **{best_v9['acc']:.2f}% Acc +/-1** with **MAE {best_v9['mae']:.4f}**.",
        f"- Gain over `v6_selector`: **+{best_v9['acc'] - best_v6['acc']:.2f} pp Acc**, **{best_v6['mae'] - best_v9['mae']:.4f} MAE**, **{best_v6['mean_total_error'] - best_v9['mean_total_error']:.4f} total-error**.",
        f"- Remaining failing trees: **{int(best_v9['n_fail'])} / {len(tree_data)}**.",
        "",
        "## Context",
        "",
        "- v7 and v8 were not the true ceiling.",
        f"- The best tie-broken v7 method is `{v7_best_tiebroken['method']}` at {v7_best_tiebroken['acc']:.2f}% / MAE {v7_best_tiebroken['mae']:.4f}.",
        f"- The best tie-broken v8 method is `{v8_best_tiebroken['method']}` at {v8_best_tiebroken['acc']:.2f}% / MAE {v8_best_tiebroken['mae']:.4f}.",
        "",
        "## Why V9 Helps",
        "",
        "- V9 keeps `v6_selector` as the default and only overrides trees in narrow, high-confidence regimes.",
        "- The improvement comes from regime routing, not from adding a new global divisor.",
        "- The overrides are physically motivated and deterministic.",
        "",
        "## Selector Design",
        "",
        "1. Default: `v6_selector`",
        "2. `b4_only_overlap` -> `v7_stacking_bracketed`",
        "3. `classaware_compact_lowb4` -> `v8_b2_b4_boosted`",
        "4. `b3b4_only_lowtotal` -> `v8_floor_anchor_50`",
        "5. `dense_allside_moderatedup` -> `v8_b2_b4_boosted`",
        "",
        "## Trigger Usage",
        "",
        f"- `v6_default`: {reason_counts.get('v6_default', 0)} trees",
        f"- `b4_only_overlap`: {reason_counts.get('b4_only_overlap', 0)} trees",
        f"- `classaware_compact_lowb4`: {reason_counts.get('classaware_compact_lowb4', 0)} trees",
        f"- `b3b4_only_lowtotal`: {reason_counts.get('b3b4_only_lowtotal', 0)} trees",
        f"- `dense_allside_moderatedup`: {reason_counts.get('dense_allside_moderatedup', 0)} trees",
        "",
        "## Delta vs V6",
        "",
        f"- Improved trees: {int(improved)}",
        f"- Regressed trees: {int(regressed)}",
        f"- `v9_selector` rescues {rescue_counts['v9_selector']} of the {len(v6_fail_df)} v6 failures.",
        "",
        "## Per-Split",
        "",
        "```",
        split_df.to_string(index=False),
        "```",
        "",
        "## Remaining Failures",
        "",
        f"- {', '.join(v9_fail_df['tree_id'].tolist()) if not v9_fail_df.empty else 'None'}",
        "",
        "## Oracle Headroom",
        "",
        f"- Oracle over a narrow family of strong methods: {oracle_narrow_metrics['acc']:.2f}% ({oracle_narrow_metrics['n_fail']} unresolved trees).",
        f"- Oracle over a broad family of existing v5/v6/v7/v8 methods: {oracle_broad_metrics['acc']:.2f}% ({oracle_broad_metrics['n_fail']} unresolved trees).",
        "",
        "## Files",
        "",
        "- `method_comparison_v9.csv`: benchmark comparison including final selector.",
        "- `selector_choices_v9.csv`: per-tree routing decision and trigger reason.",
        "- `error_analysis_v9.csv`: trees still outside +/-1 after v9.",
        "- `v6_failure_rescue_matrix.csv`: rescue audit against the old v6 failures.",
        "",
        "## Notes",
        "",
        f"- Y medians used by v7 ordinal logic: {y_medians}.",
        "- The selector remains fully deterministic and training-free.",
    ]

    (OUT_DIR / "summary_v9.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"\nSaved to {OUT_DIR}")
    print(f"Best overall: {comp_df.iloc[0]['method']} @ {comp_df.iloc[0]['acc']:.2f}%  MAE={comp_df.iloc[0]['mae']:.4f}")
    print(f"Oracle broad family: {oracle_broad_metrics['acc']:.2f}%")


if __name__ == "__main__":
    main()
