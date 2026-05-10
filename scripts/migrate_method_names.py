"""One-shot migration: rewrite old method names in CSV files to new Mxx_* names.

Run once after the 2026-05-10 rename. Idempotent: running it again is a no-op
because there are no old names left to replace.

Affects:
- column headers like 'naive_B1', 'v9_selector_total' → 'M29_baseline_naive_sum_B1'
- 'method' column values like 'v9_selector' → 'M12_selector_overrides'
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
REPORTS = REPO / "reports"

# old → new mapping (canonical 29 methods, 2026-05-10)
NAME_MAP = {
    "selector_with_b2b3":       "M01_selector_b2b3",
    "selector_iter9_trifurc":   "M02_selector_trifurc",
    "geometric_mean_blend":     "M03_blend_geometric",
    "floor_clamped_hybrid":     "M04_blend_floor_clamped",
    "hybrid_vis_corr":          "M05_blend_vis_divide",
    "visibility":               "M06_weight_visibility",
    "v2_visibility":            "M06_weight_visibility",
    "visibility_count":         "M06_weight_visibility",
    "side_coverage":            "M07_weight_coverage",
    "density_scaled_vis":       "M08_divide_density_vis",
    "v9_median_strong5":        "M09_median_strong5",
    "median_strong5":           "M09_median_strong5",
    "v8_entropy_modulated":     "M10_entropy_divide",
    "entropy_modulated":        "M10_entropy_divide",
    "v9_b2_median_v6":          "M11_median_b2",
    "b2_median_v6":             "M11_median_b2",
    "v9_selector":              "M12_selector_overrides",
    "v7_stacking_bracketed":    "M13_stack_bracket",
    "stacking_bracketed":       "M13_stack_bracket",
    "v7_stacking_density":      "M14_stack_density",
    "stacking_density":         "M14_stack_density",
    "corrected":                "M15_divide_global",
    "v1_corrected":             "M15_divide_global",
    "corrected_naive":          "M15_divide_global",
    "v8_b2_b4_boosted":         "M16_boost_b2b4",
    "b2_b4_boosted":            "M16_boost_b2b4",
    "v6_selector":              "M17_selector_regime",
    "v8_entropy_stacking":      "M18_entropy_stack",
    "adaptive_corrected":       "M19_divide_adaptive",
    "v5_adaptive_corrected":    "M19_divide_adaptive",
    "best_visibility_grid":     "M20_weight_visibility_grid",
    "v5_best_visibility":       "M20_weight_visibility_grid",
    "v7_ordinal_b3":            "M21_ordinal_b3",
    "ordinal_b3":               "M21_ordinal_b3",
    "v8_floor_anchor_50":       "M22_anchor_floor50",
    "floor_anchor_50":          "M22_anchor_floor50",
    "v8_side_agreement":        "M23_agree_side",
    "side_agreement":           "M23_agree_side",
    "class_aware_vis":          "M24_weight_class_aware",
    "v8_multi_consensus":       "M25_consensus_multi",
    "multi_consensus":          "M25_consensus_multi",
    "v8_per_side_median":       "M26_median_per_side",
    "per_side_median":          "M26_median_per_side",
    "adaptive_visibility":      "M27_weight_visibility_adaptive",
    "relaxed_match":            "M28_baseline_match_strict",
    "naive":                    "M29_baseline_naive_sum",
}

# longest-first so 'v9_b2_median_v6' replaces before 'v9_selector' etc.
ORDERED = sorted(NAME_MAP.items(), key=lambda kv: -len(kv[0]))


import re

# Build a single alternation regex (longest first) so we replace each occurrence
# at most once and longer names take precedence over shorter ones.
_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(" + "|".join(re.escape(k) for k, _ in ORDERED) + r")(?![A-Za-z0-9])"
)


def remap_token(s: str) -> str:
    """Replace any old name token; word-boundary semantics so 'visibility'
    won't match inside 'M20_weight_visibility_grid' after a prior rewrite."""
    return _PATTERN.sub(lambda m: NAME_MAP[m.group(1)], s)


def remap_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [remap_token(c) for c in df.columns]
    return df


def remap_method_values(df: pd.DataFrame) -> pd.DataFrame:
    if "method" in df.columns:
        df["method"] = df["method"].astype(str).map(remap_token)
    return df


def migrate_csv(p: Path) -> None:
    try:
        df = pd.read_csv(p)
    except Exception as e:
        print(f"  SKIP {p.relative_to(REPO)}: {e}")
        return
    df = remap_columns(df)
    df = remap_method_values(df)
    df.to_csv(p, index=False)
    print(f"  OK   {p.relative_to(REPO)}")


def main():
    csvs = sorted(REPORTS.rglob("*.csv"))
    print(f"Found {len(csvs)} CSVs under {REPORTS}.")
    for p in csvs:
        # skip the archive snapshot — preserve historical names
        if "reports_pre_rename" in p.parts:
            continue
        migrate_csv(p)
    print("Done.")


if __name__ == "__main__":
    main()
