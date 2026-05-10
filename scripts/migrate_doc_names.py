"""One-shot migration: rewrite old method names in markdown docs.

Same NAME_MAP as scripts/migrate_method_names.py, but applies to .md docs.
Only word-boundary matches; running again is a no-op.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

NAME_MAP = {
    "selector_with_b2b3":       "M01_selector_b2b3",
    "selector_iter9_trifurc":   "M02_selector_trifurc",
    "geometric_mean_blend":     "M03_blend_geometric",
    "floor_clamped_hybrid":     "M04_blend_floor_clamped",
    "hybrid_vis_corr":          "M05_blend_vis_divide",
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
    "v1_corrected":             "M15_divide_global",
    "corrected_naive":          "M15_divide_global",
    "v8_b2_b4_boosted":         "M16_boost_b2b4",
    "b2_b4_boosted":            "M16_boost_b2b4",
    "v6_selector":              "M17_selector_regime",
    "v8_entropy_stacking":      "M18_entropy_stack",
    "v5_adaptive_corrected":    "M19_divide_adaptive",
    "adaptive_corrected":       "M19_divide_adaptive",
    "v5_best_visibility":       "M20_weight_visibility_grid",
    "best_visibility_grid":     "M20_weight_visibility_grid",
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
}

ORDERED = sorted(NAME_MAP.items(), key=lambda kv: -len(kv[0]))
PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])(" + "|".join(re.escape(k) for k, _ in ORDERED) + r")(?![A-Za-z0-9_])"
)


def remap(s: str) -> str:
    return PATTERN.sub(lambda m: NAME_MAP[m.group(1)], s)


# Files to migrate. Skip historical reports + archive — keep their language.
TARGETS = [
    "CLAUDE.md",
    "AGENTS.md",
    "README.md",
    "RESEARCH.md",
    "report_10Mei2026.md",
    "report_05Mei2026.md",
]


def main():
    for rel in TARGETS:
        p = REPO / rel
        if not p.exists():
            print(f"  SKIP {rel}: missing")
            continue
        old = p.read_text(encoding="utf-8")
        new = remap(old)
        if new != old:
            p.write_text(new, encoding="utf-8")
            n = sum(1 for _ in PATTERN.finditer(old))
            print(f"  OK   {rel} ({n} replacements)")
        else:
            print(f"  noop {rel}")
    print("Done.")


if __name__ == "__main__":
    main()
