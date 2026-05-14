# algorithms/ — dedup method registry
#
# Naming convention: M<NN>_<family>_<descriptor>
# See ../NAMING.md for full glossary, mapping table, and stability rule.
#
# Each module exports predict(detections, [params]) -> dict[str, int].
# All methods deterministic, no training, no embeddings, no gradients.
#
# Mxx IDs are STABLE — assigned once on 2026-05-10, never re-shuffled.
# Initial assignment ranking-based on 953-tree benchmark, tie-break by MAE.
# Numeric order is NOT an ongoing ranking.
#
# Current top on Brand-New-Dataset-YOLO 953 trees (canonical report):
#
#   ID   Name                          Acc+/-1   MAE     n_fail
#   M02  M02_selector_trifurc          86.88%    0.3880  125     highest Acc+/-1
#   M01  M01_selector_b2b3             86.78%    0.3875  126     production-compatible
#   M03  M03_blend_geometric           86.36%    0.3877  130
#   M04  M04_blend_floor_clamped       86.25%    0.3964  131
#   M05  M05_blend_vis_divide          86.25%    0.3990  131     simple fallback
#
# Recommended:
#   - highest Acc+/-1 (full 953 dataset) -> M02_selector_trifurc
#   - production-compatible default      -> M01_selector_b2b3
#   - simple fallback                    -> M05_blend_vis_divide
#   - reference floor                    -> M29_baseline_naive_sum

METHOD_IDS = [
    "M01_selector_b2b3",
    "M02_selector_trifurc",
    "M03_blend_geometric",
    "M04_blend_floor_clamped",
    "M05_blend_vis_divide",
    "M06_weight_visibility",
    "M07_weight_coverage",
    "M08_divide_density_vis",
    "M09_median_strong5",
    "M10_entropy_divide",
    "M11_median_b2",
    "M12_selector_overrides",
    "M13_stack_bracket",
    "M14_stack_density",
    "M15_divide_global",
    "M16_boost_b2b4",
    "M17_selector_regime",
    "M18_entropy_stack",
    "M19_divide_adaptive",
    "M20_weight_visibility_grid",
    "M21_ordinal_b3",
    "M22_anchor_floor50",
    "M23_agree_side",
    "M24_weight_class_aware",
    "M25_consensus_multi",
    "M26_median_per_side",
    "M27_weight_visibility_adaptive",
    "M28_baseline_match_strict",
    "M29_baseline_naive_sum",
]
