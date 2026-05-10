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
# Ranking on Brand-New-Dataset-YOLO 953 trees (canonical, 2026-05-10):
#
#   ID   Name                          Acc±1     MAE     n_fail
#   M01  M01_selector_b2b3             86.67%   0.3982   127     ← production
#   M02  M02_selector_trifurc          86.67%   0.3987   127
#   M03  M03_blend_geometric           86.15%   0.3961   132
#   M04  M04_blend_floor_clamped       86.04%   0.4050   133
#   M05  M05_blend_vis_divide          86.04%   0.4077   133     ← simple fallback
#   M06  M06_weight_visibility         85.94%   0.3960   134
#   M07  M07_weight_coverage           85.94%   0.3930   134
#   M08  M08_divide_density_vis        85.94%   0.4020   134
#   M09  M09_median_strong5            85.73%   0.4010   136
#   M10  M10_entropy_divide            84.78%   0.4510   145
#   M11  M11_median_b2                 84.78%   0.4290   145
#   M12  M12_selector_overrides        84.68%   0.4410   146     ⚠ overfits 228 dev
#   M13  M13_stack_bracket             84.58%   0.4280   147
#   M14  M14_stack_density             84.58%   —        —
#   M15  M15_divide_global             84.37%   0.4160   149
#   M16  M16_boost_b2b4                84.37%   —        —
#   M17  M17_selector_regime           84.26%   0.4440   150
#   M18  M18_entropy_stack             —        —        —
#   M19  M19_divide_adaptive           82.58%   0.4600   166
#   M20  M20_weight_visibility_grid    80.80%   0.4600   183
#   M21  M21_ordinal_b3                low      —        —       ← broken on 953
#   M22  M22_anchor_floor50            —        —        —       ← specialist
#   M23  M23_agree_side                —        —        —
#   M24  M24_weight_class_aware        70.93%   0.5460   277
#   M25  M25_consensus_multi           18.86%   —        —       ← extreme undercount
#   M26  M26_median_per_side           18.86%   —        —       ← extreme undercount
#   M27  M27_weight_visibility_adaptive —       —        —
#   M28  M28_baseline_match_strict      5.98%   1.8110   896     ← reference floor
#   M29  M29_baseline_naive_sum         3.99%   2.2800   915     ← reference floor
#
# Recommended:
#   - production (full 953 dataset) → M01_selector_b2b3
#   - simple fallback                → M05_blend_vis_divide
#   - reference floor                → M29_baseline_naive_sum

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
