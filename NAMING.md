# NAMING.md — Method Naming Convention

**Effective:** 2026-05-10. All method names follow `M<NN>_<family>_<descriptor>`.

Hard rename (no aliases). Old names removed from code. Historical CSV snapshots preserved at `archive/reports_pre_rename_2026-05-10/`.

## Stability Rule

- `Mxx` IDs **assigned once, never re-shuffled.**
- New methods get `M(max+1)`.
- Initial assignment ranking-based on **953-tree benchmark (2026-05-10)**, tie-break by MAE ascending.
- Numeric order is **not** an ongoing ranking — read accuracy from CSV / docs.

## Family Glossary

| Family | Meaning |
|---|---|
| `selector` | Routes per regime to sub-algorithms |
| `blend` | Weighted / geometric composite of multiple estimators |
| `weight` | Geometric weighting (visibility, coverage) |
| `divide` | Global / adaptive divisor correction |
| `entropy` | Entropy-modulated divisor |
| `stack` | Stacking with bracket / density correction |
| `boost` | Per-class multiplier |
| `median` | Median-based aggregation |
| `consensus` | Multi-estimator voting |
| `anchor` | Floor-anchored specialist |
| `ordinal` | Ordinal class correction |
| `agree` | Side-agreement ratio |
| `baseline` | Reference floor (naive sum, strict match) |

## Mapping Table — Old → New (29 methods)

| New ID | New Name | Old Name(s) | Acc±1 (953) | MAE | Source File |
|:--:|---|---|:--:|:--:|---|
| M01 | `M01_selector_b2b3` | `selector_with_b2b3` | 86.67% | 0.3982 | `algorithms/M01_selector_b2b3.py` |
| M02 | `M02_selector_trifurc` | `selector_iter9_trifurc` | 86.67% | 0.3987 | `algorithms/M02_selector_trifurc.py` |
| M03 | `M03_blend_geometric` | `geometric_mean_blend` | 86.15% | 0.3961 | `algorithms/M03_blend_geometric.py` |
| M04 | `M04_blend_floor_clamped` | `floor_clamped_hybrid` | 86.04% | 0.4050 | `algorithms/M04_blend_floor_clamped.py` |
| M05 | `M05_blend_vis_divide` | `hybrid_vis_corr` | 86.04% | 0.4077 | `algorithms/M05_blend_vis_divide.py` |
| M06 | `M06_weight_visibility` | `visibility` / `v2_visibility` / `visibility_count` | 85.94% | 0.3960 | `algorithms/M06_weight_visibility.py` |
| M07 | `M07_weight_coverage` | `side_coverage` | 85.94% | 0.3930 | `algorithms/M07_weight_coverage.py` |
| M08 | `M08_divide_density_vis` | `density_scaled_vis` | 85.94% | 0.4020 | `algorithms/M08_divide_density_vis.py` |
| M09 | `M09_median_strong5` | `v9_median_strong5` / `median_strong5` | 85.73% | 0.4010 | `algorithms/M09_median_strong5.py` |
| M10 | `M10_entropy_divide` | `v8_entropy_modulated` / `entropy_modulated` | 84.78% | 0.4510 | `algorithms/M10_entropy_divide.py` |
| M11 | `M11_median_b2` | `v9_b2_median_v6` / `b2_median_v6` | 84.78% | 0.4290 | `algorithms/M11_median_b2.py` |
| M12 | `M12_selector_overrides` | `v9_selector` ⚠ overfits 228 | 84.68% | 0.4410 | `algorithms/M12_selector_overrides.py` |
| M13 | `M13_stack_bracket` | `v7_stacking_bracketed` / `stacking_bracketed` | 84.58% | 0.4280 | `algorithms/M13_stack_bracket.py` |
| M14 | `M14_stack_density` | `v7_stacking_density` / `stacking_density` | 84.58% | — | `algorithms/M14_stack_density.py` |
| M15 | `M15_divide_global` | `corrected` / `v1_corrected` / `corrected_naive` | 84.37% | 0.4160 | `algorithms/M15_divide_global.py` |
| M16 | `M16_boost_b2b4` | `v8_b2_b4_boosted` / `b2_b4_boosted` | 84.37% | — | `algorithms/M16_boost_b2b4.py` |
| M17 | `M17_selector_regime` | `v6_selector` | 84.26% | 0.4440 | `algorithms/M17_selector_regime.py` |
| M18 | `M18_entropy_stack` | `v8_entropy_stacking` | — | — | wrapper in `dedup_all_953.py` |
| M19 | `M19_divide_adaptive` | `adaptive_corrected` / `v5_adaptive_corrected` | 82.58% | 0.4600 | `algorithms/M19_divide_adaptive.py` |
| M20 | `M20_weight_visibility_grid` | `best_visibility_grid` / `v5_best_visibility` | 80.80% | 0.4600 | `algorithms/M20_weight_visibility_grid.py` |
| M21 | `M21_ordinal_b3` | `v7_ordinal_b3` / `ordinal_b3` | low | — | `algorithms/M21_ordinal_b3.py` |
| M22 | `M22_anchor_floor50` | `v8_floor_anchor_50` / `floor_anchor_50` | — | — | `algorithms/M22_anchor_floor50.py` |
| M23 | `M23_agree_side` | `v8_side_agreement` / `side_agreement` | — | — | `algorithms/M23_agree_side.py` |
| M24 | `M24_weight_class_aware` | `class_aware_vis` | 70.93% | 0.5460 | `algorithms/M24_weight_class_aware.py` |
| M25 | `M25_consensus_multi` | `v8_multi_consensus` / `multi_consensus` | 18.86% | — | `algorithms/M25_consensus_multi.py` |
| M26 | `M26_median_per_side` | `v8_per_side_median` / `per_side_median` | 18.86% | — | `algorithms/M26_median_per_side.py` |
| M27 | `M27_weight_visibility_adaptive` | `adaptive_visibility` | — | — | `algorithms/M27_weight_visibility_adaptive.py` |
| M28 | `M28_baseline_match_strict` | `relaxed_match` | 5.98% | 1.8110 | `algorithms/M28_baseline_match_strict.py` |
| M29 | `M29_baseline_naive_sum` | `naive` | 3.99% | 2.2800 | `algorithms/M29_baseline_naive_sum.py` |

**Note:** `relaxed_match` renamed `M28_baseline_match_strict` — old name was misleading (algorithm is strict-Hungarian; "relaxed" referred only to internal threshold tolerance).

## Adding a New Method

1. Pick the next available ID (`M30`, `M31`, ...).
2. Choose family from glossary (or extend glossary if genuinely new family).
3. Choose descriptor (lowercase, snake_case, ≤3 words).
4. Create `algorithms/M<NN>_<family>_<descriptor>.py` exporting `predict(detections, params=None) -> dict`.
5. Register in `scripts/dedup_brand_new_953.py` METHOD_GROUPS.
6. Append row to mapping table above.

## Cross-Imports (for refactor sanity)

- `M01_selector_b2b3` self-contained (calls own internal helpers)
- `M02_selector_trifurc` self-contained
- `M11_median_b2`, `M12_selector_overrides`, `M09_median_strong5` import `M17_selector_regime` (load_params + predict)
- `M12_selector_overrides` also imports `M13_stack_bracket`, `M16_boost_b2b4`, `M22_anchor_floor50`
