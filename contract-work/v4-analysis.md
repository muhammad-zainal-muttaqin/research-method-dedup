# V4 Dedup Research Analysis Report

**Date:** 2026-04-23  
**Analysis Scope:** Current `dedup_research_v4.py` (V4.1: Pixel-Aware Empirical Geometry), reports in `reports/dedup_research_v4/`, alignment with RESEARCH.md pure algorithmic constraints.  
**Resources Discovered (in order):**  
1. `RESEARCH.md` (Sections 0.1-0.9, 0.2a algorithmic constraints, decision metrics, what NOT to do - no training/gradients/learned params)  
2. `CLAUDE.md` (JSON schema, experiment ledger showing visibility as 92.11% ceiling, v4 in reports/)  
3. `README.md` (confirms heuristic ceiling at 92.1% with visibility; cylindrical priors mentioned in next phase)  
4. `reports/dedup_research_v4/summary_v4.md`, `method_comparison_v4.csv`, `best_method_details_v4.csv`, `error_analysis_v4.csv`, `empirical_model.json`  
5. `scripts/dedup_research_v4.py` (key patterns: load_tree_data with HSV/Laplacian crops, learn_empirical_model from _confirmedLinks, mahalanobis_distance, mahalanobis_hungarian_count with UnionFind, visibility_count, ensemble)  
6. JSON samples (`json/*.json`) confirming schema with "bunches", "appearances", "_confirmedLinks", bbox_yolo/pixel, side_index.

**Current Best Method and Exact % (Q1):**  
From `method_comparison_v4.csv` and `summary_v4.md`: **visibility** remains best at **92.11% Acc ±1**, MAE=0.2719, score=89.39 (n=228 trees). V4 additions (v4_mahalanobis_hungarian: 29.82% acc, v4_ensemble: 82.89%) did not surpass it. Confirmed in error_analysis (mostly B3/B2 errors, 17 failing trees).

**Cylindrical Priors and Per-Class Side-Pair Statistics (Q2, based on JSON schema):**  
- **Cylindrical priors:** Model tree trunk as cylinder; side views (sisi_1/3 front/back vs sisi_2/4 sides) imply expected occlusion/visibility and vertical position consistency (cy_norm bias by side_index). In v4: used via side_index in pairing (adjacent sides in Hungarian/UnionFind), y_norm/cy diffs in Mahalanobis, and visibility decay on x_norm (horizontal position proxies cylindrical projection). Per JSON: "appearances" list side+box_index, "summary.by_class" gives GT unique counts.  
- **Per-class side-pair statistics:** From `_confirmedLinks` in JSON (true matches between sideA/sideB bboxIds), v4's `learn_empirical_model` computes per-class (B1-B4) mean/cov of diffs (d_cy, d_area, d_ar, d_cx). This captures e.g. B4 (top/spiny) has tighter vertical consistency across sides than B1 (bottom). Extended with HSV means and Laplacian texture from bbox crops (pixel-aware). Empirical_model.json stores these per-class stats.

**Potential Implementation Approach for Adaptive Per-Tree Density Weighting (Q3):**  
Pure heuristic: Per-tree, compute local density (e.g. avg #detections per side or cluster density via DBSCAN on (cx,cy,area)). Weight count inversely (high-density sides get lower per-detection contribution, e.g. visibility_count variant with tree-specific sigma/decay derived from naive_count variance). Use ordinal position priors (B1 low-cy, B4 high-cy averaged across sides). Deterministic, no training - closed-form from per-tree stats only. Fits cylindrical by modulating weights based on side_index * density. Validate on 228 JSON by comparing to summary.by_class.

**Denser Ensemble Grid Details (Q4):**  
Current v4 uses simple avg of visibility + mahalanobis_hungarian + corrected. Denser grid: Cartesian product over params (visibility: alpha=[0.5,1.0,1.5], sigma=[0.2-0.5]; cost_thresh=[1.5,2.0,2.5,3.0]; ensemble weights [0.4,0.4,0.2] variants, per-class factors). Run grid on 228, rank by score=acc - 10*MAE. Pure combinatorial (no opt), output top-k to CSV. Extend with cylindrical-adjusted cost (scale Mahalanobis by expected side-pair prob from stats).

**List of All TODOs or Next Steps Mentioned (Q5):**  
- From summary_v4.md: If <95%, add Laplacian texture variance for spiny B4 or refine cylindrical priors further.  
- From RESEARCH.md (Secs 0.2a, next steps): Geometry refinement (intrinsics/extrinsics, epipolar, 3D triangulation), topological matching, stat ensemble (median/consensus, no learned), 3-class reframing (B2+B3→B23). Avoid all learned approaches. V4 pushed pixel-aware but still at 92.11% ceiling. No re-run of closed sections.

**Recommended Milestones for Mission (Vertical Slices for Validation) (Q6):**  
1. **Slice 1 (Validation):** Reproduce V4.1 on subset of 20 JSON trees; verify empirical_model matches _confirmedLinks diffs; confirm visibility best.  
2. **Slice 2 (Cylindrical):** Implement per-class side-pair prob matrix from JSON links; integrate as prior in mahalanobis cost. Measure uplift on B3/B4 errors.  
3. **Slice 3 (Adaptive Density):** Add per-tree density weighting to visibility; evaluate MAE vs baseline on full 228.  
4. **Slice 4 (Ensemble Grid):** Run denser grid (20+ combos), update method_comparison_v4.csv; select new best.  
5. **Slice 5 (Report):** Generate updated summary with bootstrap CI, per-domain breakdown; align to RESEARCH.md metrics (Acc±1 primary). All deterministic/heuristic only.

**Key Code Patterns from v4 Script (no full copy):**  
- `load_tree_data()`: PIL crop → HSV mean + Laplacian var per annotation; augments with y_norm/x_norm/area_norm/side_index.  
- `learn_empirical_model()`: Aggregate diffs from `_confirmedLinks` per-class → mean/cov/inv_cov (Mahalanobis).  
- `mahalanobis_distance()`: Vector of geometric+color+texture deltas; dynamic pad for extra dims.  
- `mahalanobis_hungarian_count()`: Per-class, per-adjacent-sides bipartite (linear_sum_assignment), UnionFind for connected components; cost_thresh filter.  
- Baselines: `visibility_count()` (Gaussian on x_norm), `corrected_naive()` (class-specific factors), ensemble round(avg).  
- `evaluate_predictions()`: Per-tree MAE, within_±1 acc, score; outputs CSVs + summary.md. Matches CLAUDE.md patterns from v2/v3.

**Suggested Worker Skill Names:** review, simplify, caveman-review, design-taste-frontend (for potential viz in reports), session-navigation.

**Output File Written:** `D:\Work\Assisten Dosen\research-method-dedup\contract-work\v4-analysis.md` (complete structured report above). No code changes made. All analysis pure read/exploration.

**Blockers/Notes:** V4.1 did not improve beyond visibility (still ceiling per RESEARCH.md). Some JSONs reference "_confirmedLinks" (present in schema but not all files). No uncertainties on constraints - fully algorithmic. No further work.
