# Final ML Assertions for Cross-area + Graph/ML
**Date:** 2026-04-23  
**Criteria:** >99% accuracy, MAE &lt; 0.05 on test split (within ±1 error per class?)  
**Sources checked:** reports/dedup_research/*, full_gt_count/summary.md, RESEARCH.md, scripts/dedup_research.py

## Graph/ML (Hungarian graphs, embed clustering)
**Status:** No implementations found.  
**Closest approximations:**  
- `feature_cluster`: DBSCAN on (y_norm, sqrt(area_norm)) features. Params: (eps, min_samples, e.g. 0.05-0.15, 1-2).  
  - Best: (0.05, 8, 1) — MAE=1.4452, acc_within_1=13.16%, total_error=5.78  
**None meet criteria** (&gt;99% acc, MAE&lt;0.05).  

## Cross-area (end-to-end json/ → reports/, test split MAE/acc)
**Status:** No dedicated cross-area (DAMIMAS vs LONSUM) pipeline.  
- All 228 JSON trees: DAMIMAS_A21B_* (no LONSUM JSON available).  
- Test split: 31 trees (DAMIMAS), `corrected` method:  
  - Sample MAE: 0.0–1.0 (e.g. 0554:1.0 False, 0557:1.0 False)  
  - within_1 True ~80% (6/31 False). Mean MAE ~0.3 (est from samples).  
- Best overall (`corrected,default`): MAE=0.2851, acc_within_1=90.79% (228 trees).  
**None meet criteria.** No MAE&lt;0.05 or &gt;99% acc. No bootstrap CI.  

## Evidence Summary
| Method | MAE | acc_within_1 | Notes |
|--------|-----|--------------|-------|
| corrected | 0.2851 | 90.79% | Best heuristic (naive / overcount factors). |
| feature_cluster | 1.44+ | &lt;15% | Embed-like (y+area). |
| y_bin | 1.73+ | &lt;7% | Position binning. |
| naive | 2.13 | 2.63% | Sum detections. |

**No VAL-FINAL-* or CROSS-* dirs found.**  
**Files:**  
- [method_comparison.csv](reports/dedup_research/method_comparison.csv)  
- [best_method_details.csv](reports/dedup_research/best_method_details.csv)  
- [summary.md](reports/dedup_research/summary.md)  

**Verdict:** No final methods pass strict thresholds. `corrected` closest baseline.
