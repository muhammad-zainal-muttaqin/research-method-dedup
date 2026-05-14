# Dedup Research V5 Report
**Date:** 2026-04-23
**Best Method:** adaptive_corrected
**Acc +/-1:** 93.86%
**MAE:** 0.2774
**Bootstrap 95% CI:** 90.79% - 96.93%

## Method Comparison
```
               method  mean_MAE  acc_within_1_error  score
   adaptive_corrected    0.2774               93.86  91.09
   best_ensemble_grid    0.2774               93.86  91.09
 best_visibility_grid    0.2664               92.54  89.88
      hybrid_vis_corr    0.2697               92.54  89.85
best_class_aware_grid    0.2917               92.54  89.63
        side_coverage    0.2697               92.11  89.41
           visibility    0.2719               92.11  89.39
   density_scaled_vis    0.2719               92.11  89.39
  adaptive_visibility    0.3366               91.67  88.30
            corrected    0.2851               90.79  87.94
  naive_mean_ensemble    0.2719               90.35  87.63
        ordinal_prior    0.3158               89.04  85.88
      class_aware_vis    0.3805               85.09  81.28
    best_relaxed_grid    1.7456                6.58 -10.88
        relaxed_match    1.8969                2.63 -16.34
                naive    2.1294                2.63 -18.66
```

## Best Grid Result
- Method: adaptive_corrected
- Score: 91.09
- Acc +/-1: 93.86%
- MAE: 0.2774
- Mean Total Error: 1.11

## Bootstrap 95% CI
- Point estimate: 93.86%
- 95% CI: [90.79%, 96.93%]
- Standard Error: 1.6087
- CI lower bound <= V4 baseline (92.11%)

## Per-Class Breakdown
| Class | MAE | Acc +/-1 |
|-------|-----|----------|
| B1 | 0.110 | 100.0% |
| B2 | 0.237 | 98.2% |
| B3 | 0.434 | 96.9% |
| B4 | 0.329 | 98.2% |

## Per-Domain Breakdown
| Domain | MAE | Acc +/-1 |
|--------|-----|----------|
| train | 0.246 | 95.9% |
| val | 0.500 | 100.0% |
| test | 0.468 | 80.6% |

## Error Analysis
- Trees with error > 1: 14 / 228 (6.1%)
- Mean error sum (failing trees): 2.93

## Final Claim
**Primary metric (4-class strict Acc +/-1): 93.86%**

Outputs in `reports/dedup_research_v5/`
