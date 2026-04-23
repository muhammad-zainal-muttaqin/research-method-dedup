# Deduplication Research v2 Report
**Date:** 2026-04-23
**Dataset:** 228 trees with JSON GT
**Goal:** Reach ≥98% within-±1 accuracy per class from raw multi-view detections (no linking info)

## Best Performing Method
**Method:** visibility
**Params:** {'alpha': 1.0, 'sigma': 0.3}
**Mean MAE:** 0.2719
**Accuracy (±1 error per class):** 92.1%
**Mean Total Error:** 1.09
**Score:** 89.39

## Per-Class Performance (Best Method)
- **B1**: MAE=0.105, trees with |error|>1=0
- **B2**: MAE=0.237, trees with |error|>1=5
- **B3**: MAE=0.452, trees with |error|>1=10
- **B4**: MAE=0.294, trees with |error|>1=4

## Method Comparison (Top 15)
| method         | params                                                                                             |   mean_MAE |   acc_within_1_error |   mean_total_error |   score |   n_trees |
|:---------------|:---------------------------------------------------------------------------------------------------|-----------:|---------------------:|-------------------:|--------:|----------:|
| stack_5        | {'corrected': 0.33, 'graph': 0.0, 'adaptive_ridge': 0.0, 'visibility': 0.66, 'rich_cluster': 0.0}  |     0.2719 |                92.11 |               1.09 |   89.39 |       228 |
| visibility     | {'alpha': 1.0, 'sigma': 0.3}                                                                       |     0.2719 |                92.11 |               1.09 |   89.39 |       228 |
| stack_5        | {'corrected': 0.0, 'graph': 0.0, 'adaptive_ridge': 0.0, 'visibility': 1.0, 'rich_cluster': 0.0}    |     0.2719 |                92.11 |               1.09 |   89.39 |       228 |
| stack_5        | {'corrected': 0.0, 'graph': 0.0, 'adaptive_ridge': 0.33, 'visibility': 0.66, 'rich_cluster': 0.0}  |     0.2719 |                92.11 |               1.09 |   89.39 |       228 |
| visibility     | {'alpha': 1.0, 'sigma': 0.25}                                                                      |     0.2884 |                92.11 |               1.15 |   89.22 |       228 |
| stack_3        | {'corrected': 0.4, 'graph': 0.0, 'adaptive_ridge': 0.6}                                            |     0.2654 |                91.67 |               1.06 |   89.01 |       228 |
| stack_5        | {'corrected': 0.33, 'graph': 0.0, 'adaptive_ridge': 0.66, 'visibility': 0.0, 'rich_cluster': 0.0}  |     0.2654 |                91.67 |               1.06 |   89.01 |       228 |
| stack_5        | {'corrected': 0.0, 'graph': 0.0, 'adaptive_ridge': 1.0, 'visibility': 0.0, 'rich_cluster': 0.0}    |     0.2654 |                91.67 |               1.06 |   89.01 |       228 |
| adaptive_ridge | {'alpha': 1.0}                                                                                     |     0.2654 |                91.67 |               1.06 |   89.01 |       228 |
| stack_5        | {'corrected': 0.0, 'graph': 0.0, 'adaptive_ridge': 0.66, 'visibility': 0.33, 'rich_cluster': 0.0}  |     0.2654 |                91.67 |               1.06 |   89.01 |       228 |
| stack_3        | {'corrected': 0.2, 'graph': 0.0, 'adaptive_ridge': 0.8}                                            |     0.2654 |                91.67 |               1.06 |   89.01 |       228 |
| stack_3        | {'corrected': 0.0, 'graph': 0.0, 'adaptive_ridge': 1.0}                                            |     0.2654 |                91.67 |               1.06 |   89.01 |       228 |
| stack_5        | {'corrected': 0.33, 'graph': 0.0, 'adaptive_ridge': 0.33, 'visibility': 0.33, 'rich_cluster': 0.0} |     0.2785 |                90.79 |               1.11 |   88    |       228 |
| stack_5        | {'corrected': 0.66, 'graph': 0.0, 'adaptive_ridge': 0.33, 'visibility': 0.0, 'rich_cluster': 0.0}  |     0.2851 |                90.79 |               1.14 |   87.94 |       228 |
| stack_5        | {'corrected': 1.0, 'graph': 0.0, 'adaptive_ridge': 0.0, 'visibility': 0.0, 'rich_cluster': 0.0}    |     0.2851 |                90.79 |               1.14 |   87.94 |       228 |

## Error Pattern Analysis
Trees failing ±1 constraint: 18 / 228 (7.9%)

Breakdown by class for failing trees:
- B1: 0 trees have |error|>1 in this class
- B2: 5 trees have |error|>1 in this class
- B3: 10 trees have |error|>1 in this class
- B4: 4 trees have |error|>1 in this class

## Key Insights
- A baseline or heuristic method achieved the best score in this sweep.
- Cross-view visibility downweighting helps but is sensitive to sigma/alpha.
- Graph matching benefits from tight tol_y and tol_area; too loose causes over-merging.
- Ridge LOO-CV is fast and effective; RF offers marginal gains at higher compute.

## Recommendation
**Gap remains:** 92.1% within ±1. Strongest remaining signal is likely visual appearance embedding; pursue embedding-based cross-view matching next.
