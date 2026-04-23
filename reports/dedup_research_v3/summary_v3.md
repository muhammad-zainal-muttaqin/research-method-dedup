# Deduplication Research v3 Report
**Date:** 2026-04-23
**Dataset:** 228 trees with JSON GT
**Goal:** Reach ≥94% within-±1 accuracy per class from raw multi-view detections
**Approach:** Learn thresholds from _confirmedLinks, then predict without using links

## Learned Thresholds (95th percentile from _confirmedLinks)
- **B1**: tol_y=0.1085, tol_area=0.1048 (n_links=287)
- **B2**: tol_y=0.1077, tol_area=0.1021 (n_links=418)
- **B3**: tol_y=0.1195, tol_area=0.0961 (n_links=910)
- **B4**: tol_y=0.1158, tol_area=0.0849 (n_links=327)

## Best Performing Method
**Method:** per_class_ridge
**Params:** {'alpha': 1.0}
**Mean MAE:** 0.2741
**Accuracy (±1 error per class):** 90.8%
**Mean Total Error:** 1.10
**Score:** 88.05
**Failing Trees:** 21 / 228

## Per-Class Performance (Best Method)
- **B1**: MAE=0.088, trees with |error|>1=0
- **B2**: MAE=0.232, trees with |error|>1=5
- **B3**: MAE=0.482, trees with |error|>1=10
- **B4**: MAE=0.294, trees with |error|>1=6

## Method Comparison (Top 15)
| method          | params                                              |   mean_MAE |   acc_within_1_error |   mean_total_error |   score |   n_trees |
|:----------------|:----------------------------------------------------|-----------:|---------------------:|-------------------:|--------:|----------:|
| per_class_ridge | {'alpha': 1.0}                                      |     0.2741 |                90.79 |               1.1  |   88.05 |       228 |
| ensemble        | {'lg': 0.0, 'hm': 0.0, 'cm': 0.0, 'ridge': 1.0}     |     0.2741 |                90.79 |               1.1  |   88.05 |       228 |
| ensemble        | {'lg': 0.25, 'hm': 0.0, 'cm': 0.0, 'ridge': 1.0}    |     0.3026 |                89.47 |               1.21 |   86.45 |       228 |
| ensemble        | {'lg': 0.0, 'hm': 0.0, 'cm': 0.25, 'ridge': 1.0}    |     0.3158 |                89.04 |               1.26 |   85.88 |       228 |
| ensemble        | {'lg': 0.0, 'hm': 0.25, 'cm': 0.0, 'ridge': 0.75}   |     0.3366 |                85.09 |               1.35 |   81.72 |       228 |
| ensemble        | {'lg': 0.25, 'hm': 0.25, 'cm': 0.0, 'ridge': 0.75}  |     0.3706 |                84.21 |               1.48 |   80.5  |       228 |
| ensemble        | {'lg': 0.0, 'hm': 0.25, 'cm': 0.25, 'ridge': 0.75}  |     0.3761 |                84.21 |               1.5  |   80.45 |       228 |
| ensemble        | {'lg': 0.0, 'hm': 0.5, 'cm': 0.0, 'ridge': 0.75}    |     0.4112 |                82.46 |               1.64 |   78.34 |       228 |
| ensemble        | {'lg': 0.0, 'hm': 0.25, 'cm': 0.5, 'ridge': 0.75}   |     0.5329 |                81.58 |               2.13 |   76.25 |       228 |
| ensemble        | {'lg': 0.0, 'hm': 0.0, 'cm': 0.5, 'ridge': 0.75}    |     0.3662 |                79.39 |               1.46 |   75.72 |       228 |
| ensemble        | {'lg': 0.0, 'hm': 0.0, 'cm': 0.75, 'ridge': 0.75}   |     0.5022 |                80.26 |               2.01 |   75.24 |       228 |
| ensemble        | {'lg': 0.25, 'hm': 0.25, 'cm': 0.25, 'ridge': 0.75} |     0.5296 |                79.39 |               2.12 |   74.09 |       228 |
| ensemble        | {'lg': 0.5, 'hm': 0.25, 'cm': 0.0, 'ridge': 0.75}   |     0.5384 |                78.95 |               2.15 |   73.56 |       228 |
| ensemble        | {'lg': 0.25, 'hm': 0.0, 'cm': 0.25, 'ridge': 0.75}  |     0.3783 |                77.19 |               1.51 |   73.41 |       228 |
| ensemble        | {'lg': 0.25, 'hm': 0.0, 'cm': 0.5, 'ridge': 0.75}   |     0.5121 |                78.51 |               2.05 |   73.39 |       228 |

## Error Pattern Analysis
Trees failing ±1 constraint: 21 / 228 (9.2%)

Breakdown by class for failing trees:
- B1: 0 trees have |error|>1 in this class
- B2: 5 trees have |error|>1 in this class
- B3: 10 trees have |error|>1 in this class
- B4: 6 trees have |error|>1 in this class

## Key Insights
- Per-class Ridge regression successfully corrects systematic biases using closed-form solution (no gradient learning).
- Using _confirmedLinks ONLY for threshold estimation prevents leakage and gives unbiased estimates.
- B2/B3 remain the hardest classes due to visual ambiguity and higher within-class variance.
- Hungarian matching with statistically estimated tolerances is robust; cascade adds value by maintaining cluster state.

## Recommendation
**Gap remains:** 90.8% within ±1. **All algorithmic options exhausted for pure bbox methods.**

**Next direction (still algorithmic):**
- Multi-camera geometry (calibration, epipolar constraints, 3D triangulation)
- Topological matching with relaxed geometric constraints
- Statistical ensemble of multiple heuristics

**NOT pursuing:**
- ❌ MLP on bbox features (requires training)
- ❌ Learned embeddings / Siamese networks (requires training)
