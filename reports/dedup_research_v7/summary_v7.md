# Dedup Research V7 Report
**Date:** 2026-04-24
**Goal:** Break 93.86% ceiling with generalization-first approach

## Governance
- Y_MEDIANS computed from full dataset (no per-class ordinal leakage)
- LOTO refits BASE_FACTORS + density slope per fold (227 train → 1 predict)
- No grid search on test split

## Full-Dataset Method Comparison
```
                 method   acc    mae
     stacking_bracketed 94.30 0.2643
       stacking_density 94.30 0.2708
            v7_combined 93.86 0.2697
     adaptive_bracketed 93.86 0.2708
 b3_quadratic_bracketed 93.86 0.2741
  adaptive_corrected_v5 93.86 0.2774
           b3_quadratic 93.86 0.2807
          visibility_v5 92.54 0.2664
    v7_combined_ordinal 92.11 0.2971
LOTO_adaptive_corrected 91.67 0.2862
   ordinal_b3_modulated 91.23 0.2939
  merged_b23_diagnostic 46.05 0.8629
```

## LOTO Cross-Validation (adaptive_corrected)
- Point estimate (full dataset): 93.86%
- LOTO estimate: 91.67%
- Gap (point − LOTO): 2.19pp
- Interpretation: MILD OVERFIT detected — gap > 1.5pp

## Per-Split Breakdown (stacking_bracketed)
```
split   n    acc    mae
 test  31  83.87 0.4274
train 196  95.92 0.2372
  val   1 100.00 0.5000
```

## Per-Split Baseline (adaptive_corrected_v5)
```
split   n    acc    mae
 test  31  80.65 0.4677
train 196  95.92 0.2462
  val   1 100.00 0.5000
```

## Error Analysis (Best Method: stacking_bracketed)
- Trees with error > 1: 13 / 228
- Mean MAE on failing trees: 0.7115  (baseline was 0.7357 avg per failing tree)
- B3 mean error on failing trees: 1.3846
- Gap to 95%: need 2 more trees correct

## Direction-by-Direction Findings
| Direction | Method | Acc | MAE | vs Baseline |
|-----------|--------|-----|-----|-------------|
| — | stacking_bracketed | 94.30% | 0.2643 | +0.44pp |
| — | stacking_density | 94.30% | 0.2708 | +0.44pp |
| — | v7_combined | 93.86% | 0.2697 | +0.0pp |
| — | adaptive_bracketed | 93.86% | 0.2708 | +0.0pp |
| — | b3_quadratic_bracketed | 93.86% | 0.2741 | +0.0pp |
| — | adaptive_corrected_v5 | 93.86% | 0.2774 | +0.0pp |
| — | b3_quadratic | 93.86% | 0.2807 | +0.0pp |
| — | visibility_v5 | 92.54% | 0.2664 | -1.32pp |
| — | v7_combined_ordinal | 92.11% | 0.2971 | -1.75pp |
| — | LOTO_adaptive_corrected | 91.67% | 0.2862 | -2.19pp |
| — | ordinal_b3_modulated | 91.23% | 0.2939 | -2.63pp |
| — | merged_b23_diagnostic | 46.05% | 0.8629 | -47.81pp |

## Conclusions
- LOTO gap = 2.19pp → some overfit present
- Best new method: stacking_bracketed @ 94.30%
- Target 95% = 217 correct trees (2 still needed)
