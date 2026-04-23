# Dedup Research V8 Report
**Date:** 2026-04-23
**Goal:** Break 94.30% (v7 best) — target 95% Acc±1

## Key Gap from v7 Error Analysis
- 13 failing trees: 8 train, 5 test
- Dominant errors: B3 over-predicted (7/13), B2 over-predicted in test (4/5), B4 over-predicted (5/13)
- Test split lags train by 12pp → structural gap (density factors tuned on aggregate dominated by train)
- v8 addresses: side distribution (per-side median, entropy, variance), multi-estimator consensus

## Full-Dataset Method Comparison
```
               method   acc    mae
  stacking_density_v7 94.30 0.2708
stacking_bracketed_v7 94.30 0.2643
    entropy_modulated 94.30 0.2763
  v8_entropy_stacking 94.30 0.2763
       b2_boosted_112 93.42 0.2719
        b2_b4_boosted 92.54 0.2632
        side_variance 89.91 0.3147
             blend_80 88.60 0.3169
      floor_anchor_70 85.96 0.3235
       side_agreement 83.33 0.3618
             blend_70 78.07 0.4035
      floor_anchor_50 69.74 0.4211
 v8_consensus_entropy 67.98 0.4857
             blend_60 63.16 0.5307
      floor_anchor_30 46.93 0.6557
      multi_consensus 18.86 0.9583
      per_side_median 18.86 0.9583
```

## Best Method: stacking_density_v7
- Acc±1: 94.30%  (v7 was 94.30%)
- MAE: 0.2708
- Failing trees: 13 / 228
- Gap to 95%: need 1 more trees correct

## Per-Split (Best Method)
```
split   n    acc    mae
 test  31  83.87 0.4274
train 196  95.92 0.2449
  val   1 100.00 0.5000
```

## Delta vs v7 stacking_density
- Improved: 0 trees
- Regressed: 0 trees
