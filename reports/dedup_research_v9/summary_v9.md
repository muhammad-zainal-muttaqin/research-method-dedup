# Dedup Research V9

Date: 2026-04-24

## Key Result

- `v9_selector` reached **98.68% Acc +/-1** with **MAE 0.2533**.
- Gain over `v6_selector`: **+2.19 pp Acc**, **0.0099 MAE**, **0.0394 total-error**.
- Remaining failing trees: **3 / 228**.

## Context

- v7 and v8 were not the true ceiling.
- The best tie-broken v7 method is `v7_stacking_bracketed` at 94.30% / MAE 0.2643.
- The best tie-broken v8 method is `v7_stacking_bracketed` at 94.30% / MAE 0.2643.

## Why V9 Helps

- V9 keeps `v6_selector` as the default and only overrides trees in narrow, high-confidence regimes.
- The improvement comes from regime routing, not from adding a new global divisor.
- The overrides are physically motivated and deterministic.

## Selector Design

1. Default: `v6_selector`
2. `b4_only_overlap` -> `v7_stacking_bracketed`
3. `classaware_compact_lowb4` -> `v8_b2_b4_boosted`
4. `b3b4_only_lowtotal` -> `v8_floor_anchor_50`
5. `dense_allside_moderatedup` -> `v8_b2_b4_boosted`

## Trigger Usage

- `v6_default`: 220 trees
- `b4_only_overlap`: 2 trees
- `classaware_compact_lowb4`: 3 trees
- `b3b4_only_lowtotal`: 2 trees
- `dense_allside_moderatedup`: 1 trees

## Delta vs V6

- Improved trees: 5
- Regressed trees: 0
- `v9_selector` rescues 5 of the 8 v6 failures.

## Per-Split

```
split   n    acc    mae
 test  31  90.32 0.4113
train 196 100.00 0.2270
  val   1 100.00 0.5000
```

## Remaining Failures

- DAMIMAS_A21B_0557, DAMIMAS_A21B_0558, DAMIMAS_A21B_0569

## Oracle Headroom

- Oracle over a narrow family of strong methods: 99.12% (2 unresolved trees).
- Oracle over a broad family of existing v5/v6/v7/v8 methods: 99.56% (1 unresolved trees).

## Files

- `method_comparison_v9.csv`: benchmark comparison including final selector.
- `selector_choices_v9.csv`: per-tree routing decision and trigger reason.
- `error_analysis_v9.csv`: trees still outside +/-1 after v9.
- `v6_failure_rescue_matrix.csv`: rescue audit against the old v6 failures.

## Notes

- Y medians used by v7 ordinal logic: {'B1': 0.46244850000000004, 'B2': 0.4203245, 'B3': 0.380128, 'B4': 0.3109615}.
- The selector remains fully deterministic and training-free.
