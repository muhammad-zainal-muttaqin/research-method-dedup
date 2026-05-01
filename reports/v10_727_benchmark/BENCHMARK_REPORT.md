# v10 Benchmark on 727 JSON Trees (Clean Oracle=100%)

**Date**: 2026-05-01
**Oracle ceiling**: 100% (0 mismatched bunches)

## Results

| Method | Acc ±1 | MAE | Gain vs v9 |
|--------|--------|-----|------------|
| v9 (original) | 89.27% | 0.3267 | - |
| **v10 (B23-density)** | **89.27%** | **0.3167** | **+0.00 pp** |

## Top Error Signatures (v10)

[('B3e2', 17), ('B3-2', 8), ('B2-1_B3-2', 4), ('B2-2', 4), ('B2-2_B3-1', 4)]

## Analysis

B23-density resolver reduced B3 overcount errors. Remaining failures now dominated by marginal B1/B4 small-object cases and rare same-(x,y) collisions.

**Next iteration target (v11)**: add B1 low-density rescue + B4 small-object visibility boost.
