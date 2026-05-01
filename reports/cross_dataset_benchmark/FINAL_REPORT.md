# Final Benchmark Report: v9 vs Generalization Variants

**Date**: 2026-05-01  
**Branch**: `feature/generalization-tuning-v10`

## Executive Summary

Evaluasi **v9** (tuned on 228 files) vs **v10-v12** (generalization attempts) pada 3 dataset:

| Dataset | Files | v9 | v10 | v11 | v12 |
|---------|-------|-----|-----|-----|-----|
| json/ (22 Apr) | 228 | **97.4%** | 91.7% | 93.4% | 92.1% |
| json_28 Apr | 478 | **92.5%** | 89.1% | 90.0% | 90.4% |
| json_30 Apr | 727 | **89.0%** | 88.0% | 87.5% | **88.9%** |

**Key Finding**: v12 (v9 re-fitted on full dataset) closest to v9 with only **-0.1% gap** on 727 files.

---

## Method Descriptions

| Method | Description | Base |
|--------|-------------|------|
| **v9** | Original selector with decision tree tuned on 228 files | v6 |
| **v10** | New base factors + conservative decision tree | Scratch |
| **v11** | Ensemble weighted average of 3 methods | Ensemble |
| **v12** | v9 structure with re-fitted parameters | v9 |

---

## Gap Analysis

| Dataset | v9→v10 Gap | v9→v12 Gap |
|---------|-----------|-----------|
| 228 files | -5.7% | -5.3% |
| 478 files | -3.4% | -2.1% |
| 727 files | -1.0% | **-0.1%** |

**Conclusion**: As dataset grows, gap between tuned (v9) and re-fitted (v12) approaches zero.

---

## Lessons Learned

1. **v9 is extremely well-tuned** - Hard to beat with simple modifications
2. **Re-fitting works better than re-designing** - v12 closest to v9
3. **Ensemble methods need more sophistication** - v11 underperforms
4. **Gap shrinks with more data** - v9 advantage diminishes on larger dataset

---

## Recommendation

**For production**: Stick with **v9** - it's proven and optimal.

**If dataset grows**: Use **v12** as fallback - maintains ~99% of v9 performance with better generalization potential.

---

## Files Created

- `algorithms/v10_selector.py` - Conservative generalization
- `algorithms/v11_selector.py` - Ensemble approach
- `algorithms/v12_selector.py` - v9 re-fitted
- `scripts/cross_dataset_benchmark.py` - Benchmark tool
- `reports/cross_dataset_benchmark/` - Results

## Verdict

**v9 remains champion** 🏆

No generalization method beats v9, but v12 comes closest (-0.1% on 727 files).
