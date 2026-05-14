# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context

**Task:** Multi-view oil palm fruit bunch counting. Convert detections from 4–8 photo sides per tree into **unique bunch count per maturity class**. Naive sum overcounts ~83.4% (same bunch seen across sides; factor 1.834 from 727-tree GT). Read `RESEARCH.md` Section 0 (esp. 0.6–0.9) before deep work.

**Constraint:** **100% algorithmic / heuristic only.** No training, embeddings, backprop, or learned matchers. All methods must be deterministic and parameter-free (no gradient computation).

**Dataset:** DAMIMAS (854) + LONSUM (99) = **953 trees**. **Single canonical source: `Brand-New-Dataset-YOLO/`** (953 trees COMPLETE). Layout: flat `images/` + `labels/`, `json/` (GT), `train.txt`/`val.txt`/`test.txt` (split membership), `split_manifest.csv` (per-tree split + stratification keys). All consumers — heuristic dedup, E2E pipeline, GT counting — read from this single root. Earlier snapshots (228/478/727/882) and previous duplicate roots (`json/`, `dataset/`, `Tested-Brand-New-Dataset-YOLO/`) live under `archive/_to_review/` as read-only history.

Mostly 4 sides/tree, 45 have 8. Images 960×1280 JPEG. Classes ordinal B1→B4. **Core hard problem: B2↔B3 visually ambiguous** (irreducible per JSON-01 audit, label noise = 0%).

## Setup

```bash
pip install -r requirements.txt
```

Run all scripts from workspace root. Outputs go to `reports/<script>/`.

## Running Scripts

```bash
# GT counting all 953 trees (no GPU, ~1 min)
python scripts/count_all_trees.py

# PRIMARY benchmark — full 953-tree Brand-New-Dataset-YOLO (canonical)
python scripts/dedup_brand_new_953.py       # → reports/dedup_brand_new_953/

# Multi-dimensional benchmark + per-method reports
python scripts/benchmark_multidim.py        # 4-dim evaluation: accuracy, speed, robustness, domain → reports/benchmark_multidim/
python scripts/generate_method_reports.py   # per-method breakdown → reports/methods/

# E2E / ML track (separate from heuristic — see ml-track/CLAUDE-TRAINING.md)
python scripts/build_counting_features.py
python scripts/run_counting_rf.py
python scripts/run_counting_svm.py
python scripts/run_e2e_pipeline.py          # unified harness; per-track wrappers: run_e2e_{m01,rf,svm,inference}.py
python scripts/generate_training_summary.py
python scripts/export_gt_parquet.py
```

**Runtime-library scripts (not benchmarks; imported by the primary):** `scripts/dedup_all_953.py` and `scripts/dedup_research_v5..v9.py` define method bodies that `dedup_brand_new_953.py` re-uses via `import dedup_all_953 as base` → which in turn imports `dedup_research_v6..v9`. `algorithms/M18_entropy_stack.py` also wraps `dedup_research_v8`. Do **not** call these scripts as top-level commands; they only matter as imports.

**Archived 2026-05-14 → `archive/_to_review/`:** all `dedup_research_v1..v4.py`, `dedup_v5_focused.py`, `dedup_all_trees_final.py`, `dedup_nonjson_compare.py`, the four legacy benchmark report folders (`benchmark_228/478/727/882/`), the v2/v3/v4/v6/v7/v8/v9 dedup_research report folders, and one-shot migrators (`migrate_*.py`, `fix_image_filename_bug.py`, `regen_tree_id.py`, `cleanup_repo.py`, `build_brand_new_dataset.py`, `generate_hf_metadata.py`, `flatten_dataset.py`, `find_first_5.py`, `generate_sample_viz.py`). Restore by `Move-Item archive\_to_review\scripts\<file> scripts\<file>`.

## Current Best (as of 2026-05-10)

**PRIMARY BENCHMARK: 953-tree Brand-New-Dataset-YOLO** (canonical, full GT). Earlier 228/478/727/882 numbers are historical only.

**Naming convention (effective 2026-05-10):** all methods use `M<NN>_<family>_<descriptor>`. See `NAMING.md` for full mapping table from old names. IDs are stable — assigned once, never re-shuffled.

### Acc ±1 on 953 Brand-New-Dataset-YOLO trees (PRIMARY)

Note: on balanced 4-class dataset, Macro class-MAE ≡ flat MAE numerically. Full per-class MAE / bias / Total-count MAE breakdown in `README.md` benchmark table or `reports/dedup_brand_new_953/accuracy_953.csv`.

| Rank | Method | Acc ±1 | Macro class-MAE | Total-count MAE | Total ±1 | Exact profile | n_fail |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `M01_selector_b2b3` | **86.67%** | **0.3982** | 1.4145 | 74.08% | 26.34% | 127 |
| 2 | `M02_selector_trifurc` | 86.67% | 0.3987 | 1.4145 | 74.08% | 26.34% | 127 |
| 3 | `M03_blend_geometric` | 86.15% | 0.3961 | 1.4061 | 74.50% | 26.86% | 132 |
| 4 | `M04_blend_floor_clamped` | 86.04% | 0.4050 | 1.4103 | 74.19% | 25.81% | 133 |
| 5 | `M05_blend_vis_divide` | 86.04% | 0.4077 | 1.4145 | 73.98% | 25.29% | 133 |
| 6 | `M06_weight_visibility` | 85.94% | 0.3956 | 1.3641 | 73.56% | 25.29% | 134 |
| 7 | `M07_weight_coverage` | 85.94% | 0.3930 | 1.3599 | 73.77% | 25.81% | 134 |
| 8 | `M08_divide_density_vis` | 85.94% | 0.4024 | 1.3914 | 73.56% | 25.39% | 134 |
| 9 | `M09_median_strong5` | 85.73% | 0.4006 | 1.4638 | 72.51% | 27.39% | 136 |
| 10 | `M10_entropy_divide` | 84.78% | 0.4507 | 1.6348 | 66.32% | 23.92% | 145 |
| 11 | `M11_median_b2` | 84.78% | 0.4294 | 1.5603 | 69.78% | 23.08% | 145 |
| 12 | `M12_selector_overrides` | 84.68% | 0.4410 | 1.6044 | 68.21% | 22.35% | 146 |
| 13 | `M13_stack_bracket` | 84.58% | 0.4284 | 1.5729 | 68.52% | 25.39% | 147 |
| 14 | `M14_stack_density` | 84.58% | 0.4347 | 1.5939 | 67.89% | 23.92% | 147 |
| 15 | `M15_divide_global` | 84.37% | 0.4158 | 1.4596 | 68.52% | 23.29% | 149 |
| 16 | `M16_boost_b2b4` | 84.37% | 0.4111 | 1.4911 | 71.98% | 26.86% | 149 |
| 17 | `M17_selector_regime` | 84.26% | 0.4436 | 1.6149 | 67.89% | 21.93% | 150 |
| 18 | `M18_entropy_stack` | 84.78% | 0.4507 | 1.6348 | 66.32% | 23.92% | 145 |
| 19 | `M19_divide_adaptive` | 82.58% | 0.4599 | 1.6905 | 65.58% | 21.51% | 166 |
| 20 | `M20_weight_visibility_grid` | 80.80% | 0.4596 | 1.5656 | 65.90% | 19.73% | 183 |
| 21 | `M23_agree_side` | 80.80% | 0.4273 | 1.5603 | 65.37% | 22.35% | 183 |
| 22 | `M27_weight_visibility_adaptive` | 80.27% | 0.4790 | 1.6474 | 64.01% | 18.57% | 188 |
| 23 | `M24_weight_class_aware` | 70.93% | 0.5456 | 1.8111 | 58.45% | 12.38% | 277 |
| 24 | `M22_anchor_floor50` | 69.99% | 0.4525 | 1.5540 | 60.55% | 16.89% | 286 |
| 25 | `M25_consensus_multi` | 25.29% | 0.9121 | 3.6401 | 16.79% | 5.46% | 712 |
| 26 | `M26_median_per_side` | 25.29% | 0.9121 | 3.6401 | 16.79% | 5.46% | 712 |
| 27 | `M28_baseline_match_strict` | 5.98% | 1.8114 | 7.0147 | 5.04% | 2.41% | 896 |
| 28 | `M29_baseline_naive_sum` | 3.99% | 2.2804 | 9.1217 | 2.83% | 1.89% | 915 |
| 29 | `M21_ordinal_b3` | 0.73% | 3.5842 | 14.3368 | 0.00% | 0.00% | 946 |

Source: `reports/dedup_brand_new_953/accuracy_953.csv`.

**Surprising findings:**
- **`M01_selector_b2b3` (#1, 2026-05-10)** — selector trifurc + B2↔B3 split correction. +0.63 pp over `M05_blend_vis_divide`, −2.32% MAE. Validated train/val/test held-out, no overfit. Code: `algorithms/M01_selector_b2b3.py`. Report: `report_10Mei2026.md`.
- **`M05_blend_vis_divide`** — previous champion; simple weighted avg of visibility + adaptive divide
- **`M12_selector_overrides` regresses to 84.68%** — confirms severe overfit on 228 dev set (97.37%)
- **Simple visibility family dominates** (top 4 all visibility variants)
- **Strict matching (`M28_baseline_match_strict`, `M29_baseline_naive_sum`) catastrophically fails** at scale

### Historical 228-tree dev set (for reference only)

Fresh benchmark re-run (2026-05-08) on all 4 archive snapshots. Earlier "primary benchmark = 882 trees" superseded.

### Acc ±1 on 228 trees (historical, M12 development set)

| Rank | Method | Acc ±1 | Notes |
|---:|---|---:|---|
| 1 | `M12_selector_overrides` | **97.37%** | Narrow overrides on regime selector — **overfits 228** |
| 2 | `M11_median_b2` | 96.05% | median variant |
| 3 | `M17_selector_regime` | 96.05% | regime selector backbone |
| 4 | `M13_stack_bracket` | 94.30% | stacking best |
| 5 | `M14_stack_density` | 94.30% | density-corrected stacking |
| 6 | `M10_entropy_divide` | 94.30% | entropy modulation |
| 7 | `M19_divide_adaptive` | 93.86% | adaptive divisor |
| 8 | `M16_boost_b2b4` | 92.54% | per-class boost specialist |
| 9 | `M20_weight_visibility_grid` | 92.54% | visibility grid |
| 10 | `M06_weight_visibility` | 92.54% | basic visibility |
| 11 | `M15_divide_global` | 90.79% | global divisor baseline |

### Cross-Dataset Regression (Acc ±1, fresh re-run 2026-05-10)

Sources: `reports/benchmark_228/`, `reports/benchmark_478/`, `reports/benchmark_727/`, `reports/benchmark_882/`, `reports/dedup_brand_new_953/`.

| Method | 228 | 478 | 727 | 882 | **953** | Delta 228→953 |
|---|---:|---:|---:|---:|---:|---:|
| `M12_selector_overrides` | 97.37% | 92.68% | 89.27% | 88.78% | 84.68% | −12.69 pp |
| `M11_median_b2` | 96.05% | 92.68% | 89.00% | 88.78% | 84.78% | −11.27 pp |
| `M17_selector_regime` | 96.05% | 91.84% | 88.86% | 88.55% | 84.26% | −11.79 pp |
| `M10_entropy_divide` | 94.30% | 91.63% | 88.86% | 88.78% | 84.78% | −9.52 pp |
| `M13_stack_bracket` | 94.30% | 91.84% | 88.45% | 88.44% | 84.58% | −9.72 pp |
| `M19_divide_adaptive` | 93.86% | 89.96% | 86.11% | 86.28% | 82.58% | −11.28 pp |
| `M06_weight_visibility` | 92.54% | 90.38% | 89.41% | 89.34% | **85.94%** | **−6.60 pp** |
| `M15_divide_global` | 90.79% | 89.12% | 87.90% | 88.21% | 84.37% | **−6.42 pp** |
| `M01_selector_b2b3` | — | — | — | — | **86.67%** | — (new top, 2026-05-10) |
| `M02_selector_trifurc` | — | — | — | — | 86.67% | — (iter11) |
| `M03_blend_geometric` | — | — | — | — | 86.15% | — (iter11) |
| `M04_blend_floor_clamped` | — | — | — | — | 86.04% | — (iter11) |
| `M05_blend_vis_divide` | — | — | — | — | 86.04% | — (prev champion) |

**Key regression findings (UPDATED 2026-05-10 with 953-tree results):**
- `M01_selector_b2b3` **NEW TOP** at 953 (86.67%, Macro class-MAE 0.3982) — selector trifurc + B2↔B3 split correction, validated held-out
- `M06_weight_visibility` most stable from 228 (−6.60 pp) — simple generalizes best
- `M15_divide_global` second-most stable (−6.42 pp)
- `M12_selector_overrides` drops **12.69 pp** at 953 — narrow overrides catastrophically overfit 228 dev set
- All complex selectors (`M17`, `M13`, `M10`, `M12`) regress 9–13 pp from 228 → 953
- All methods land 82–86% at 953 trees (no catastrophic drop, but ceiling lower than expected)

**Recommendations (UPDATED 2026-05-10 post-iter11):**
- **Production / full 953-tree dataset** → `M01_selector_b2b3` (86.67%, Macro class-MAE 0.3982) — current top, validated held-out, no overfit
- **Simplest fallback** → `M05_blend_vis_divide` (86.04%) — single-line weighted blend
- **Historical 228-tree set** → `M12_selector_overrides` (97.37%) — overfits, dev-set only
- **No missing JSON anymore** — Brand-New-Dataset-YOLO is complete

**M12 logic (regime overrides on top of M17_selector_regime):**
1. default → `M17_selector_regime`
2. `b4_only_overlap` → `M13_stack_bracket`
3. `classaware_compact_lowb4` → `M16_boost_b2b4`
4. `b3b4_only_lowtotal` → `M22_anchor_floor50`
5. `dense_allside_moderatedup` → `M16_boost_b2b4`

## Method Evolution (Why M01 Wins)

Generation labels (v1..v9, iter11) are historical research milestones; production code uses Mxx names only. See `NAMING.md` for old↔new mapping.

| Gen | Best Method (new ID) | Acc ±1 | Lesson |
|---|---|---:|---|
| naive (M29) | `M29_baseline_naive_sum` | very poor | overcount ~78.8% baseline |
| v1 (M15) | `M15_divide_global` | 90.79% | global divisor already beats naive hugely |
| v2 (M06) | `M06_weight_visibility` | 92.11% | bbox geometry / position matters |
| v3 | per_class_ridge | 90.79% | learned-from-link thresholds didn't break ceiling |
| v4 (M06) | `M06_weight_visibility` | 92.11% | adding HSV + Hungarian didn't beat v2 |
| v5 (M19) | `M19_divide_adaptive` | 93.86% | adaptive divisor + class-aware family — first stable >93% |
| v6 (M17) | `M17_selector_regime` | **96.49%** | **turning point** — no single global rule wins; route per regime |
| v7 (M13) | `M13_stack_bracket` | 94.30% | stacking/density family strong but loses to v6 |
| v8 (M13) | `M13_stack_bracket` | 94.30% | entropy/per-side signals add nothing |
| v9 (M12) | `M12_selector_overrides` | **97.37%** (228) / 84.68% (953) | narrow overrides on M17 — best on 228, severely overfits at scale |
| **— production (2026-05-10)** | `M05_blend_vis_divide` | **86.04%** (953) | weighted vis + adaptive divide — wins on full canonical |
| **— iter11 (2026-05-10)** | `M01_selector_b2b3` | **86.67%** (953) | selector trifurc + B2↔B3 split correction — new top, validated held-out |

**Key takeaway:** strict matching (Hungarian, graph, cluster) **fails** on noisy TXT labels (<20% accuracy). Adaptive statistical correction + regime-routing wins on small dev sets but **overfits**. At full 953-tree scale, simpler methods (`M05_blend_vis_divide`, `M06_weight_visibility`) generalize best. B2↔B3 ambiguity is the irreducible ceiling, not label noise.

## Dataset Status (2026-05-10)

`Brand-New-Dataset-YOLO/` is the **complete, canonical 953-tree dataset**. All previously-missing 71 trees now have JSON GT. The "Missing JSON Pipeline" workflow is retired — use Brand-New-Dataset-YOLO for everything.

Verified dedup ratio ≈ 0.55 (best methods reduce naive 18,544 detections to ~10,055 unique bunches across 953 trees). See `reports/dedup_brand_new_953/totals.csv`.

## Repository Layout

```
Brand-New-Dataset-YOLO/        953 trees COMPLETE (canonical, single source)
  data.yaml                    YOLO config (4 classes B1–B4)
  images/                      flat images (3993 .jpg)
  labels/                      flat YOLO TXT labels (3992 .txt)
  json/                        953 JSON GT files — PRIMARY GT
  train.txt / val.txt / test.txt   split membership (images/<name>.jpg per line)
  split_manifest.csv           tree_id → split + stratification keys
algorithms/                    one Mxx_*.py per method, all deterministic.
  __init__.py                  ranked performance table (read for algo selection)
  M01_selector_b2b3.py         current production (86.67% on 953)
  M17_selector_regime.py       regime backbone + load_params()
                               (reads reports/dedup_research_v5/method_comparison_v5.csv)
  M18_entropy_stack.py         wraps scripts/dedup_research_v8.v8_entropy_stacking
scripts/
  dedup_brand_new_953.py       PRIMARY benchmark (953 trees)
  dedup_all_953.py             runtime library — imported by primary, not a CLI entry
  dedup_research_v5..v9.py     runtime library — method bodies imported by dedup_all_953
  count_*.py, benchmark_*.py, generate_*.py, build_counting_features.py,
  run_counting_*.py, run_e2e_*.py, export_gt_parquet.py
reports/                       per-script outputs.
  dedup_brand_new_953/         PRIMARY benchmark output
  dedup_research_v5/           still live (params source for M17)
  benchmark_multidim/, methods/, counting_*/, e2e_*/, full_gt_count/, json_05/, label_audit/
ml-track/                      E2E ML training & inference artefak (Phase 2 cleanup, 2026-05-14).
  baseline-run/                YOLO training logs + weights + SUMMARY.md
    weights/                   5 renamed best.pt (y26n/s/m + ablations)
  predictions/                 YOLO inference outputs per detector variant
  CLAUDE-TRAINING.md           ML training onboarding (RunPod / Vast.ai)
  local_data.yaml              YOLO config consumed by ultralytics
archive/                       read-only history.
  reports_pre_rename_2026-05-10/  pre-rename CSV snapshot (M-naming migration)
  _to_review/                  staged-for-removal items from 2026-05-14 cleanup;
                               see archive/_to_review/README.md
RESEARCH.md                    primary research doc — read Section 0 first
README.md                      project overview + method evolution narrative
report_10Mei2026.md            953-tree benchmark (M01 champion analysis)
AGENTS.md                      thin pointer → CLAUDE.md
NAMING.md                      Mxx naming-convention table
```

## algorithms/ Package

Each `algorithms/*.py` exports `predict(detections: list[dict], params: dict) -> dict[str, int]`.

- `detections`: list of `{"class": "B1"–"B4", "x_norm": float, "y_norm": float, "side_index": int}`
- `params`: from `M17_selector_regime.load_params()` (reads CSV from reports/)
- Returns: `{"B1": int, "B2": int, "B3": int, "B4": int}`

`M17_selector_regime.load_params()` must be called once and the result passed to all algo `predict()` calls. `M12_selector_overrides` internally calls `M17_selector_regime` — don't double-call separately.

Algo ranked by 953-tree Acc±1 (see `algorithms/__init__.py` and `NAMING.md` for full table). For new code, import via `from algorithms.M01_selector_b2b3 import predict` or whichever ID needed.

## JSON Schema (per tree)

```json
{
  "tree_id": "20260422-DAMIMAS-001",
  "split": "train",
  "images": {"sisi_1": {"annotations": [{"class_name": "B3", "bbox_yolo": [...], "box_index": 0}]}},
  "bunches": [{"bunch_id": 1, "class": "B3", "appearance_count": 2, "appearances": [...]}],
  "summary": {"total_unique_bunches": 8, "by_class": {"B1": 1, "B2": 2, "B3": 5, "B4": 0}}
}
```

`summary.by_class` is the dedup ground truth.

## Decision Metric

- **Counting (primary):** % trees within ±1 error per class.
- **Mandatory metrics for every benchmark run and report:**
  1. **Per-class MAE** (`MAE_B1`, `MAE_B2`, `MAE_B3`, `MAE_B4`) — mean absolute error for each maturity class.
  2. **Macro class-MAE** — unweighted average of the four per-class MAEs.
  3. **Exact-profile accuracy** — percentage of trees where the predicted vector `[B1,B2,B3,B4]` exactly matches the ground-truth vector (zero error in all classes simultaneously).
  4. **Total-count MAE** — MAE of the sum `B1+B2+B3+B4` per tree.
  5. **Total ±1 accuracy** — percentage of trees where the predicted total count is within ±1 of the ground-truth total.
  6. **Per-class mean error (bias)** — signed mean error per class, indicating systematic over/under-count direction.
- Secondary: Mean Total Error.
- **YOLO model (legacy AR29 baseline):** mAP50-95 (not mAP@0.5). Bootstrap 95% CI required vs AR29; gap <0.005 = noise.

## What NOT to Do

**Algorithmic constraint (hard):**
- ❌ Siamese / CNN embedding (training)
- ❌ MultiViewAggregator with neck features (learned features)
- ❌ MLP on bbox features (training)
- ❌ Learned thresholds via backprop
- ❌ Strict matching (Hungarian/graph/cluster) on TXT labels — broken by coordinate noise

**Don't re-run** (per RESEARCH.md §30.4): imgsz 800, focal loss, naive oversampling, two-stage classifiers (DINOv2/EfficientNet/CORAL), YOLOv9e, RT-DETR-L, RF-DETR, SGD/AdamW sweep, label_smoothing, long brute-force grids.

**Don't pursue further grid search past M12.** 3/228 remaining failures are likely irreducible without cross-view embeddings (excluded by constraint).
