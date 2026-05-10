# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context

**Task:** Multi-view oil palm fruit bunch counting. Convert detections from 4–8 photo sides per tree into **unique bunch count per maturity class**. Naive sum overcounts ~83.4% (same bunch seen across sides; factor 1.834 from 727-tree GT). Read `RESEARCH.md` Section 0 (esp. 0.6–0.9) before deep work.

**Constraint:** **100% algorithmic / heuristic only.** No training, embeddings, backprop, or learned matchers. All methods must be deterministic and parameter-free (no gradient computation).

**Dataset:** DAMIMAS (854) + LONSUM (99) = **953 trees**. **Canonical JSON GT: `Brand-New-Dataset-YOLO/json/` = 953 trees (COMPLETE, no missing — created 2026-05-09).** This supersedes all earlier snapshots:
- `Brand-New-Dataset-YOLO/` (953, full) — current canonical, contains `images/`, `labels/`, `json/`, `data.yaml`
- `json_05 Mei 2026/` (882, legacy) — kept for historical reference only
- `json/` (228, legacy) — kept for v9 dev-set reproduction
- `archive/` (228/478/727 snapshots) — read-only history

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

# JSON-05 + JSON-01 audit (228 JSON trees)
python scripts/count_gt_vs_naive.py

# Dedup research generations (each writes to reports/dedup_research_vN/)
python scripts/dedup_research.py       # v1: grid search (corrected, visibility, graph, cluster)
python scripts/dedup_research_v2.py    # v2: visibility + adaptive ridge + ensemble
python scripts/dedup_research_v3.py    # v3: thresholds from _confirmedLinks
python scripts/dedup_research_v4.py    # v4: pixel-aware HSV + Mahalanobis + Hungarian
python scripts/dedup_research_v5.py    # v5: adaptive density-corrected
python scripts/dedup_v5_focused.py     # v5 focused variant
python scripts/dedup_research_v6.py    # v6: regime selector (96.49%)
python scripts/dedup_research_v7.py    # v7: stacking + density family
python scripts/dedup_research_v8.py    # v8: entropy + per-side distribution
python scripts/dedup_research_v9.py    # v9: narrow regime overrides on v6 (CURRENT BEST 98.68%)

# Multi-dimensional benchmark + method reports
python scripts/benchmark_multidim.py         # 4-dim evaluation: accuracy, speed, robustness, domain
python scripts/generate_method_reports.py    # per-method breakdown reports → reports/methods/

# PRIMARY benchmark — full 953-tree Brand-New-Dataset-YOLO (canonical, 2026-05-10)
python scripts/dedup_brand_new_953.py       # → reports/dedup_brand_new_953/

# Legacy cross-dataset benchmarks (228/478/727/882, fresh re-run 2026-05-08)
# Kept for historical reference — Brand-New-Dataset-YOLO supersedes these
# Results in reports/benchmark_228/, benchmark_478/, benchmark_727/, benchmark_882/

# Legacy inference scripts (228 JSON + 725 TXT, no longer canonical)
python scripts/dedup_all_953.py             # all 16 methods on all 953 trees (legacy mix)
python scripts/dedup_all_trees_final.py     # all methods on 953 trees (legacy)
python scripts/dedup_nonjson_compare.py     # non-JSON validation (legacy, no missing JSON anymore)
```

## Current Best (as of 2026-05-10)

**PRIMARY BENCHMARK: 953-tree Brand-New-Dataset-YOLO** (canonical, full GT). Earlier 228/478/727/882 numbers are historical only.

**Naming convention (effective 2026-05-10):** all methods use `M<NN>_<family>_<descriptor>`. See `NAMING.md` for full mapping table from old names. IDs are stable — assigned once, never re-shuffled.

### Acc ±1 on 953 Brand-New-Dataset-YOLO trees (PRIMARY)

| Rank | Method | Acc ±1 | MAE | n_fail |
|---:|---|---:|---:|---:|
| 1 | `M01_selector_b2b3` | **86.67%** | **0.3982** | 127 |
| 2 | `M02_selector_trifurc` | 86.67% | 0.3987 | 127 |
| 3 | `M03_blend_geometric` | 86.15% | 0.3961 | 132 |
| 4 | `M04_blend_floor_clamped` | 86.04% | 0.4050 | 133 |
| 5 | `M05_blend_vis_divide` | 86.04% | 0.4077 | 133 |
| 6 | `M06_weight_visibility` | 85.94% | 0.3956 | 134 |
| 7 | `M07_weight_coverage` | 85.94% | 0.3930 | 134 |
| 8 | `M08_divide_density_vis` | 85.94% | 0.4024 | 134 |
| 9 | `M09_median_strong5` | 85.73% | 0.4006 | 136 |
| 10 | `M10_entropy_divide` | 84.78% | 0.4507 | 145 |
| 11 | `M11_median_b2` | 84.78% | 0.4288 | 145 |
| 12 | `M12_selector_overrides` | 84.68% | 0.4413 | 146 |
| 13 | `M13_stack_bracket` | 84.58% | 0.4279 | 147 |
| 14 | `M14_stack_density` | 84.58% | 0.4274 | 147 |
| 15 | `M15_divide_global` | 84.37% | 0.4158 | 149 |
| 16 | `M16_boost_b2b4` | 84.37% | 0.4111 | 149 |
| 17 | `M17_selector_regime` | 84.26% | 0.4436 | 150 |
| 18 | `M18_entropy_stack` | 84.78% | 0.4507 | 145 |
| 19 | `M19_divide_adaptive` | 82.58% | 0.4599 | 166 |
| 20 | `M20_weight_visibility_grid` | 80.80% | 0.4596 | 183 |
| 21 | `M23_agree_side` | 80.80% | 0.4273 | 183 |
| 22 | `M27_weight_visibility_adaptive` | 80.27% | 0.4790 | 188 |
| 23 | `M24_weight_class_aware` | 70.93% | 0.5456 | 277 |
| 24 | `M22_anchor_floor50` | 69.99% | 0.4525 | 286 |
| 25 | `M25_consensus_multi` | 25.29% | 0.9121 | 712 |
| 26 | `M26_median_per_side` | 25.29% | 0.9121 | 712 |
| 27 | `M28_baseline_match_strict` | 5.98% | 1.8114 | 896 |
| 28 | `M29_baseline_naive_sum` | 3.99% | 2.2804 | 915 |
| 29 | `M21_ordinal_b3` | 0.73% | 3.5842 | 946 |

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
- `M01_selector_b2b3` **NEW TOP** at 953 (86.67%, MAE 0.3982) — selector trifurc + B2↔B3 split correction, validated held-out
- `M06_weight_visibility` most stable from 228 (−6.60 pp) — simple generalizes best
- `M15_divide_global` second-most stable (−6.42 pp)
- `M12_selector_overrides` drops **12.69 pp** at 953 — narrow overrides catastrophically overfit 228 dev set
- All complex selectors (`M17`, `M13`, `M10`, `M12`) regress 9–13 pp from 228 → 953
- All methods land 82–86% at 953 trees (no catastrophic drop, but ceiling lower than expected)

**Recommendations (UPDATED 2026-05-10 post-iter11):**
- **Production / full 953-tree dataset** → `M01_selector_b2b3` (86.67%, MAE 0.3982) — current top, validated held-out, no overfit
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
Brand-New-Dataset-YOLO/  953 trees COMPLETE (canonical, created 2026-05-09)
  data.yaml              YOLO config (4 classes B1–B4)
  images/{train,val,test}/  source images
  labels/{train,val,test}/  YOLO TXT labels
  json/                     953 JSON GT files (1 per tree_name) — PRIMARY GT
json_05 Mei 2026/      882 JSON GT files (legacy, superseded by Brand-New-Dataset-YOLO)
json/                  228 JSON files (legacy subset, used by older scripts)
05 Mei 2026/           Raw export from tools_sawit/ app
archive/               Snapshot archives (read-only, do not modify)
  json_22 April 2026/  228-tree snapshot (= json/ root copy)
  json_28 April 2026/  478-tree snapshot
  json_30 April 2026/  727-tree snapshot (legacy benchmark)
  05 Mei 2026 raw/     raw export before dedup
tools_sawit/           Web app (vanilla JS) used to produce JSON GT.
                       Schema v2: filename = tree_name.json, varietas derived
                       per-tree from name prefix. See tools_sawit/README.md.
dataset/
  data.yaml            YOLO config (path: /workspace/dataset)
  images/{train,val,test}/
  labels/{train,val,test}/
algorithms/            standalone algo modules — each exports predict(detections, params) -> dict
  __init__.py          ranked performance table (read this for algo selection)
  M01_selector_b2b3.py current production (953 best, 86.67%)
  M12_selector_overrides.py best on 228 — imports M17_selector_regime + 3 specialists
  M17_selector_regime.py    regime backbone + load_params() (reads reports/dedup_research_v5/...)
  M<NN>_*.py            one algo = one file, all deterministic, no training
scripts/               see "Running Scripts" — count_*, dedup_*, benchmark_*, generate_*
  dedup_brand_new_953.py  PRIMARY benchmark — runs all 16 methods on Brand-New-Dataset-YOLO
  dedup_all_953.py        legacy: 228 JSON + 725 TXT (no longer needed, kept for repro)
  build_brand_new_dataset.py  builds Brand-New-Dataset-YOLO/ from 05 Mei 2026/ source
reports/<script>/      every script writes its outputs here
reports/dedup_brand_new_953/ Acc±1 on full 953 canonical (PRIMARY BENCHMARK)
reports/benchmark_228/       Acc±1 on 228-tree archive (legacy, dev set)
reports/benchmark_478/       Acc±1 on 478-tree archive (legacy)
reports/benchmark_727/       Acc±1 on 727-tree archive (legacy)
reports/benchmark_882/       Acc±1 on 882-tree (legacy, superseded by 953)
reports/benchmark_multidim/  multi-dim benchmark (accuracy, speed, robustness, domain)
reports/methods/             per-method breakdown reports with traceability
contract-work/         validation contracts, v4 analysis, dry-run + algorithmic-advancement reports
RESEARCH.md            primary research doc — read Section 0 first
README.md              project overview + method evolution narrative + cross-dataset table
report_05Mei2026.md    882-tree benchmark results (per-method full breakdown)
AGENTS.md              agent configuration
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
