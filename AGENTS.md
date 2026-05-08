# AGENTS.md

Operational instructions for AI agents working in this repo. Mirror of `CLAUDE.md` — keep both in sync.

## Project Context

**Task:** Multi-view oil palm fruit bunch counting. Convert detections from 4–8 photo sides per tree into **unique bunch count per maturity class**. Naive sum overcounts ~83.4% (same bunch seen across sides; factor 1.834 from 727-tree GT). Read `RESEARCH.md` Section 0 (esp. 0.6–0.9) before deep work.

**Constraint:** **100% algorithmic / heuristic only.** No training, embeddings, backprop, or learned matchers. All methods must be deterministic and parameter-free (no gradient computation).

**Dataset:** DAMIMAS (854) + LONSUM (99) = **953 trees**. JSON GT canonical: `json_05 Mei 2026/` = **882 unique trees** (dedup'd from `05 Mei 2026/` raw output, 1 file per `tree_name`, varietas correct). Legacy `json/` = **228 trees** (subset, retained for historical benchmark reproduction). 71 trees still missing JSON (45 8-sided DAMIMAS_A21B_0810–0854 + 19 LONSUM_A21A_0056–0074 + 7 DAMIMAS scattered) — TXT predictions exist in `dataset/labels/`, JSON GT pending re-generation via `tools_sawit/` app once annotator finishes. See `json_05 Mei 2026/_MISSING.md`. Mostly 4 sides/tree, 45 have 8. Images 960×1280 JPEG. Classes ordinal B1→B4. **Core hard problem: B2↔B3 visually ambiguous** (irreducible per JSON-01 audit, label noise = 0%).

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

# Cross-dataset benchmark (228/478/727/882 trees — fresh re-run 2026-05-08)
# Results in reports/benchmark_228/, benchmark_478/, benchmark_727/, benchmark_882/

# Final inference + comparison (all 953 trees)
python scripts/dedup_all_953.py             # all 16 methods on all 953 trees (newer)
python scripts/dedup_all_trees_final.py     # all methods on 953 trees
python scripts/dedup_nonjson_compare.py     # non-JSON validation + report
```

## Current Best (as of 2026-05-08)

Fresh benchmark re-run (2026-05-08) on all 4 archive snapshots. **Primary benchmark = 882 trees** (`json_05 Mei 2026/`). 228-tree numbers are historical reference only.

### Acc ±1 on 228 trees (historical, v9 development set)

| Rank | Method | Acc ±1 | Notes |
|---:|---|---:|---|
| 1 | `v9_selector` | **97.37%** | Narrow overrides on v6 — **overfits 228** |
| 2 | `b2_median_v6` | 96.05% | v9 variant |
| 3 | `v6_selector` | 96.05% | v9 backbone |
| 4 | `stacking_bracketed` | 94.30% | v7 best |
| 5 | `stacking_density` | 94.30% | v7 |
| 6 | `entropy_modulated` | 94.30% | v8 — ties v7 |
| 7 | `adaptive_corrected` | 93.86% | v5 |
| 8 | `b2_b4_boosted` | 92.54% | v8 specialist |
| 9 | `best_visibility_grid` | 92.54% | v5 |
| 10 | `v2_visibility` | 92.54% | v2 |
| 11 | `v1_corrected` | 90.79% | v1 baseline |

### Cross-Dataset Regression (Acc ±1, fresh re-run 2026-05-08)

Sources: `reports/benchmark_228/`, `reports/benchmark_478/`, `reports/benchmark_727/`, `reports/benchmark_882/`.

| Method | 228 | 478 | 727 | **882** | Delta 228→882 |
|---|---:|---:|---:|---:|---:|
| `v9_selector` | 97.37% | 92.68% | 89.27% | 88.78% | −8.59 pp |
| `v9_b2_median_v6` | 96.05% | 92.68% | 89.00% | 88.78% | −7.27 pp |
| `v6_selector` | 96.05% | 91.84% | 88.86% | 88.55% | −7.50 pp |
| `v8_entropy_modulated` | 94.30% | 91.63% | 88.86% | 88.78% | −5.52 pp |
| `v7_stacking_bracketed` | 94.30% | 91.84% | 88.45% | 88.44% | −5.86 pp |
| `v7_stacking_density` | 94.30% | 91.84% | 88.45% | 88.44% | −5.86 pp |
| `v5_adaptive_corrected` | 93.86% | 89.96% | 86.11% | 86.28% | −7.58 pp |
| `v8_b2_b4_boosted` | 92.54% | 91.00% | 87.62% | 88.21% | −4.33 pp |
| `v2_visibility` | 92.54% | 90.38% | **89.41%** | **89.34%** | **−3.20 pp** |
| `v5_best_visibility` | 92.54% | 90.38% | **89.41%** | **89.34%** | **−3.20 pp** |
| `v1_corrected` | 90.79% | 89.12% | 87.90% | 88.21% | **−2.58 pp** |

**Key regression findings:**
- `v2_visibility` and `v5_best_visibility` most stable — best at 882 (89.34%), smallest delta
- `v1_corrected` most stable overall (−2.58 pp) — simple generalizes best
- `v9_selector` drops 8.59 pp — narrow overrides overfit 228-tree patterns
- All methods land 86–89% at 882 trees (no catastrophic drop)

**Recommendations:**
- **Historical 228-tree set** → `v9_selector` (97.37%)
- **Full 882-tree canonical benchmark** → `v2_visibility` or `v5_best_visibility` (89.34%)
- **Missing JSON (71 trees)** → `v5_best_visibility` or `v7_stacking_bracketed`

**v9 logic (regime overrides on top of v6_selector):**
1. default → `v6_selector`
2. `b4_only_overlap` → `v7_stacking_bracketed`
3. `classaware_compact_lowb4` → `v8_b2_b4_boosted`
4. `b3b4_only_lowtotal` → `v8_floor_anchor_50`
5. `dense_allside_moderatedup` → `v8_b2_b4_boosted`

## Method Evolution (Why v9 Wins)

| Gen | Best Method | Acc ±1 | Lesson |
|---|---|---:|---|
| naive | — | very poor | overcount ~78.8% baseline |
| v1 | `corrected` | 90.79% | global divisor already beats naive hugely |
| v2 | `visibility` | 92.11% | bbox geometry / position matters |
| v3 | `per_class_ridge` | 90.79% | learned-from-link thresholds didn't break ceiling |
| v4 | `visibility` | 92.11% | adding HSV + Hungarian didn't beat v2 |
| v5 | `adaptive_corrected` | 93.86% | adaptive divisor + class-aware family — first stable >93% |
| v6 | `v6_selector` | **96.49%** | **turning point** — no single global rule wins; route per regime |
| v7 | `stacking_bracketed` | 94.30% | stacking/density family strong but loses to v6 |
| v8 | `stacking_bracketed_v7` | 94.30% | entropy/per-side signals add nothing |
| v9 | `v9_selector` | **97.37%** (228) / 88.78% (882) | narrow overrides on v6 — best on 228, overfits at scale |

**Key takeaway:** strict matching (Hungarian, graph, cluster) **fails** on noisy TXT labels (<20% accuracy). Adaptive statistical correction + regime-routing wins on 228-tree set. At scale (882 trees), simpler methods (`v2_visibility`, `v1_corrected`) generalize best. B2↔B3 ambiguity is the irreducible ceiling, not label noise.

## Missing JSON Pipeline (71 trees, TXT-only)

With 882 of 953 trees now having JSON GT, only **71 trees** still lack ground truth. TXT labels have coordinate + classification noise (B2↔B3 swaps).

Best methods on 882-tree canonical benchmark (most stable = recommended for unvalidated trees):

| Method | Acc ±1 @882 | Delta 228→882 | Verdict |
|---|---:|---:|---|
| `v2_visibility` | **89.34%** | −3.20 pp | Most stable, recommended |
| `v5_best_visibility` | **89.34%** | −3.20 pp | Most stable, recommended |
| `v7_stacking_bracketed` | 88.44% | −5.86 pp | Good generalization |
| `v9_selector` | 88.78% | −8.59 pp | Overfits 228, avoid for unvalidated |

Verified dedup ratio ≈ 56% (from JSON-05: naive ÷ 1.788). See `reports/nonjson_dedup_report.md`.

## Repository Layout

```
json_05 Mei 2026/      882 JSON GT files (current canonical, 1 per tree_name)
                       _MISSING.md lists 71 trees pending re-annotation
json/                  228 JSON files (legacy subset, used by older scripts)
05 Mei 2026/           Raw export from tools_sawit/ app (1433 files, 5 folders;
                       see json_05 Mei 2026/ for the dedup'd canonical version)
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
  v9_selector.py       best on 228 — imports v6_selector + 3 specialist algos
  v6_selector.py       backbone + load_params() (reads reports/dedup_research_v5/...)
  *.py                 one algo = one file, all deterministic, no training
scripts/               see "Running Scripts" — count_*, dedup_*, benchmark_*, generate_*
  dedup_all_953.py     run all 16 methods on all 953 trees
reports/<script>/      every script writes its outputs here
reports/benchmark_228/       Acc±1 on 228-tree archive (fresh re-run 2026-05-08)
reports/benchmark_478/       Acc±1 on 478-tree archive
reports/benchmark_727/       Acc±1 on 727-tree archive
reports/benchmark_882/       Acc±1 on 882-tree canonical (PRIMARY BENCHMARK)
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
- `params`: from `v6_selector.load_params()` (reads CSV from reports/)
- Returns: `{"B1": int, "B2": int, "B3": int, "B4": int}`

`v6_selector.load_params()` must be called once and the result passed to all algo `predict()` calls. `v9_selector` internally calls `v6_selector` — don't double-call v6 separately.

Algo ranked by JSON-228 Acc±1 (see `algorithms/__init__.py` for full table). For new code importing these, use `from algorithms.v9_selector import predict` or whichever rank is needed.

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

- **Counting (primary):** % trees within ±1 error per class. Secondary: MAE, Mean Total Error.
- **YOLO model (legacy AR29 baseline):** mAP50-95 (not mAP@0.5). Bootstrap 95% CI required vs AR29; gap <0.005 = noise.

## What NOT to Do

**Algorithmic constraint (hard):**
- ❌ Siamese / CNN embedding (training)
- ❌ MultiViewAggregator with neck features (learned features)
- ❌ MLP on bbox features (training)
- ❌ Learned thresholds via backprop
- ❌ Strict matching (Hungarian/graph/cluster) on TXT labels — broken by coordinate noise

**Don't re-run** (per RESEARCH.md §30.4): imgsz 800, focal loss, naive oversampling, two-stage classifiers (DINOv2/EfficientNet/CORAL), YOLOv9e, RT-DETR-L, RF-DETR, SGD/AdamW sweep, label_smoothing, long brute-force grids.

**Don't pursue further grid search past v9.** 3/228 remaining failures are likely irreducible without cross-view embeddings (excluded by constraint).
