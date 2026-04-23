# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context

**Task:** Multi-view oil palm fruit bunch counting. Convert detections from 4–8 photo sides per tree into **unique bunch count per maturity class**. Naive sum overcounts ~78.8% (same bunch seen across sides). Read `RESEARCH.md` Section 0 (esp. 0.6–0.9) before deep work.

**Constraint:** **100% algorithmic / heuristic only.** No training, embeddings, backprop, or learned matchers. All methods must be deterministic and parameter-free (no gradient computation).

**Dataset:** DAMIMAS (854) + LONSUM (99) = **953 trees**. 228 have JSON GT (multi-view bunch links); 725 only have YOLO TXT predictions. Mostly 4 sides/tree, 45 newest have 8. Images 960×1280 JPEG. Classes ordinal B1→B4 (B1 ripest/lowest, B4 spiny/topmost). **Core hard problem: B2↔B3 visually ambiguous** (irreducible per JSON-01 audit, label noise = 0%).

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

# Final inference + comparison
python scripts/dedup_all_trees_final.py    # all methods on 953 trees
python scripts/dedup_nonjson_compare.py    # non-JSON validation + report
```

## Current Best (as of 2026-04-24)

Benchmark: 228 JSON trees, **Acc ±1 per class per tree** (primary), MAE + Mean Total Error (secondary).

| Rank | Method | Acc ±1 | MAE | Notes |
|---:|---|---:|---:|---|
| 1 | `v9_selector` | **98.68%** | **0.2533** | Only 3/228 trees still fail |
| 2 | `v9_b2_median_v6` | 96.49% | 0.2588 | |
| 3 | `v6_selector` | 96.49% | 0.2632 | v9 default backbone |
| 4 | `v9_median_strong5` | 95.18% | 0.2390 | |
| 5 | `stacking_bracketed_v7` | 94.30% | 0.2643 | |

**Recommendations:**
- **JSON trees (228) with GT** → `v9_selector`
- **Non-JSON trees (725) without GT** → prefer `hybrid_vis_corr`, `side_coverage`, `stacking_density_v7`, `best_visibility_grid`, or `visibility`. Don't assume v9_selector wins here — its benchmark is JSON-only.

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
| v9 | `v9_selector` | **98.68%** | narrow high-confidence overrides on v6 default |

**Key takeaway:** strict matching (Hungarian, graph, cluster) **fails** on noisy TXT labels (<20% accuracy). Adaptive statistical correction + regime-routing wins. B2↔B3 ambiguity is the irreducible ceiling, not label noise.

## Non-JSON Pipeline (725 trees, TXT-only)

TXT labels have coordinate + classification noise (B2↔B3 swaps). Validation on 228 JSON trees (older method comparison):

| Method | Acc ±1 | Verdict |
|---|---:|---|
| `visibility` | 92.11% | Recommended |
| `corrected` | 90.79% | Recommended |
| `hungarian_match` | 18.86% | Mild undercount |
| `cascade_match` / `learned_graph` / `feature_cluster` | <5% | **Broken on TXT — do not use** |
| `naive` | 2.63% | Baseline only |

Verified dedup ratio ≈ 56% (from JSON-05: naive ÷ 1.788). On 725 non-JSON trees: `corrected` → 57.4%, `visibility` → 55.7% (both valid). v7+ methods may also apply but unvalidated.

## Repository Layout

```
json/                  228 JSON files (multi-view bunch-linking GT)
dataset/
  data.yaml            YOLO config (path: /workspace/dataset)
  images/{train,val,test}/
  labels/{train,val,test}/
algorithms/            standalone algo modules — each exports predict(detections, params) -> dict
  __init__.py          ranked performance table (read this for algo selection)
  v9_selector.py       CURRENT BEST — imports v6_selector + 3 specialist algos
  v6_selector.py       backbone + load_params() (reads reports/dedup_research_v5/...)
  *.py                 one algo = one file, all deterministic, no training
scripts/               see "Running Scripts" — count_*, dedup_research_v1..v9, dedup_*_final
  dedup_all_953.py     run all methods on all 953 trees (newer than dedup_all_trees_final)
reports/<script>/      every script writes its outputs here
contract-work/         validation contracts, v4 analysis, dry-run + algorithmic-advancement reports
RESEARCH.md            primary research doc — read Section 0 first
README.md              project overview + method evolution narrative
AGENTS.md              agent configuration
tod.md                 working notes
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
