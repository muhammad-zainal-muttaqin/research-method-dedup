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

# GT validation audits (read-only)
python scripts/audit_same_side_dup.py          # bunch dgn 2+ appearance same side → reports/audit_same_side_dup/
python scripts/audit_impossible_visibility.py  # bunch melanggar geometric adjacency → reports/audit_impossible_visibility/

# Targeted GT fix (5 simple wrap-around trees, hardcoded)
python scripts/fix_wrap_around_links.py        # also: --dry-run, --only <tree_id>

# E2E / ML track (separate from heuristic — see ml-track/CLAUDE-TRAINING.md)
python scripts/build_counting_features.py
python scripts/run_counting_rf.py
python scripts/run_counting_svm.py
python scripts/run_e2e_pipeline.py          # unified harness; per-track wrappers: run_e2e_{m01,rf,svm,inference}.py
python scripts/generate_training_summary.py
python scripts/export_gt_parquet.py
```

**Runtime-library scripts (not benchmarks; imported by the primary):** `scripts/dedup_all_953.py` and `scripts/dedup_research_v5..v9.py` define method bodies that `dedup_brand_new_953.py` re-uses via `import dedup_all_953 as base` → which in turn imports `dedup_research_v6..v9`. `algorithms/M18_entropy_stack.py` also wraps `dedup_research_v8`. Do **not** call these scripts as top-level commands; they only matter as imports.

**Archived 2026-05-14 → `archive/_to_review/`:** `dedup_research_v2..v4.py`, `dedup_v5_focused.py`, `dedup_all_trees_final.py`, `dedup_nonjson_compare.py`, the four legacy benchmark report folders (`benchmark_228/478/727/882/`), the v2/v3/v4/v6/v7/v8/v9 dedup_research report folders, and one-shot migrators (`migrate_*.py`, `fix_image_filename_bug.py`, `regen_tree_id.py`, `cleanup_repo.py`, `build_brand_new_dataset.py`, `generate_hf_metadata.py`, `flatten_dataset.py`, `find_first_5.py`, `generate_sample_viz.py`). Restore by `Move-Item archive\_to_review\scripts\<file> scripts\<file>`.

## Current Best (as of 2026-05-16)

**PRIMARY BENCHMARK: 953-tree Brand-New-Dataset-YOLO** (canonical, full GT). Earlier 228/478/727/882 numbers are historical only.

**GT corrected 2026-05-15/16:** 8 wrap-around link bugs fixed + 9 over-link 8-side trees fixed (manual + rule relaxation max_dist 2→3) + 31 4-side trees auto-healed via `scripts/heal_4side_visibility.py` (largest-bbox=home heuristic). Net +~62 unique bunches across ~48 trees. All methods gained ~0.8–1.6 pp Acc±1 vs pre-fix baseline.

**Naming convention (effective 2026-05-10):** all methods use `M<NN>_<family>_<descriptor>`. See `NAMING.md` for full mapping table from old names. IDs are stable — assigned once, never re-shuffled.

### Acc ±1 on 953 Brand-New-Dataset-YOLO trees (PRIMARY, post-GT-fix 2026-05-16)

Note: on balanced 4-class dataset, Macro class-MAE ≡ flat MAE numerically. Full per-class MAE / bias / Total-count MAE breakdown in `README.md` benchmark table or `reports/dedup_brand_new_953/accuracy_953.csv`.

| Rank | Method | Acc ±1 | Macro class-MAE | Total-count MAE | Total ±1 | Exact profile | n_fail |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `M01_selector_b2b3` | **87.62%** | 0.3746 | 1.3305 | 75.13% | 27.07% | 118 |
| 2 | `M02_selector_trifurc` | 87.62% | 0.3757 | 1.3305 | 75.13% | 27.07% | 118 |
| 3 | `M03_blend_geometric` | 86.99% | 0.3767 | 1.3410 | 75.24% | 27.60% | 124 |
| 4 | `M04_blend_floor_clamped` | 86.99% | 0.3848 | 1.3421 | 74.92% | 26.55% | 124 |
| 5 | `M05_blend_vis_divide` | 86.99% | 0.3875 | 1.3463 | 74.71% | 26.02% | 124 |
| 6 | `M06_weight_visibility` | 86.88% | 0.3709 | 1.2802 | 74.19% | 26.02% | 125 |
| 7 | `M07_weight_coverage` | 86.88% | **0.3683** | **1.2760** | 74.40% | 26.55% | 125 |
| 8 | `M08_divide_density_vis` | 86.88% | 0.3801 | 1.3169 | 74.19% | 26.13% | 125 |
| 9 | `M09_median_strong5` | 86.67% | 0.3825 | 1.3956 | 73.87% | 27.91% | 127 |
| 10 | `M11_median_b2` | 85.94% | 0.4137 | 1.4995 | 71.56% | 23.50% | 134 |
| 11 | `M12_selector_overrides` | 85.94% | 0.4247 | 1.5435 | 70.09% | 22.88% | 134 |
| 12 | `M15_divide_global` | 85.94% | 0.3909 | 1.3641 | 70.30% | 23.50% | 134 |
| 13 | `M10_entropy_divide` | 85.83% | 0.4328 | 1.5677 | 67.89% | 24.55% | 135 |
| 14 | `M18_entropy_stack` | 85.83% | 0.4328 | 1.5677 | 67.89% | 24.55% | 135 |
| 15 | `M13_stack_bracket` | 85.62% | 0.4103 | 1.5068 | 70.09% | 25.81% | 137 |
| 16 | `M17_selector_regime` | 85.62% | 0.4273 | 1.5540 | 69.88% | 22.35% | 137 |
| 17 | `M14_stack_density` | 85.62% | 0.4166 | 1.5278 | 69.46% | 24.34% | 137 |
| 18 | `M16_boost_b2b4` | 85.41% | 0.3932 | 1.4239 | 73.56% | 27.28% | 139 |
| 19 | `M19_divide_adaptive` | 83.95% | 0.4441 | 1.6296 | 67.58% | 21.83% | 153 |
| 20 | `M20_weight_visibility_grid` | 82.27% | 0.4336 | 1.4722 | 67.79% | 20.15% | 169 |
| 21 | `M27_weight_visibility_adaptive` | 81.43% | 0.4544 | 1.5551 | 65.58% | 19.10% | 177 |
| 22 | `M23_agree_side` | 81.11% | 0.4069 | 1.4953 | 65.90% | 22.77% | 180 |
| 23 | `M24_weight_class_aware` | 72.40% | 0.5220 | 1.7209 | 60.55% | 12.91% | 263 |
| 24 | `M22_anchor_floor50` | 69.78% | 0.4281 | 1.4732 | 60.23% | 16.89% | 288 |
| 25 | `M25_consensus_multi` | 23.19% | 0.9037 | 3.6149 | 14.90% | 4.30% | 732 |
| 26 | `M26_median_per_side` | 23.19% | 0.9037 | 3.6149 | 14.90% | 4.30% | 732 |
| 27 | `M28_baseline_match_strict` | 5.14% | 1.8059 | 6.9906 | 4.41% | 1.99% | 904 |
| 28 | `M29_baseline_naive_sum` | 3.78% | 2.2867 | 9.1469 | 2.52% | 1.36% | 917 |
| 29 | `M21_ordinal_b3` | 0.73% | 3.5769 | 14.3075 | 0.00% | 0.00% | 946 |

Source: `reports/dedup_brand_new_953/accuracy_953.csv`.

**Surprising findings (post-GT-fix 2026-05-16):**
- **`M01_selector_b2b3` (#1)** — 87.62%, +0.95 pp vs pre-fix. Champion stable.
- **`M07_weight_coverage` lowest Macro MAE** (0.3683) and Total-count MAE (1.2760) — different optimum dari Acc±1 ranking.
- **All top methods improved 0.8–1.6 pp** after GT cleanup (62 violations + 8 wrap-around fixed).
- **`M12_selector_overrides` jumps to 85.94%** but still regress vs 228 (97.37%) — overfit confirmed.
- **Strict matching (`M28`, `M29`) tetap catastrophically fail** at scale.

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

Sources: `archive/_to_review/reports/benchmark_228/`, `archive/_to_review/reports/benchmark_478/`, `archive/_to_review/reports/benchmark_727/`, `archive/_to_review/reports/benchmark_882/`, `reports/dedup_brand_new_953/`.

| Method | 228 | 478 | 727 | 882 | **953** (post-GT-fix) | Delta 228→953 |
|---|---:|---:|---:|---:|---:|---:|
| `M12_selector_overrides` | 97.37% | 92.68% | 89.27% | 88.78% | 85.94% | −11.43 pp |
| `M11_median_b2` | 96.05% | 92.68% | 89.00% | 88.78% | 85.94% | −10.11 pp |
| `M17_selector_regime` | 96.05% | 91.84% | 88.86% | 88.55% | 85.62% | −10.43 pp |
| `M10_entropy_divide` | 94.30% | 91.63% | 88.86% | 88.78% | 85.83% | −8.47 pp |
| `M13_stack_bracket` | 94.30% | 91.84% | 88.45% | 88.44% | 85.62% | −8.68 pp |
| `M19_divide_adaptive` | 93.86% | 89.96% | 86.11% | 86.28% | 83.95% | −9.91 pp |
| `M06_weight_visibility` | 92.54% | 90.38% | 89.41% | 89.34% | **86.88%** | **−5.66 pp** |
| `M15_divide_global` | 90.79% | 89.12% | 87.90% | 88.21% | 85.94% | **−4.85 pp** |
| `M01_selector_b2b3` | — | — | — | — | **87.62%** | — (top 2026-05-16) |
| `M02_selector_trifurc` | — | — | — | — | 87.62% | — |
| `M03_blend_geometric` | — | — | — | — | 86.99% | — |
| `M04_blend_floor_clamped` | — | — | — | — | 86.99% | — |
| `M05_blend_vis_divide` | — | — | — | — | 86.99% | — |

**Key regression findings (UPDATED 2026-05-16 post-GT-fix):**
- `M01_selector_b2b3` champion at 953 (**87.62%**, Macro class-MAE 0.3746)
- `M15_divide_global` most stable from 228 (**−4.85 pp**) — simple generalizes best
- `M06_weight_visibility` second most stable (−5.66 pp)
- `M12_selector_overrides` drops **11.43 pp** at 953 — narrow overrides still overfit 228 dev set despite GT improvement
- All complex selectors (`M17`, `M13`, `M10`, `M12`) regress 8–11 pp from 228 → 953
- All methods land 83–88% at 953 trees post-GT-fix; ceiling lifted ~1 pp universally

**Recommendations (UPDATED 2026-05-16 post-GT-fix):**
- **Production / full 953-tree dataset** → `M01_selector_b2b3` (87.62%, Macro class-MAE 0.3746) — current top
- **Lowest Macro MAE** → `M07_weight_coverage` (0.3683, 86.88% Acc±1) — alternative if MAE preferred over Acc±1
- **Simplest fallback** → `M05_blend_vis_divide` (86.99%) — single-line weighted blend
- **Historical 228-tree set** → `M12_selector_overrides` (97.37%) — overfits, dev-set only
- **GT clean** — 0 violations across all audits (same-side dup + geometric visibility)

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
| **— GT-fix (2026-05-16)** | `M01_selector_b2b3` | **87.62%** (953) | same M01, but GT cleaned (8 wrap-around + 9 8-side over-link + 31 4-side healed) — universal +0.8–1.6 pp lift |

**Key takeaway:** strict matching (Hungarian, graph, cluster) **fails** on noisy TXT labels (<20% accuracy). Adaptive statistical correction + regime-routing wins on small dev sets but **overfits**. At full 953-tree scale, simpler methods (`M05_blend_vis_divide`, `M06_weight_visibility`) generalize best. B2↔B3 ambiguity is the irreducible ceiling, not label noise.

## Dataset Status (2026-05-16)

`Brand-New-Dataset-YOLO/` is the **complete, canonical 953-tree dataset**. All previously-missing 71 trees now have JSON GT. The "Missing JSON Pipeline" workflow is retired — use Brand-New-Dataset-YOLO for everything.

**GT total per class** (post-fix 2026-05-16): B1=954, B2=1,791, B3=5,067, B4=2,011, **TOTAL=9,823 unique bunches** dari 18,544 raw detections (ratio ≈ 0.53). See `reports/full_gt_count/`.

**GT validation status:** 0 violations across both audits (same-side dup + geometric visibility). 48 trees fixed total: 8 wrap-around + 9 8-side over-link + 31 4-side auto-heal. Backups di `archive/json_pre_*_2026-05-1{5,6}/`.

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
  "version": 3,
  "tree_id": "20260422-DAMIMAS-001",
  "split": "train",
  "metadata": {"date": "...", "varietas": "DAMIMAS", "fix_log": [...]},
  "images": {"sisi_1": {"side_index": 0, "bbox_count": 3, "annotations": [{"class_name": "B3", "bbox_yolo": [...], "box_index": 0}]}},
  "bunches": [{"bunch_id": 1, "class": "B3", "class_mismatch": false, "appearance_count": 2, "appearances": [{"side": "sisi_1", "side_index": 0, "box_index": 0, "class_name": "B3", "bbox_pixel": [...]}]}],
  "_confirmedLinks": [{"linkId": "lnk-0", "sideA": 0, "bboxIdA": "b0", "sideB": 1, "bboxIdB": "b0"}],
  "summary": {"total_unique_bunches": 8, "total_detections": 14, "duplicates_linked": 6, "by_class": {"B1": 1, "B2": 2, "B3": 5, "B4": 0, "other": 0}, "by_side": {...}}
}
```

`summary.by_class` is the dedup ground truth. `bunches` derived from `_confirmedLinks` via UnionFind connected components (boxes linked across sides = same physical bunch).

**`_confirmedLinks` semantics:** annotator pairs bboxes across adjacent sides; `bboxIdA` / `bboxIdB` use `b<box_index>` notation referring to position in `images.sisi_X.annotations[]`. Wrong link = bunch ke-merged dgn extra box → over-link bug; missing link = bunch ke-split → under-link bug.

## Ground-truth Validation Rules

Two structural invariants every JSON GT must satisfy. Run the audit scripts to detect violations.

### 1. Same-side uniqueness

A bunch cannot appear ≥ 2 times in the same `side_index`. Camera at one side captures each physical bunch at most once. Violation = annotator over-linked across sides → connected-components pulled an extra box into the bunch.

Detector: `scripts/audit_same_side_dup.py` → `reports/audit_same_side_dup/`.

Status (2026-05-15): 0 violations after fixing the 8 wrap-around trees reported by RA (`DAMIMAS_A21B_{0287, 0309, 0320, 0335, 0336, 0359, 0323, 0362}`).

### 2. Geometric adjacency (visibility cone)

A bunch is at one physical location on the tree. Camera at adjacent sides also sees it; camera at far sides cannot. Formal rule (updated 2026-05-16 after RA visual validation):

- **4-side trees:** max circular distance from home = **1**. ≤ 3 sides visible total. Mustahil di sisi opposite (distance 2).
  - Example: home=`sisi_1` → visible {`sisi_4`, `sisi_1`, `sisi_2`}; mustahil `sisi_3`.
- **8-side trees:** max circular distance from home = **3**. ≤ 6 sides visible total (large/prominent bunches with wider camera reach). Mustahil ≥ 7 sides.
  - Normal: home + 4 immediate neighbors (5 sides, distance ≤ 2).
  - Edge case: large bunches can reach 6 sides (distance ≤ 3).
  - Example: home=`sisi_3` → can extend to `{sisi_8, sisi_1, sisi_2, sisi_3, sisi_4, sisi_5}`; mustahil `sisi_6`, `sisi_7`.

Validity test per bunch: ada candidate `home ∈ appearance_sides` di mana semua appearance lain dalam `max_dist` hop circular. Tidak ada home valid → violation (mustahil geometri).

Severity:
- **violation** — no valid home → impossible bunch
- **warn** — valid but uses full reach (3 sides for 4-side / 6 sides for 8-side, beyond normal 2 / 4)

Detector: `scripts/audit_impossible_visibility.py` → `reports/audit_impossible_visibility/`.

Status (2026-05-16): **0 violations** after auto-heal via `scripts/heal_4side_visibility.py` (heuristic: home = appearance side dgn bbox area terbesar; drop offending side opposite). 31 trees auto-healed, +42 unique bunches added (offending boxes jadi singleton). Backups di `archive/json_pre_visibility_heal_4side_2026-05-16/`.

Earlier history: 62 violations → 53 (after 4 manual 8-side fixes) → 42 (after 8-side rule relaxation max_dist 2→3) → 0 (after 4-side auto-heal).

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
