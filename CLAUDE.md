# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running Scripts

```bash
# GT counting semua pohon (953) — no GPU, ~1 min
python scripts/count_all_trees.py

# JSON-05 + JSON-01 audit (228 pohon JSON saja)
python scripts/count_gt_vs_naive.py
```

Semua script dijalankan dari workspace root dan menulis output ke `reports/`.

## Project Context

**Task:** Multi-view oil palm fruit bunch counting using ground truth JSON labels (no model inference yet). The research direction is defined in `RESEARCH.md` — read Section 0 first, especially 0.6–0.9, before anything else.

**Dataset:** DAMIMAS (854 pohon) + LONSUM (99 pohon) = **953 pohon total**. Mayoritas 4 sisi per pohon; 45 pohon terbaru punya 8 sisi. Total ~3,992+ images (960×1280 JPEG). 4 maturity classes:
- `B1` — reddish, fully ripe, lowest position
- `B2` — half-red/black transition, above B1
- `B3` — fully black, above B2
- `B4` — smallest, spiny, black→green, topmost

**Key constraint:** B1→B4 is ordinal (maturity scale). B2↔B3 are visually ambiguous — this is the core hard problem.

## Repository Layout

```
json/               228 JSON files (one per tree) with multi-view bunch-linking
dataset/
  data.yaml         YOLO dataset config (path: /workspace/dataset)
  images/{train,val,test}/
  labels/{train,val,test}/
scripts/
  count_all_trees.py     GT counting semua 953 pohon (228 JSON-dedup + 725 TXT-naive)
  count_gt_vs_naive.py   JSON-05 + JSON-01 audit (228 pohon JSON)
reports/
  full_gt_count/    count_all_trees.csv, summary_by_domain.csv, summary_by_split.csv, summary.md
  json_05/          count_mae_gt.csv — GT vs naive sum per tree (228 pohon)
  label_audit/      per_class_inconsistency.csv, leak_pairs.csv, inconsistent_bunches.csv
```

## JSON Schema (per tree)

```json
{
  "tree_id": "20260422-DAMIMAS-001",
  "tree_name": "DAMIMAS_A21B_0001",
  "split": "train",
  "images": {
    "sisi_1": {
      "annotations": [{"class_name": "B3", "bbox_yolo": [...], "box_index": 0}, ...]
    }
  },
  "bunches": [
    {
      "bunch_id": 1,
      "class": "B3",
      "class_mismatch": false,
      "appearance_count": 2,
      "appearances": [{"side": "sisi_1", "box_index": 0, "class_name": "B3"}, ...]
    }
  ],
  "summary": {
    "total_unique_bunches": 8,
    "total_detections": 17,
    "by_class": {"B1": 1, "B2": 2, "B3": 5, "B4": 0}
  }
}
```

`summary.by_class` = GT unique bunch count per class (the dedup ground truth).  
Naive sum = sum of all `annotations` across 4 sides without dedup → ~79% overcounting on average.

## Experiment Status (as of 2026-04-23)

| Exp | Description | Status | Key result |
|-----|-------------|--------|------------|
| AR29 | YOLO11l 640 b16 standard val | **Baseline** | 0.264 mAP50-95 |
| AR34 | YOLO11l 80ep train+test | Upper bound (not fair) | 0.269 |
| JSON-05 | GT counting vs naive sum | **DONE** | 78.8% overcounting; dedup essential |
| JSON-01 | Label consistency audit | **DONE** | 0% mismatch all classes → labels clean |
| JSON-02/03/04 | Retrain paths | Deferred | Run only if needed |

**JSON-01 verdict:** `H-LBL-1 FALSIFIED` — B2/B3 label noise is not the ceiling; B2↔B3 confusion is irreducible visual ambiguity.

**Full GT counting (all 953 trees):** `scripts/count_all_trees.py` — DONE. Output di `reports/full_gt_count/`.

**Next step:** Implement `MultiViewAggregator` (Section 23 in RESEARCH.md) — inference-based counting pipeline menggunakan prediksi model (butuh weights AR29).

## Decision Metric

`mAP50-95` is the primary metric (not `mAP@0.5`). All comparisons must include bootstrap 95% CI vs AR29. A gap < 0.005 mAP50-95 is noise.

## What NOT to re-run

Per RESEARCH.md Section 30.4 — do not re-attempt: imgsz 800, focal loss, naive oversampling, two-stage classifiers (DINOv2/EfficientNet/CORAL), YOLOv9e, RT-DETR-L, RF-DETR, SGD/AdamW sweep, label_smoothing, long brute-force runs.
