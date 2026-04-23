# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running Scripts

```bash
# GT counting semua pohon (953) — no GPU, ~1 min
python scripts/count_all_trees.py

# JSON-05 + JSON-01 audit (228 pohon JSON saja)
python scripts/count_gt_vs_naive.py

# Dedup research heuristik pada 228 pohon JSON (v1 grid search)
python scripts/dedup_research.py

# Dedup research v2 — visibility + adaptive ridge + ensemble stack
python scripts/dedup_research_v2.py

# Dedup research v3 — learned thresholds dari _confirmedLinks, then predict
python scripts/dedup_research_v3.py

# Final dedup all 953 pohon (228 JSON-validated + 717 non-JSON)
python scripts/dedup_all_trees_final.py

# Non-JSON dedup comparison & report generation
python scripts/dedup_nonjson_compare.py
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
  count_all_trees.py          GT counting semua 953 pohon (228 JSON-dedup + 725 TXT-naive)
  count_gt_vs_naive.py        JSON-05 + JSON-01 audit (228 pohon JSON)
  dedup_research.py           v1: Grid search heuristik (corrected, visibility, graph, cluster)
  dedup_research_v2.py        v2: Adaptive ridge + ensemble stack
  dedup_research_v3.py        v3: Learned thresholds dari _confirmedLinks + Ridge/Hungarian
  dedup_all_trees_final.py    Final run: semua metode pada 953 pohon
  dedup_nonjson_compare.py    Validasi & report non-JSON dedup
reports/
  full_gt_count/              count_all_trees.csv, summary_by_domain.csv, summary_by_split.csv, summary.md
  json_05/                    count_mae_gt.csv — GT vs naive sum per tree (228 pohon)
  label_audit/                per_class_inconsistency.csv, leak_pairs.csv, inconsistent_bunches.csv
  dedup_research/             method_comparison.csv, best_method_details.csv, summary.md
  dedup_research_v2/          method_comparison_v2.csv, error_analysis_v2.csv, summary_v2.md
  dedup_research_v3/          method_comparison_v3.csv, error_analysis_v3.csv, learned_thresholds.json, summary_v3.md
  dedup_all_trees_final/      all_trees_dedup_counts.csv, json_228_accuracy.csv, nonjson_725_*.csv
  nonjson_dedup_compare/      all_trees_dedup_counts.csv, json_accuracy_validation.csv, nonjson_counts_by_method.csv
  nonjson_dedup_report.md     Laporan utama non-JSON dedup
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
| Dedup v1 | Heuristic grid search (corrected, visibility, graph, cluster) | **DONE** | Best: corrected, 90.8% ±1 acc |
| Dedup v2 | Visibility + adaptive ridge + ensemble stack | **DONE** | Best: visibility, 92.1% ±1 acc |
| Dedup v3 | Learned thresholds + per-class Ridge | **DONE** | Best: per_class_ridge, 90.8% ±1 acc |
| Dedup Final | All methods on 953 trees (228 JSON + 717 non-JSON) | **DONE** | corrected & visibility viable; graph/cascade fail on TXT |
| JSON-02/03/04 | Retrain paths | Deferred | Run only if needed |

**JSON-01 verdict:** `H-LBL-1 FALSIFIED` — B2/B3 label noise is not the ceiling; B2↔B3 confusion is irreducible visual ambiguity.

**Full GT counting (all 953 trees):** `scripts/count_all_trees.py` — DONE. Output di `reports/full_gt_count/`.

**Dedup research verdict:** Heuristic bbox ceiling ≈ **92%** (visibility method). Graph matching, cascade, and clustering **fail** on noisy TXT labels (< 20% accuracy). To break past 92% need embedding-based cross-view matching.

## Non-JSON Dedup Pipeline (717 pohon tanpa JSON)

Pohon non-JSON hanya memiliki YOLO TXT labels (prediksi model), bukan anotasi manual. TXT labels memiliki **noise koordinat** dan **noise klasifikasi** (B2↔B3 sering tertukar), sehingga metode yang bergantung pada matching ketat gagal.

### Validasi Akurasi pada 228 Pohon JSON

| Method | Mean MAE | Acc ±1 | Mean Total Err | Verdict |
|--------|---------:|-------:|---------------:|---------|
| **visibility** | 0.2719 | **92.11%** | 1.09 | **Recommended** |
| **corrected** | 0.2851 | 90.79% | 1.14 | **Recommended** |
| hungarian_match | 1.0976 | 18.86% | 4.39 | Undercount ringan |
| cascade_match | 1.7730 | 4.39% | 7.09 | Undercount parah |
| learned_graph | 1.8202 | 4.39% | 7.28 | Undercount parah |
| feature_cluster | 1.8728 | 3.51% | 7.49 | Undercount parah |
| naive | 2.1294 | 2.63% | 8.52 | Baseline (jangan pakai) |

### Hasil Dedup pada 717 Pohon Non-JSON

| Method | Total Count | Rasio vs Naive | Status |
|--------|------------:|---------------:|--------|
| naive | 13,665 | 100.0% | Overcount |
| **corrected** | 7,779 | **57.4%** | **Valid** |
| **visibility** | 7,510 | **55.7%** | **Valid** |
| hungarian_match | 5,775 | 42.6% | Undercount |
| learned_graph | 2,821 | 23.9% | Undercount parah |
| cascade_match | 2,650 | 22.5% | Undercount parah |
| feature_cluster | 2,411 | 20.6% | Undercount parah |

**Rasio dedup yang benar ≈ 56%** (dari verifikasi JSON-05: naive / 1.788). corrected (57.4%) dan visibility (55.7%) sangat mendekati ground truth.

### Rekomendasi Produksi

| Skenario | Rekomendasi |
|----------|-------------|
| Pohon non-JSON (717) | Gunakan **`corrected`** atau **`visibility`**. Rasio ~55–57% sudah terverifikasi |
| Pohon ber-JSON (228) | Gunakan **`visibility`** (92.1% ±1) atau `corrected` (90.8% ±1) |
| Hindari | `learned_graph`, `cascade_match`, `feature_cluster` untuk TXT labels |

## MultiViewAggregator & Next Step

**Apa itu MultiViewAggregator?**

`MultiViewAggregator` adalah pipeline inference-based counting yang menggunakan **feature embedding dari YOLO neck** (bukan hanya bbox koordinat) untuk melakukan cross-view matching. Cara kerjanya:
1. Deteksi setiap sisi menggunakan model YOLO (AR29 weights)
2. Capture feature map dari neck (pre-head layer) via forward hook
3. ROI-pool embedding per bbox
4. Link deteksi antar-sisi dengan greedy clustering berdasarkan **cosine similarity embedding** + intra-class constraint
5. Output = unique bunch count per kelas per pohon

Detail implementasi ada di `RESEARCH.md` Section 23. Skrip evaluasi referensi: `scripts/eval_multiview.py`.

**Next step apa sekarang?**

Berdasarkan hasil dedup research, **heuristic bbox ceiling ≈ 92%**. Untuk tembus > 95% butuh signal visual yang lebih kuat dari sekadar koordinat bbox. Dua jalur parallel:

1. **Implement & evaluasi MultiViewAggregator** — gunakan AR29 weights, jalankan pada 228 pohon JSON, bandingkan count MAE vs visibility/corrected. Target: beat 92.1% ±1 accuracy.
2. **Siamese/CNN embedding pada bbox crops** — kalau neck embedding tidak cukup, latih network terpisah pada crop bbox untuk cross-view similarity.

Jika MultiViewAggregator tidak beat heuristic ceiling setelah tuning, rekomendasi = pivot ke retrain YOLO dengan data JSON-annotated untuk meningkatkan kualitas TXT labels, atau pertimbangkan task reframing (3-class: B1, B23, B4).

## Decision Metric

`mAP50-95` is the primary metric (not `mAP@0.5`). All comparisons must include bootstrap 95% CI vs AR29. A gap < 0.005 mAP50-95 is noise.

For **counting pipeline**, primary metric = **% trees within ±1 error per class** (secondary: Mean MAE, Mean Total Error).

## What NOT to re-run

Per RESEARCH.md Section 30.4 — do not re-attempt: imgsz 800, focal loss, naive oversampling, two-stage classifiers (DINOv2/EfficientNet/CORAL), YOLOv9e, RT-DETR-L, RF-DETR, SGD/AdamW sweep, label_smoothing, long brute-force runs.

**New:** Do NOT tune heuristic bbox parameters further — ceiling ≈ 92% already reached. Graph/cascade/cluster methods on TXT labels are fundamentally broken due to coordinate noise.
