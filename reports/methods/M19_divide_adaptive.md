# `v5_adaptive_corrected` — Primary Metrics Breakdown

**Implementasi:** [`algorithms/adaptive_corrected.py`](../../algorithms/adaptive_corrected.py)  
**Dataset:** 228 pohon JSON (228 baris cocok dengan `method=v5_adaptive_corrected`)  
**Raw data lengkap:** [`../benchmark_multidim/accuracy_per_tree.csv`](../benchmark_multidim/accuracy_per_tree.csv)  
**Per-method slice (filter sudah diterapkan):** [`v5_adaptive_corrected_per_tree.csv`](v5_adaptive_corrected_per_tree.csv)  
**Summary CSV:** [`../benchmark_multidim/accuracy_summary.csv`](../benchmark_multidim/accuracy_summary.csv)

Seluruh angka di bawah dihitung ulang dari `accuracy_per_tree.csv` oleh `scripts/generate_method_reports.py`.

## Primary Metrics

| Metric | Value | Derivation |
|---|---:|---|
| Macro class-MAE | **0.2774** | mean(per-class MAE) |
| Exact accuracy | **26.32%** | 60/228 pohon dengan err_B* = 0 di semua kelas |
| Total count MAE | **0.9342** | mean \|Σpred − Σgt\| per pohon |
| Total ±1 accuracy | **79.82%** | 182/228 pohon dengan \|Σpred − Σgt\| ≤ 1 |
| Acc ±1 per kelas per pohon (pelengkap) | 93.86% | 214/228 pohon dengan semua err_B* dalam ±1 |

## Per-Class MAE

Sumber: kolom `err_B*` di `accuracy_per_tree.csv` (sudah absolute).

| Class | MAE | Derivation |
|---|---:|---|
| B1 | **0.1096** | mean(err_B1) across 228 pohon |
| B2 | **0.2368** | mean(err_B2) across 228 pohon |
| B3 | **0.4342** | mean(err_B3) across 228 pohon |
| B4 | **0.3289** | mean(err_B4) across 228 pohon |

Cross-check versus [`accuracy_per_class.csv`](../benchmark_multidim/accuracy_per_class.csv):

| Class | MAE (csv) | over_count | under_count | exact | within1 | pct_within1 |
|---|---:|---:|---:|---:|---:|---:|
| B1 | 0.1096 | 0 | 0 | 203 | 228 | 100.00% |
| B2 | 0.2368 | 2 | 2 | 178 | 224 | 98.25% |
| B3 | 0.4342 | 6 | 1 | 136 | 221 | 96.93% |
| B4 | 0.3289 | 3 | 1 | 157 | 224 | 98.25% |

## Per-Class Mean Error (Bias)

Sumber: `pred_B* − gt_B*` di `accuracy_per_tree.csv`. Nilai `+` = overcount, `−` = undercount, `0` = tidak bias.

| Class | Mean Error | Derivation |
|---|---:|---|
| B1 | **+0.039** | mean(pred_B1 − gt_B1) across 228 pohon |
| B2 | **+0.061** | mean(pred_B2 − gt_B2) across 228 pohon |
| B3 | **+0.057** | mean(pred_B3 − gt_B3) across 228 pohon |
| B4 | **+0.066** | mean(pred_B4 − gt_B4) across 228 pohon |

## Kecepatan (pelengkap)

Sumber: [`speed_summary.csv`](../benchmark_multidim/speed_summary.csv) (30 repetisi × 228 pohon)

- Mean: **0.0073 ms/pohon** (136,242 pohon/detik)
- Median: 0.0073 ms
- Std: 0.0003 ms

## Robustness terhadap Noise Koordinat (pelengkap)

Sumber: [`robustness_summary.csv`](../benchmark_multidim/robustness_summary.csv)

| σ (noise_pct) | Acc ±1 | MAE | n_fail | Acc drop vs σ=0 |
|---:|---:|---:|---:|---:|
| 0% | 93.86% | 0.2774 | 14 | +0.00% |
| 5% | 93.86% | 0.2774 | 14 | +0.00% |
| 10% | 93.86% | 0.2774 | 14 | +0.00% |
| 20% | 93.86% | 0.2774 | 14 | +0.00% |

## Pohon yang Gagal (Acc±1 fail = 14)

| tree_id | split | domain | MAE | err_B1 | err_B2 | err_B3 | err_B4 |
|---|---|---|---:|---:|---:|---:|---:|
| `DAMIMAS_A21B_0244` | train | DAMIMAS | 0.50 | 0 | 0 | 0 | 2 |
| `DAMIMAS_A21B_0002` | train | DAMIMAS | 1.00 | 1 | 0 | 1 | 2 |
| `DAMIMAS_A21B_0257` | train | DAMIMAS | 0.75 | 0 | 1 | 2 | 0 |
| `DAMIMAS_A21B_0259` | train | DAMIMAS | 0.50 | 0 | 2 | 0 | 0 |
| `DAMIMAS_A21B_0273` | train | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0035` | train | DAMIMAS | 0.50 | 0 | 2 | 0 | 0 |
| `DAMIMAS_A21B_0043` | train | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0045` | train | DAMIMAS | 0.75 | 0 | 0 | 2 | 1 |
| `DAMIMAS_A21B_0546` | test | DAMIMAS | 0.50 | 0 | 2 | 0 | 0 |
| `DAMIMAS_A21B_0554` | test | DAMIMAS | 1.25 | 0 | 1 | 2 | 2 |
| `DAMIMAS_A21B_0557` | test | DAMIMAS | 1.00 | 1 | 0 | 2 | 1 |
| `DAMIMAS_A21B_0558` | test | DAMIMAS | 1.00 | 0 | 2 | 1 | 1 |
| `DAMIMAS_A21B_0569` | test | DAMIMAS | 1.00 | 1 | 0 | 1 | 2 |
| `DAMIMAS_A21B_0574` | test | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |

## Sample 10 Baris Per-Tree

Kolom penuh tersedia di per-method CSV di atas. Preview:

| tree_id | split | ok | gt_B1 | gt_B2 | gt_B3 | gt_B4 | pred_B1 | pred_B2 | pred_B3 | pred_B4 |
|---|---|---|---|---|---|---|---|---|---|---|
| DAMIMAS_A21B_0001 | train | True | 1 | 2 | 5 | 0 | 1 | 3 | 6 | 0 |
| DAMIMAS_A21B_0244 | train | False | 0 | 0 | 0 | 7 | 0 | 0 | 0 | 9 |
| DAMIMAS_A21B_0577 | train | True | 3 | 0 | 4 | 2 | 3 | 0 | 3 | 2 |
| DAMIMAS_A21B_0002 | train | False | 1 | 0 | 7 | 4 | 2 | 0 | 6 | 6 |
| DAMIMAS_A21B_0245 | train | True | 0 | 0 | 3 | 2 | 0 | 0 | 2 | 3 |
| DAMIMAS_A21B_0578 | train | True | 1 | 5 | 1 | 0 | 1 | 4 | 1 | 0 |
| DAMIMAS_A21B_0003 | train | True | 1 | 2 | 5 | 1 | 1 | 3 | 6 | 1 |
| DAMIMAS_A21B_0246 | train | True | 0 | 6 | 6 | 4 | 0 | 5 | 5 | 4 |
| DAMIMAS_A21B_0579 | train | True | 5 | 3 | 6 | 1 | 5 | 4 | 6 | 1 |
| DAMIMAS_A21B_0004 | train | True | 0 | 0 | 8 | 0 | 0 | 0 | 8 | 0 |
