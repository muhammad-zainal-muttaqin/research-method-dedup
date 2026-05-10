# `v8_entropy_modulated` — Primary Metrics Breakdown

**Implementasi:** [`algorithms/entropy_modulated.py`](../../algorithms/entropy_modulated.py)  
**Dataset:** 228 pohon JSON (228 baris cocok dengan `method=v8_entropy_modulated`)  
**Raw data lengkap:** [`../benchmark_multidim/accuracy_per_tree.csv`](../benchmark_multidim/accuracy_per_tree.csv)  
**Per-method slice (filter sudah diterapkan):** [`v8_entropy_modulated_per_tree.csv`](v8_entropy_modulated_per_tree.csv)  
**Summary CSV:** [`../benchmark_multidim/accuracy_summary.csv`](../benchmark_multidim/accuracy_summary.csv)

Seluruh angka di bawah dihitung ulang dari `accuracy_per_tree.csv` oleh `scripts/generate_method_reports.py`.

## Primary Metrics

| Metric | Value | Derivation |
|---|---:|---|
| Macro class-MAE | **0.2763** | mean(per-class MAE) |
| Exact accuracy | **30.70%** | 70/228 pohon dengan err_B* = 0 di semua kelas |
| Total count MAE | **0.8772** | mean \|Σpred − Σgt\| per pohon |
| Total ±1 accuracy | **78.95%** | 180/228 pohon dengan \|Σpred − Σgt\| ≤ 1 |
| Acc ±1 per kelas per pohon (pelengkap) | 94.30% | 215/228 pohon dengan semua err_B* dalam ±1 |

## Per-Class MAE

Sumber: kolom `err_B*` di `accuracy_per_tree.csv` (sudah absolute).

| Class | MAE | Derivation |
|---|---:|---|
| B1 | **0.0965** | mean(err_B1) across 228 pohon |
| B2 | **0.2544** | mean(err_B2) across 228 pohon |
| B3 | **0.4342** | mean(err_B3) across 228 pohon |
| B4 | **0.3202** | mean(err_B4) across 228 pohon |

Cross-check versus [`accuracy_per_class.csv`](../benchmark_multidim/accuracy_per_class.csv):

| Class | MAE (csv) | over_count | under_count | exact | within1 | pct_within1 |
|---|---:|---:|---:|---:|---:|---:|
| B1 | 0.0965 | 0 | 0 | 206 | 228 | 100.00% |
| B2 | 0.2544 | 1 | 2 | 173 | 225 | 98.68% |
| B3 | 0.4342 | 6 | 2 | 137 | 220 | 96.49% |
| B4 | 0.3202 | 2 | 1 | 158 | 225 | 98.68% |

## Per-Class Mean Error (Bias)

Sumber: `pred_B* − gt_B*` di `accuracy_per_tree.csv`. Nilai `+` = overcount, `−` = undercount, `0` = tidak bias.

| Class | Mean Error | Derivation |
|---|---:|---|
| B1 | **+0.070** | mean(pred_B1 − gt_B1) across 228 pohon |
| B2 | **+0.070** | mean(pred_B2 − gt_B2) across 228 pohon |
| B3 | **+0.048** | mean(pred_B3 − gt_B3) across 228 pohon |
| B4 | **+0.110** | mean(pred_B4 − gt_B4) across 228 pohon |

## Kecepatan (pelengkap)

Sumber: [`speed_summary.csv`](../benchmark_multidim/speed_summary.csv) (30 repetisi × 228 pohon)

- Mean: **0.1046 ms/pohon** (9,558 pohon/detik)
- Median: 0.1044 ms
- Std: 0.0023 ms

## Robustness terhadap Noise Koordinat (pelengkap)

Sumber: [`robustness_summary.csv`](../benchmark_multidim/robustness_summary.csv)

| σ (noise_pct) | Acc ±1 | MAE | n_fail | Acc drop vs σ=0 |
|---:|---:|---:|---:|---:|
| 0% | 94.30% | 0.2763 | 13 | +0.00% |
| 5% | 93.86% | 0.2818 | 14 | +0.44% |
| 10% | 92.98% | 0.2873 | 16 | +1.32% |
| 20% | 92.98% | 0.2873 | 16 | +1.32% |

## Pohon yang Gagal (Acc±1 fail = 13)

| tree_id | split | domain | MAE | err_B1 | err_B2 | err_B3 | err_B4 |
|---|---|---|---:|---:|---:|---:|---:|
| `DAMIMAS_A21B_0002` | train | DAMIMAS | 1.00 | 1 | 0 | 1 | 2 |
| `DAMIMAS_A21B_0257` | train | DAMIMAS | 0.75 | 0 | 1 | 2 | 0 |
| `DAMIMAS_A21B_0259` | train | DAMIMAS | 0.50 | 0 | 2 | 0 | 0 |
| `DAMIMAS_A21B_0273` | train | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0035` | train | DAMIMAS | 0.50 | 0 | 2 | 0 | 0 |
| `DAMIMAS_A21B_0281` | train | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0043` | train | DAMIMAS | 0.75 | 0 | 1 | 2 | 0 |
| `DAMIMAS_A21B_0045` | train | DAMIMAS | 0.75 | 0 | 0 | 2 | 1 |
| `DAMIMAS_A21B_0546` | test | DAMIMAS | 0.50 | 0 | 2 | 0 | 0 |
| `DAMIMAS_A21B_0554` | test | DAMIMAS | 1.25 | 0 | 1 | 2 | 2 |
| `DAMIMAS_A21B_0557` | test | DAMIMAS | 1.00 | 1 | 0 | 2 | 1 |
| `DAMIMAS_A21B_0569` | test | DAMIMAS | 1.00 | 1 | 0 | 1 | 2 |
| `DAMIMAS_A21B_0574` | test | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |

## Sample 10 Baris Per-Tree

Kolom penuh tersedia di per-method CSV di atas. Preview:

| tree_id | split | ok | gt_B1 | gt_B2 | gt_B3 | gt_B4 | pred_B1 | pred_B2 | pred_B3 | pred_B4 |
|---|---|---|---|---|---|---|---|---|---|---|
| DAMIMAS_A21B_0001 | train | True | 1 | 2 | 5 | 0 | 1 | 3 | 6 | 0 |
| DAMIMAS_A21B_0244 | train | True | 0 | 0 | 0 | 7 | 0 | 0 | 0 | 8 |
| DAMIMAS_A21B_0577 | train | True | 3 | 0 | 4 | 2 | 3 | 0 | 3 | 2 |
| DAMIMAS_A21B_0002 | train | False | 1 | 0 | 7 | 4 | 2 | 0 | 6 | 6 |
| DAMIMAS_A21B_0245 | train | True | 0 | 0 | 3 | 2 | 0 | 0 | 2 | 3 |
| DAMIMAS_A21B_0578 | train | True | 1 | 5 | 1 | 0 | 1 | 4 | 1 | 0 |
| DAMIMAS_A21B_0003 | train | True | 1 | 2 | 5 | 1 | 1 | 3 | 6 | 1 |
| DAMIMAS_A21B_0246 | train | True | 0 | 6 | 6 | 4 | 0 | 5 | 5 | 4 |
| DAMIMAS_A21B_0579 | train | True | 5 | 3 | 6 | 1 | 5 | 4 | 6 | 1 |
| DAMIMAS_A21B_0004 | train | True | 0 | 0 | 8 | 0 | 0 | 0 | 8 | 0 |
