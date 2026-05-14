# `M19_divide_adaptive` — Primary Metrics Breakdown

**Implementasi:** [`algorithms/M19_divide_adaptive.py`](../../algorithms/M19_divide_adaptive.py)  
**Dataset:** 228 pohon JSON (953 baris cocok dengan `method=M19_divide_adaptive`)  
**Raw data lengkap:** [`../benchmark_multidim/accuracy_per_tree.csv`](../benchmark_multidim/accuracy_per_tree.csv)  
**Per-method slice (filter sudah diterapkan):** [`M19_divide_adaptive_per_tree.csv`](M19_divide_adaptive_per_tree.csv)  
**Summary CSV:** [`../benchmark_multidim/accuracy_summary.csv`](../benchmark_multidim/accuracy_summary.csv)

Seluruh angka di bawah dihitung ulang dari `accuracy_per_tree.csv` oleh `scripts/generate_method_reports.py`.

## Primary Metrics

| Metric | Value | Derivation |
|---|---:|---|
| Macro class-MAE | **0.4583** | mean(per-class MAE) |
| Exact accuracy | **21.51%** | 205/953 pohon dengan err_B* = 0 di semua kelas |
| Total count MAE | **1.6842** | mean \|Σpred − Σgt\| per pohon |
| Total ±1 accuracy | **65.69%** | 626/953 pohon dengan \|Σpred − Σgt\| ≤ 1 |
| Acc ±1 per kelas per pohon (pelengkap) | 82.69% | 788/953 pohon dengan semua err_B* dalam ±1 |

## Per-Class MAE

Sumber: kolom `err_B*` di `accuracy_per_tree.csv` (sudah absolute).

| Class | MAE | Derivation |
|---|---:|---|
| B1 | **0.1973** | mean(err_B1) across 953 pohon |
| B2 | **0.3736** | mean(err_B2) across 953 pohon |
| B3 | **0.9286** | mean(err_B3) across 953 pohon |
| B4 | **0.3337** | mean(err_B4) across 953 pohon |

Cross-check versus [`accuracy_per_class.csv`](../benchmark_multidim/accuracy_per_class.csv):

| Class | MAE (csv) | over_count | under_count | exact | within1 | pct_within1 |
|---|---:|---:|---:|---:|---:|---:|
| B1 | 0.1973 | 25 | 0 | 810 | 928 | 97.38% |
| B2 | 0.3736 | 43 | 6 | 721 | 904 | 94.86% |
| B3 | 0.9286 | 114 | 14 | 437 | 825 | 86.57% |
| B4 | 0.3337 | 19 | 7 | 677 | 927 | 97.27% |

## Per-Class Mean Error (Bias)

Sumber: `pred_B* − gt_B*` di `accuracy_per_tree.csv`. Nilai `+` = overcount, `−` = undercount, `0` = tidak bias.

| Class | Mean Error | Derivation |
|---|---:|---|
| B1 | **+0.147** | mean(pred_B1 − gt_B1) across 953 pohon |
| B2 | **+0.256** | mean(pred_B2 − gt_B2) across 953 pohon |
| B3 | **+0.589** | mean(pred_B3 − gt_B3) across 953 pohon |
| B4 | **+0.080** | mean(pred_B4 − gt_B4) across 953 pohon |

## Kecepatan (pelengkap)

Sumber: [`speed_summary.csv`](../benchmark_multidim/speed_summary.csv) (30 repetisi × 953 pohon)

- Mean: **0.0102 ms/pohon** (97,930 pohon/detik)
- Median: 0.0101 ms
- Std: 0.0002 ms

## Robustness terhadap Noise Koordinat (pelengkap)

Sumber: [`robustness_summary.csv`](../benchmark_multidim/robustness_summary.csv)

| σ (noise_pct) | Acc ±1 | MAE | n_fail | Acc drop vs σ=0 |
|---:|---:|---:|---:|---:|
| 0% | 82.69% | 0.4583 | 165 | +0.00% |
| 5% | 82.69% | 0.4583 | 165 | +0.00% |
| 10% | 82.69% | 0.4583 | 165 | +0.00% |
| 20% | 82.69% | 0.4583 | 165 | +0.00% |

## Pohon yang Gagal (Acc±1 fail = 165)

| tree_id | split | domain | MAE | err_B1 | err_B2 | err_B3 | err_B4 |
|---|---|---|---:|---:|---:|---:|---:|
| `DAMIMAS_A21B_0002` | train | DAMIMAS | 1.00 | 1 | 0 | 1 | 2 |
| `DAMIMAS_A21B_0035` | train | DAMIMAS | 0.50 | 0 | 2 | 0 | 0 |
| `DAMIMAS_A21B_0043` | train | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0045` | train | DAMIMAS | 0.75 | 0 | 0 | 2 | 1 |
| `DAMIMAS_A21B_0060` | val | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0061` | train | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0079` | train | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0085` | train | DAMIMAS | 0.75 | 0 | 2 | 1 | 0 |
| `DAMIMAS_A21B_0102` | train | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0116` | val | DAMIMAS | 0.75 | 0 | 0 | 2 | 1 |
| `DAMIMAS_A21B_0121` | train | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0124` | train | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0128` | train | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0133` | train | DAMIMAS | 0.75 | 0 | 0 | 3 | 0 |
| `DAMIMAS_A21B_0135` | train | DAMIMAS | 0.50 | 0 | 2 | 0 | 0 |
| `DAMIMAS_A21B_0136` | train | DAMIMAS | 0.75 | 0 | 0 | 2 | 1 |
| `DAMIMAS_A21B_0137` | train | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0139` | train | DAMIMAS | 0.75 | 0 | 0 | 2 | 1 |
| `DAMIMAS_A21B_0140` | train | DAMIMAS | 1.00 | 0 | 1 | 1 | 2 |
| `DAMIMAS_A21B_0141` | train | DAMIMAS | 1.00 | 0 | 0 | 3 | 1 |
| `DAMIMAS_A21B_0143` | train | DAMIMAS | 0.75 | 0 | 0 | 3 | 0 |
| `DAMIMAS_A21B_0147` | val | DAMIMAS | 1.00 | 0 | 0 | 3 | 1 |
| `DAMIMAS_A21B_0150` | val | DAMIMAS | 0.75 | 0 | 1 | 2 | 0 |
| `DAMIMAS_A21B_0151` | val | DAMIMAS | 1.00 | 0 | 0 | 3 | 1 |
| `DAMIMAS_A21B_0152` | val | DAMIMAS | 1.25 | 0 | 1 | 3 | 1 |
| `DAMIMAS_A21B_0154` | val | DAMIMAS | 0.75 | 0 | 0 | 1 | 2 |
| `DAMIMAS_A21B_0162` | val | DAMIMAS | 0.75 | 0 | 0 | 3 | 0 |
| `DAMIMAS_A21B_0163` | val | DAMIMAS | 1.50 | 1 | 0 | 3 | 2 |
| `DAMIMAS_A21B_0170` | val | DAMIMAS | 0.75 | 0 | 0 | 0 | 3 |
| `DAMIMAS_A21B_0175` | val | DAMIMAS | 0.75 | 0 | 0 | 3 | 0 |
| `DAMIMAS_A21B_0176` | val | DAMIMAS | 0.75 | 1 | 0 | 0 | 2 |
| `DAMIMAS_A21B_0177` | val | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0180` | val | DAMIMAS | 0.75 | 0 | 0 | 1 | 2 |
| `DAMIMAS_A21B_0183` | val | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0193` | val | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0199` | test | DAMIMAS | 0.75 | 1 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0201` | test | DAMIMAS | 0.75 | 1 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0206` | test | DAMIMAS | 0.75 | 0 | 1 | 2 | 0 |
| `DAMIMAS_A21B_0208` | test | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0214` | test | DAMIMAS | 0.50 | 0 | 0 | 0 | 2 |
| `DAMIMAS_A21B_0234` | test | DAMIMAS | 0.75 | 0 | 0 | 1 | 2 |
| `DAMIMAS_A21B_0237` | test | DAMIMAS | 1.00 | 1 | 1 | 2 | 0 |
| `DAMIMAS_A21B_0243` | val | DAMIMAS | 1.00 | 1 | 3 | 0 | 0 |
| `DAMIMAS_A21B_0244` | train | DAMIMAS | 0.50 | 0 | 0 | 0 | 2 |
| `DAMIMAS_A21B_0257` | train | DAMIMAS | 0.75 | 0 | 1 | 2 | 0 |
| `DAMIMAS_A21B_0259` | train | DAMIMAS | 0.50 | 0 | 2 | 0 | 0 |
| `DAMIMAS_A21B_0273` | train | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0289` | val | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0291` | train | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0303` | train | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0309` | train | DAMIMAS | 0.75 | 2 | 1 | 0 | 0 |
| `DAMIMAS_A21B_0311` | val | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0315` | train | DAMIMAS | 0.50 | 0 | 2 | 0 | 0 |
| `DAMIMAS_A21B_0318` | train | DAMIMAS | 0.50 | 0 | 0 | 0 | 2 |
| `DAMIMAS_A21B_0321` | test | DAMIMAS | 0.75 | 1 | 2 | 0 | 0 |
| `DAMIMAS_A21B_0323` | test | DAMIMAS | 1.00 | 0 | 0 | 4 | 0 |
| `DAMIMAS_A21B_0324` | train | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0325` | test | DAMIMAS | 0.75 | 0 | 0 | 2 | 1 |
| `DAMIMAS_A21B_0331` | test | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0334` | train | DAMIMAS | 0.75 | 0 | 0 | 1 | 2 |
| `DAMIMAS_A21B_0336` | train | DAMIMAS | 1.00 | 2 | 0 | 1 | 1 |
| `DAMIMAS_A21B_0341` | train | DAMIMAS | 1.00 | 0 | 0 | 4 | 0 |
| `DAMIMAS_A21B_0342` | val | DAMIMAS | 1.25 | 0 | 3 | 2 | 0 |
| `DAMIMAS_A21B_0343` | train | DAMIMAS | 0.75 | 0 | 2 | 1 | 0 |
| `DAMIMAS_A21B_0346` | test | DAMIMAS | 1.00 | 1 | 0 | 2 | 1 |
| `DAMIMAS_A21B_0352` | train | DAMIMAS | 0.75 | 1 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0353` | train | DAMIMAS | 0.75 | 1 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0358` | train | DAMIMAS | 0.75 | 0 | 0 | 3 | 0 |
| `DAMIMAS_A21B_0360` | train | DAMIMAS | 1.00 | 0 | 2 | 1 | 1 |
| `DAMIMAS_A21B_0361` | train | DAMIMAS | 1.00 | 1 | 0 | 2 | 1 |
| `DAMIMAS_A21B_0362` | train | DAMIMAS | 2.25 | 0 | 0 | 9 | 0 |
| `DAMIMAS_A21B_0364` | train | DAMIMAS | 1.00 | 1 | 1 | 2 | 0 |
| `DAMIMAS_A21B_0366` | train | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0374` | train | DAMIMAS | 0.75 | 0 | 0 | 2 | 1 |
| `DAMIMAS_A21B_0388` | val | DAMIMAS | 1.00 | 1 | 0 | 3 | 0 |
| `DAMIMAS_A21B_0389` | train | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0398` | val | DAMIMAS | 0.75 | 1 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0414` | train | DAMIMAS | 0.75 | 0 | 0 | 3 | 0 |
| `DAMIMAS_A21B_0429` | train | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0430` | train | DAMIMAS | 0.75 | 0 | 0 | 2 | 1 |
| `DAMIMAS_A21B_0440` | train | DAMIMAS | 0.75 | 0 | 1 | 2 | 0 |
| `DAMIMAS_A21B_0447` | train | DAMIMAS | 0.75 | 0 | 1 | 2 | 0 |
| `DAMIMAS_A21B_0486` | train | DAMIMAS | 1.25 | 1 | 1 | 1 | 2 |
| `DAMIMAS_A21B_0489` | val | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0503` | train | DAMIMAS | 0.50 | 0 | 2 | 0 | 0 |
| `DAMIMAS_A21B_0506` | train | DAMIMAS | 1.00 | 1 | 1 | 0 | 2 |
| `DAMIMAS_A21B_0522` | train | DAMIMAS | 0.50 | 0 | 2 | 0 | 0 |
| `DAMIMAS_A21B_0531` | train | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0539` | train | DAMIMAS | 0.75 | 0 | 1 | 2 | 0 |
| `DAMIMAS_A21B_0545` | train | DAMIMAS | 0.75 | 0 | 0 | 3 | 0 |
| `DAMIMAS_A21B_0550` | val | DAMIMAS | 0.50 | 0 | 2 | 0 | 0 |
| `DAMIMAS_A21B_0558` | train | DAMIMAS | 0.50 | 0 | 2 | 0 | 0 |
| `DAMIMAS_A21B_0559` | train | DAMIMAS | 0.75 | 0 | 0 | 0 | 3 |
| `DAMIMAS_A21B_0566` | train | DAMIMAS | 1.00 | 0 | 2 | 1 | 1 |
| `DAMIMAS_A21B_0571` | train | DAMIMAS | 0.50 | 0 | 2 | 0 | 0 |
| `DAMIMAS_A21B_0669` | train | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0702` | train | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0704` | train | DAMIMAS | 0.75 | 0 | 1 | 2 | 0 |
| `DAMIMAS_A21B_0708` | val | DAMIMAS | 1.00 | 1 | 0 | 2 | 1 |
| `DAMIMAS_A21B_0712` | test | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0716` | test | DAMIMAS | 0.75 | 0 | 0 | 2 | 1 |
| `DAMIMAS_A21B_0721` | train | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0723` | train | DAMIMAS | 1.50 | 0 | 2 | 2 | 2 |
| `DAMIMAS_A21B_0726` | test | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0727` | test | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0729` | train | DAMIMAS | 0.75 | 1 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0732` | train | DAMIMAS | 0.75 | 0 | 0 | 1 | 2 |
| `DAMIMAS_A21B_0737` | test | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0741` | train | DAMIMAS | 1.00 | 0 | 1 | 2 | 1 |
| `DAMIMAS_A21B_0747` | train | DAMIMAS | 1.25 | 0 | 1 | 2 | 2 |
| `DAMIMAS_A21B_0751` | train | DAMIMAS | 0.75 | 0 | 1 | 2 | 0 |
| `DAMIMAS_A21B_0756` | train | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0759` | val | DAMIMAS | 0.75 | 0 | 0 | 2 | 1 |
| `DAMIMAS_A21B_0774` | train | DAMIMAS | 1.00 | 1 | 0 | 3 | 0 |
| `DAMIMAS_A21B_0778` | test | DAMIMAS | 0.75 | 0 | 0 | 2 | 1 |
| `DAMIMAS_A21B_0810` | val | DAMIMAS | 4.50 | 1 | 2 | 12 | 3 |
| `DAMIMAS_A21B_0811` | train | DAMIMAS | 4.00 | 4 | 1 | 11 | 0 |
| `DAMIMAS_A21B_0812` | val | DAMIMAS | 3.75 | 5 | 4 | 6 | 0 |
| `DAMIMAS_A21B_0813` | test | DAMIMAS | 1.50 | 1 | 0 | 4 | 1 |
| `DAMIMAS_A21B_0814` | test | DAMIMAS | 1.75 | 0 | 0 | 6 | 1 |
| `DAMIMAS_A21B_0815` | train | DAMIMAS | 5.00 | 2 | 4 | 14 | 0 |
| `DAMIMAS_A21B_0816` | train | DAMIMAS | 3.00 | 2 | 5 | 5 | 0 |
| `DAMIMAS_A21B_0817` | val | DAMIMAS | 2.00 | 5 | 2 | 1 | 0 |
| `DAMIMAS_A21B_0818` | train | DAMIMAS | 4.25 | 0 | 8 | 9 | 0 |
| `DAMIMAS_A21B_0819` | train | DAMIMAS | 4.00 | 0 | 4 | 11 | 1 |
| `DAMIMAS_A21B_0820` | train | DAMIMAS | 5.50 | 0 | 6 | 16 | 0 |
| `DAMIMAS_A21B_0821` | val | DAMIMAS | 1.25 | 3 | 1 | 1 | 0 |
| `DAMIMAS_A21B_0822` | test | DAMIMAS | 3.00 | 1 | 1 | 10 | 0 |
| `DAMIMAS_A21B_0823` | train | DAMIMAS | 6.00 | 3 | 2 | 16 | 3 |
| `DAMIMAS_A21B_0824` | val | DAMIMAS | 4.25 | 3 | 10 | 4 | 0 |
| `DAMIMAS_A21B_0825` | val | DAMIMAS | 1.50 | 2 | 0 | 4 | 0 |
| `DAMIMAS_A21B_0826` | val | DAMIMAS | 1.50 | 0 | 3 | 3 | 0 |
| `DAMIMAS_A21B_0827` | train | DAMIMAS | 4.50 | 2 | 2 | 13 | 1 |
| `DAMIMAS_A21B_0828` | train | DAMIMAS | 4.25 | 1 | 9 | 7 | 0 |
| `DAMIMAS_A21B_0829` | val | DAMIMAS | 2.50 | 2 | 2 | 5 | 1 |
| `DAMIMAS_A21B_0830` | train | DAMIMAS | 2.25 | 4 | 2 | 3 | 0 |
| `DAMIMAS_A21B_0831` | train | DAMIMAS | 5.75 | 1 | 4 | 14 | 4 |
| `DAMIMAS_A21B_0832` | train | DAMIMAS | 3.50 | 0 | 6 | 7 | 1 |
| `DAMIMAS_A21B_0833` | train | DAMIMAS | 1.25 | 3 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0834` | train | DAMIMAS | 4.25 | 1 | 7 | 5 | 4 |
| `DAMIMAS_A21B_0835` | train | DAMIMAS | 2.50 | 0 | 0 | 10 | 0 |
| `DAMIMAS_A21B_0836` | train | DAMIMAS | 3.50 | 3 | 4 | 7 | 0 |
| `DAMIMAS_A21B_0837` | train | DAMIMAS | 2.25 | 3 | 0 | 6 | 0 |
| `DAMIMAS_A21B_0838` | train | DAMIMAS | 3.50 | 0 | 1 | 5 | 8 |
| `DAMIMAS_A21B_0839` | val | DAMIMAS | 3.50 | 2 | 3 | 9 | 0 |
| `DAMIMAS_A21B_0840` | train | DAMIMAS | 3.75 | 3 | 8 | 4 | 0 |
| `DAMIMAS_A21B_0841` | train | DAMIMAS | 0.75 | 1 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0842` | train | DAMIMAS | 4.50 | 3 | 3 | 8 | 4 |
| `DAMIMAS_A21B_0843` | test | DAMIMAS | 2.00 | 0 | 2 | 6 | 0 |
| `DAMIMAS_A21B_0844` | train | DAMIMAS | 3.75 | 1 | 6 | 7 | 1 |
| `DAMIMAS_A21B_0846` | train | DAMIMAS | 2.25 | 0 | 5 | 4 | 0 |
| `DAMIMAS_A21B_0847` | test | DAMIMAS | 3.50 | 2 | 2 | 10 | 0 |
| `DAMIMAS_A21B_0848` | val | DAMIMAS | 2.50 | 0 | 3 | 7 | 0 |
| `DAMIMAS_A21B_0849` | train | DAMIMAS | 1.00 | 2 | 1 | 1 | 0 |
| `DAMIMAS_A21B_0850` | train | DAMIMAS | 4.00 | 2 | 4 | 9 | 1 |
| `DAMIMAS_A21B_0851` | train | DAMIMAS | 4.00 | 0 | 7 | 8 | 1 |
| `DAMIMAS_A21B_0852` | train | DAMIMAS | 0.50 | 0 | 2 | 0 | 0 |
| `DAMIMAS_A21B_0853` | train | DAMIMAS | 1.50 | 2 | 2 | 2 | 0 |
| `DAMIMAS_A21B_0854` | train | DAMIMAS | 3.75 | 4 | 4 | 5 | 2 |
| `LONSUM_A21A_0027` | train | LONSUM | 0.75 | 0 | 0 | 1 | 2 |
| `LONSUM_A21A_0035` | train | LONSUM | 0.50 | 0 | 0 | 2 | 0 |
| `LONSUM_A21A_0051` | test | LONSUM | 0.50 | 0 | 0 | 2 | 0 |
| `LONSUM_A21A_0091` | val | LONSUM | 0.50 | 0 | 0 | 2 | 0 |
| `LONSUM_A21A_0092` | train | LONSUM | 0.50 | 0 | 0 | 2 | 0 |
| `LONSUM_A21A_0096` | train | LONSUM | 0.50 | 0 | 0 | 2 | 0 |

## Sample 10 Baris Per-Tree

Kolom penuh tersedia di per-method CSV di atas. Preview:

| tree_id | split | ok | gt_B1 | gt_B2 | gt_B3 | gt_B4 | pred_B1 | pred_B2 | pred_B3 | pred_B4 |
|---|---|---|---|---|---|---|---|---|---|---|
| DAMIMAS_A21B_0001 | train | True | 1 | 2 | 5 | 0 | 1 | 3 | 6 | 0 |
| DAMIMAS_A21B_0002 | train | False | 1 | 0 | 7 | 4 | 2 | 0 | 6 | 6 |
| DAMIMAS_A21B_0003 | train | True | 1 | 2 | 5 | 1 | 1 | 3 | 6 | 1 |
| DAMIMAS_A21B_0004 | train | True | 0 | 0 | 8 | 0 | 0 | 0 | 8 | 0 |
| DAMIMAS_A21B_0005 | test | True | 1 | 3 | 3 | 2 | 1 | 3 | 3 | 1 |
| DAMIMAS_A21B_0006 | train | True | 1 | 0 | 5 | 3 | 1 | 0 | 4 | 4 |
| DAMIMAS_A21B_0007 | train | True | 3 | 3 | 8 | 3 | 3 | 2 | 8 | 3 |
| DAMIMAS_A21B_0008 | train | True | 2 | 3 | 4 | 0 | 2 | 3 | 4 | 0 |
| DAMIMAS_A21B_0009 | train | True | 0 | 0 | 3 | 3 | 0 | 0 | 3 | 2 |
| DAMIMAS_A21B_0010 | train | True | 0 | 2 | 6 | 3 | 0 | 3 | 6 | 4 |
