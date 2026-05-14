# `M16_boost_b2b4` — Primary Metrics Breakdown

**Implementasi:** [`algorithms/M16_boost_b2b4.py`](../../algorithms/M16_boost_b2b4.py)  
**Dataset:** 228 pohon JSON (953 baris cocok dengan `method=M16_boost_b2b4`)  
**Raw data lengkap:** [`../benchmark_multidim/accuracy_per_tree.csv`](../benchmark_multidim/accuracy_per_tree.csv)  
**Per-method slice (filter sudah diterapkan):** [`M16_boost_b2b4_per_tree.csv`](M16_boost_b2b4_per_tree.csv)  
**Summary CSV:** [`../benchmark_multidim/accuracy_summary.csv`](../benchmark_multidim/accuracy_summary.csv)

Seluruh angka di bawah dihitung ulang dari `accuracy_per_tree.csv` oleh `scripts/generate_method_reports.py`.

## Primary Metrics

| Metric | Value | Derivation |
|---|---:|---|
| Macro class-MAE | **0.4048** | mean(per-class MAE) |
| Exact accuracy | **26.86%** | 256/953 pohon dengan err_B* = 0 di semua kelas |
| Total count MAE | **1.4659** | mean \|Σpred − Σgt\| per pohon |
| Total ±1 accuracy | **72.19%** | 688/953 pohon dengan \|Σpred − Σgt\| ≤ 1 |
| Acc ±1 per kelas per pohon (pelengkap) | 84.58% | 806/953 pohon dengan semua err_B* dalam ±1 |

## Per-Class MAE

Sumber: kolom `err_B*` di `accuracy_per_tree.csv` (sudah absolute).

| Class | MAE | Derivation |
|---|---:|---|
| B1 | **0.1647** | mean(err_B1) across 953 pohon |
| B2 | **0.3190** | mean(err_B2) across 953 pohon |
| B3 | **0.8447** | mean(err_B3) across 953 pohon |
| B4 | **0.2907** | mean(err_B4) across 953 pohon |

Cross-check versus [`accuracy_per_class.csv`](../benchmark_multidim/accuracy_per_class.csv):

| Class | MAE (csv) | over_count | under_count | exact | within1 | pct_within1 |
|---|---:|---:|---:|---:|---:|---:|
| B1 | 0.1647 | 24 | 0 | 837 | 929 | 97.48% |
| B2 | 0.3190 | 33 | 14 | 749 | 906 | 95.07% |
| B3 | 0.8447 | 93 | 17 | 461 | 843 | 88.46% |
| B4 | 0.2907 | 11 | 18 | 715 | 924 | 96.96% |

## Per-Class Mean Error (Bias)

Sumber: `pred_B* − gt_B*` di `accuracy_per_tree.csv`. Nilai `+` = overcount, `−` = undercount, `0` = tidak bias.

| Class | Mean Error | Derivation |
|---|---:|---|
| B1 | **+0.135** | mean(pred_B1 − gt_B1) across 953 pohon |
| B2 | **+0.073** | mean(pred_B2 − gt_B2) across 953 pohon |
| B3 | **+0.477** | mean(pred_B3 − gt_B3) across 953 pohon |
| B4 | **-0.110** | mean(pred_B4 − gt_B4) across 953 pohon |

## Kecepatan (pelengkap)

Sumber: [`speed_summary.csv`](../benchmark_multidim/speed_summary.csv) (30 repetisi × 953 pohon)

- Mean: **0.0486 ms/pohon** (20,557 pohon/detik)
- Median: 0.0485 ms
- Std: 0.0006 ms

## Robustness terhadap Noise Koordinat (pelengkap)

Sumber: [`robustness_summary.csv`](../benchmark_multidim/robustness_summary.csv)

| σ (noise_pct) | Acc ±1 | MAE | n_fail | Acc drop vs σ=0 |
|---:|---:|---:|---:|---:|
| 0% | 84.58% | 0.4048 | 147 | +0.00% |
| 5% | 83.00% | 0.4229 | 162 | +1.58% |
| 10% | 82.48% | 0.4326 | 167 | +2.10% |
| 20% | 82.48% | 0.4328 | 167 | +2.10% |

## Pohon yang Gagal (Acc±1 fail = 147)

| tree_id | split | domain | MAE | err_B1 | err_B2 | err_B3 | err_B4 |
|---|---|---|---:|---:|---:|---:|---:|
| `DAMIMAS_A21B_0002` | train | DAMIMAS | 1.00 | 1 | 0 | 1 | 2 |
| `DAMIMAS_A21B_0034` | train | DAMIMAS | 0.50 | 0 | 0 | 0 | 2 |
| `DAMIMAS_A21B_0035` | train | DAMIMAS | 0.75 | 0 | 3 | 0 | 0 |
| `DAMIMAS_A21B_0043` | train | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0045` | train | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0073` | test | DAMIMAS | 0.50 | 0 | 0 | 0 | 2 |
| `DAMIMAS_A21B_0079` | train | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0102` | train | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0116` | val | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0121` | train | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0124` | train | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0128` | train | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0133` | train | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0139` | train | DAMIMAS | 0.75 | 0 | 0 | 2 | 1 |
| `DAMIMAS_A21B_0141` | train | DAMIMAS | 1.00 | 0 | 0 | 3 | 1 |
| `DAMIMAS_A21B_0143` | train | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0147` | val | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0151` | val | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0152` | val | DAMIMAS | 1.00 | 0 | 1 | 2 | 1 |
| `DAMIMAS_A21B_0162` | val | DAMIMAS | 0.75 | 0 | 0 | 2 | 1 |
| `DAMIMAS_A21B_0163` | val | DAMIMAS | 1.00 | 1 | 0 | 2 | 1 |
| `DAMIMAS_A21B_0170` | val | DAMIMAS | 0.50 | 0 | 0 | 0 | 2 |
| `DAMIMAS_A21B_0175` | val | DAMIMAS | 0.75 | 0 | 0 | 3 | 0 |
| `DAMIMAS_A21B_0176` | val | DAMIMAS | 0.50 | 0 | 0 | 0 | 2 |
| `DAMIMAS_A21B_0177` | val | DAMIMAS | 0.75 | 0 | 1 | 2 | 0 |
| `DAMIMAS_A21B_0193` | val | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0201` | test | DAMIMAS | 0.75 | 1 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0208` | test | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0214` | test | DAMIMAS | 0.50 | 0 | 0 | 0 | 2 |
| `DAMIMAS_A21B_0234` | test | DAMIMAS | 0.75 | 0 | 0 | 1 | 2 |
| `DAMIMAS_A21B_0237` | test | DAMIMAS | 1.00 | 1 | 1 | 2 | 0 |
| `DAMIMAS_A21B_0243` | val | DAMIMAS | 0.75 | 1 | 2 | 0 | 0 |
| `DAMIMAS_A21B_0246` | test | DAMIMAS | 1.00 | 0 | 2 | 1 | 1 |
| `DAMIMAS_A21B_0257` | train | DAMIMAS | 0.75 | 0 | 1 | 2 | 0 |
| `DAMIMAS_A21B_0259` | train | DAMIMAS | 0.75 | 0 | 3 | 0 | 0 |
| `DAMIMAS_A21B_0268` | train | DAMIMAS | 1.00 | 0 | 2 | 1 | 1 |
| `DAMIMAS_A21B_0273` | train | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0281` | train | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0289` | val | DAMIMAS | 0.75 | 0 | 1 | 2 | 0 |
| `DAMIMAS_A21B_0291` | train | DAMIMAS | 1.00 | 0 | 1 | 2 | 1 |
| `DAMIMAS_A21B_0311` | val | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0315` | train | DAMIMAS | 0.50 | 0 | 2 | 0 | 0 |
| `DAMIMAS_A21B_0318` | train | DAMIMAS | 0.50 | 0 | 0 | 0 | 2 |
| `DAMIMAS_A21B_0320` | train | DAMIMAS | 0.75 | 0 | 1 | 0 | 2 |
| `DAMIMAS_A21B_0321` | test | DAMIMAS | 0.75 | 1 | 2 | 0 | 0 |
| `DAMIMAS_A21B_0323` | test | DAMIMAS | 1.00 | 0 | 0 | 4 | 0 |
| `DAMIMAS_A21B_0325` | test | DAMIMAS | 0.75 | 0 | 0 | 2 | 1 |
| `DAMIMAS_A21B_0334` | train | DAMIMAS | 0.75 | 0 | 0 | 1 | 2 |
| `DAMIMAS_A21B_0336` | train | DAMIMAS | 1.00 | 2 | 0 | 1 | 1 |
| `DAMIMAS_A21B_0341` | train | DAMIMAS | 0.75 | 0 | 0 | 3 | 0 |
| `DAMIMAS_A21B_0342` | val | DAMIMAS | 1.00 | 0 | 2 | 2 | 0 |
| `DAMIMAS_A21B_0348` | test | DAMIMAS | 0.75 | 1 | 0 | 0 | 2 |
| `DAMIMAS_A21B_0352` | train | DAMIMAS | 1.00 | 1 | 0 | 2 | 1 |
| `DAMIMAS_A21B_0358` | train | DAMIMAS | 0.75 | 0 | 0 | 2 | 1 |
| `DAMIMAS_A21B_0361` | train | DAMIMAS | 1.00 | 1 | 0 | 2 | 1 |
| `DAMIMAS_A21B_0362` | train | DAMIMAS | 2.00 | 0 | 0 | 8 | 0 |
| `DAMIMAS_A21B_0364` | train | DAMIMAS | 0.75 | 1 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0374` | train | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0388` | val | DAMIMAS | 1.00 | 1 | 0 | 3 | 0 |
| `DAMIMAS_A21B_0389` | train | DAMIMAS | 0.75 | 0 | 0 | 2 | 1 |
| `DAMIMAS_A21B_0391` | val | DAMIMAS | 0.50 | 0 | 2 | 0 | 0 |
| `DAMIMAS_A21B_0395` | train | DAMIMAS | 0.75 | 0 | 2 | 1 | 0 |
| `DAMIMAS_A21B_0396` | train | DAMIMAS | 0.50 | 0 | 0 | 0 | 2 |
| `DAMIMAS_A21B_0398` | val | DAMIMAS | 1.00 | 1 | 0 | 2 | 1 |
| `DAMIMAS_A21B_0414` | train | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0429` | train | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0430` | train | DAMIMAS | 1.00 | 0 | 0 | 2 | 2 |
| `DAMIMAS_A21B_0440` | train | DAMIMAS | 1.00 | 0 | 1 | 2 | 1 |
| `DAMIMAS_A21B_0447` | train | DAMIMAS | 1.25 | 0 | 2 | 2 | 1 |
| `DAMIMAS_A21B_0450` | train | DAMIMAS | 1.00 | 1 | 0 | 2 | 1 |
| `DAMIMAS_A21B_0471` | val | DAMIMAS | 0.50 | 0 | 2 | 0 | 0 |
| `DAMIMAS_A21B_0475` | test | DAMIMAS | 0.75 | 0 | 2 | 1 | 0 |
| `DAMIMAS_A21B_0494` | train | DAMIMAS | 0.75 | 0 | 0 | 1 | 2 |
| `DAMIMAS_A21B_0497` | test | DAMIMAS | 1.00 | 0 | 1 | 2 | 1 |
| `DAMIMAS_A21B_0522` | train | DAMIMAS | 0.50 | 0 | 2 | 0 | 0 |
| `DAMIMAS_A21B_0523` | train | DAMIMAS | 0.50 | 0 | 2 | 0 | 0 |
| `DAMIMAS_A21B_0524` | train | DAMIMAS | 0.50 | 0 | 0 | 0 | 2 |
| `DAMIMAS_A21B_0531` | train | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0545` | train | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0559` | train | DAMIMAS | 0.75 | 0 | 0 | 0 | 3 |
| `DAMIMAS_A21B_0571` | train | DAMIMAS | 0.50 | 0 | 2 | 0 | 0 |
| `DAMIMAS_A21B_0656` | train | DAMIMAS | 0.50 | 0 | 0 | 0 | 2 |
| `DAMIMAS_A21B_0669` | train | DAMIMAS | 0.75 | 0 | 0 | 2 | 1 |
| `DAMIMAS_A21B_0702` | train | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0708` | val | DAMIMAS | 0.75 | 1 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0712` | test | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0716` | test | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0721` | train | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0723` | train | DAMIMAS | 1.25 | 0 | 1 | 2 | 2 |
| `DAMIMAS_A21B_0726` | test | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0727` | test | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0729` | train | DAMIMAS | 0.75 | 1 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0732` | train | DAMIMAS | 0.50 | 0 | 0 | 0 | 2 |
| `DAMIMAS_A21B_0741` | train | DAMIMAS | 1.00 | 0 | 1 | 2 | 1 |
| `DAMIMAS_A21B_0747` | train | DAMIMAS | 1.25 | 0 | 1 | 2 | 2 |
| `DAMIMAS_A21B_0759` | val | DAMIMAS | 0.75 | 0 | 0 | 2 | 1 |
| `DAMIMAS_A21B_0774` | train | DAMIMAS | 0.75 | 0 | 0 | 3 | 0 |
| `DAMIMAS_A21B_0778` | test | DAMIMAS | 0.75 | 0 | 0 | 2 | 1 |
| `DAMIMAS_A21B_0810` | val | DAMIMAS | 4.00 | 1 | 2 | 11 | 2 |
| `DAMIMAS_A21B_0811` | train | DAMIMAS | 3.00 | 3 | 1 | 8 | 0 |
| `DAMIMAS_A21B_0812` | val | DAMIMAS | 3.25 | 5 | 3 | 5 | 0 |
| `DAMIMAS_A21B_0813` | test | DAMIMAS | 1.50 | 1 | 0 | 4 | 1 |
| `DAMIMAS_A21B_0814` | test | DAMIMAS | 1.50 | 0 | 0 | 5 | 1 |
| `DAMIMAS_A21B_0815` | train | DAMIMAS | 4.75 | 2 | 4 | 13 | 0 |
| `DAMIMAS_A21B_0816` | train | DAMIMAS | 2.50 | 2 | 4 | 4 | 0 |
| `DAMIMAS_A21B_0817` | val | DAMIMAS | 1.75 | 4 | 2 | 1 | 0 |
| `DAMIMAS_A21B_0818` | train | DAMIMAS | 4.00 | 0 | 7 | 9 | 0 |
| `DAMIMAS_A21B_0819` | train | DAMIMAS | 3.75 | 0 | 4 | 10 | 1 |
| `DAMIMAS_A21B_0820` | train | DAMIMAS | 5.00 | 0 | 5 | 15 | 0 |
| `DAMIMAS_A21B_0821` | val | DAMIMAS | 1.25 | 3 | 1 | 1 | 0 |
| `DAMIMAS_A21B_0822` | test | DAMIMAS | 2.75 | 1 | 1 | 9 | 0 |
| `DAMIMAS_A21B_0823` | train | DAMIMAS | 5.50 | 3 | 2 | 14 | 3 |
| `DAMIMAS_A21B_0824` | val | DAMIMAS | 3.75 | 3 | 8 | 4 | 0 |
| `DAMIMAS_A21B_0825` | val | DAMIMAS | 1.75 | 2 | 0 | 4 | 1 |
| `DAMIMAS_A21B_0826` | val | DAMIMAS | 1.50 | 0 | 3 | 3 | 0 |
| `DAMIMAS_A21B_0827` | train | DAMIMAS | 4.25 | 2 | 2 | 11 | 2 |
| `DAMIMAS_A21B_0828` | train | DAMIMAS | 3.75 | 1 | 7 | 7 | 0 |
| `DAMIMAS_A21B_0829` | val | DAMIMAS | 2.25 | 2 | 2 | 4 | 1 |
| `DAMIMAS_A21B_0830` | train | DAMIMAS | 2.25 | 4 | 2 | 3 | 0 |
| `DAMIMAS_A21B_0831` | train | DAMIMAS | 5.25 | 1 | 4 | 13 | 3 |
| `DAMIMAS_A21B_0832` | train | DAMIMAS | 3.00 | 0 | 5 | 6 | 1 |
| `DAMIMAS_A21B_0833` | train | DAMIMAS | 1.00 | 2 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0834` | train | DAMIMAS | 3.50 | 1 | 5 | 5 | 3 |
| `DAMIMAS_A21B_0835` | train | DAMIMAS | 2.25 | 0 | 0 | 9 | 0 |
| `DAMIMAS_A21B_0836` | train | DAMIMAS | 3.25 | 3 | 3 | 7 | 0 |
| `DAMIMAS_A21B_0837` | train | DAMIMAS | 2.00 | 3 | 0 | 5 | 0 |
| `DAMIMAS_A21B_0838` | train | DAMIMAS | 3.00 | 0 | 1 | 5 | 6 |
| `DAMIMAS_A21B_0839` | val | DAMIMAS | 3.25 | 2 | 3 | 8 | 0 |
| `DAMIMAS_A21B_0840` | train | DAMIMAS | 3.00 | 3 | 5 | 4 | 0 |
| `DAMIMAS_A21B_0841` | train | DAMIMAS | 0.75 | 1 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0842` | train | DAMIMAS | 4.25 | 3 | 2 | 8 | 4 |
| `DAMIMAS_A21B_0843` | test | DAMIMAS | 1.75 | 0 | 2 | 5 | 0 |
| `DAMIMAS_A21B_0844` | train | DAMIMAS | 3.50 | 1 | 5 | 7 | 1 |
| `DAMIMAS_A21B_0846` | train | DAMIMAS | 1.75 | 0 | 4 | 3 | 0 |
| `DAMIMAS_A21B_0847` | test | DAMIMAS | 3.25 | 2 | 2 | 9 | 0 |
| `DAMIMAS_A21B_0848` | val | DAMIMAS | 2.25 | 0 | 2 | 7 | 0 |
| `DAMIMAS_A21B_0849` | train | DAMIMAS | 1.00 | 2 | 1 | 1 | 0 |
| `DAMIMAS_A21B_0850` | train | DAMIMAS | 4.00 | 2 | 4 | 9 | 1 |
| `DAMIMAS_A21B_0851` | train | DAMIMAS | 3.00 | 0 | 5 | 6 | 1 |
| `DAMIMAS_A21B_0852` | train | DAMIMAS | 0.50 | 0 | 2 | 0 | 0 |
| `DAMIMAS_A21B_0853` | train | DAMIMAS | 1.25 | 2 | 1 | 2 | 0 |
| `DAMIMAS_A21B_0854` | train | DAMIMAS | 3.50 | 4 | 3 | 5 | 2 |
| `LONSUM_A21A_0003` | val | LONSUM | 0.50 | 0 | 0 | 0 | 2 |
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
| DAMIMAS_A21B_0006 | train | True | 1 | 0 | 5 | 3 | 1 | 0 | 4 | 3 |
| DAMIMAS_A21B_0007 | train | True | 3 | 3 | 8 | 3 | 3 | 2 | 8 | 3 |
| DAMIMAS_A21B_0008 | train | True | 2 | 3 | 4 | 0 | 2 | 3 | 4 | 0 |
| DAMIMAS_A21B_0009 | train | True | 0 | 0 | 3 | 3 | 0 | 0 | 3 | 2 |
| DAMIMAS_A21B_0010 | train | True | 0 | 2 | 6 | 3 | 0 | 3 | 6 | 3 |
