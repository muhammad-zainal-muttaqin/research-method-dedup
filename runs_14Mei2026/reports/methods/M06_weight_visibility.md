# `M06_weight_visibility` — Primary Metrics Breakdown

**Implementasi:** [`algorithms/M06_weight_visibility.py`](../../algorithms/M06_weight_visibility.py)  
**Dataset:** 228 pohon JSON (953 baris cocok dengan `method=M06_weight_visibility`)  
**Raw data lengkap:** [`../benchmark_multidim/accuracy_per_tree.csv`](../benchmark_multidim/accuracy_per_tree.csv)  
**Per-method slice (filter sudah diterapkan):** [`M06_weight_visibility_per_tree.csv`](M06_weight_visibility_per_tree.csv)  
**Summary CSV:** [`../benchmark_multidim/accuracy_summary.csv`](../benchmark_multidim/accuracy_summary.csv)

Seluruh angka di bawah dihitung ulang dari `accuracy_per_tree.csv` oleh `scripts/generate_method_reports.py`.

## Primary Metrics

| Metric | Value | Derivation |
|---|---:|---|
| Macro class-MAE | **0.3864** | mean(per-class MAE) |
| Exact accuracy | **24.34%** | 232/953 pohon dengan err_B* = 0 di semua kelas |
| Total count MAE | **1.3358** | mean \|Σpred − Σgt\| per pohon |
| Total ±1 accuracy | **73.77%** | 703/953 pohon dengan \|Σpred − Σgt\| ≤ 1 |
| Acc ±1 per kelas per pohon (pelengkap) | 85.73% | 817/953 pohon dengan semua err_B* dalam ±1 |

## Per-Class MAE

Sumber: kolom `err_B*` di `accuracy_per_tree.csv` (sudah absolute).

| Class | MAE | Derivation |
|---|---:|---|
| B1 | **0.2046** | mean(err_B1) across 953 pohon |
| B2 | **0.3190** | mean(err_B2) across 953 pohon |
| B3 | **0.7272** | mean(err_B3) across 953 pohon |
| B4 | **0.2949** | mean(err_B4) across 953 pohon |

Cross-check versus [`accuracy_per_class.csv`](../benchmark_multidim/accuracy_per_class.csv):

| Class | MAE (csv) | over_count | under_count | exact | within1 | pct_within1 |
|---|---:|---:|---:|---:|---:|---:|
| B1 | 0.2046 | 26 | 0 | 798 | 927 | 97.27% |
| B2 | 0.3190 | 32 | 12 | 731 | 909 | 95.38% |
| B3 | 0.7272 | 64 | 28 | 477 | 861 | 90.35% |
| B4 | 0.2949 | 5 | 24 | 704 | 924 | 96.96% |

## Per-Class Mean Error (Bias)

Sumber: `pred_B* − gt_B*` di `accuracy_per_tree.csv`. Nilai `+` = overcount, `−` = undercount, `0` = tidak bias.

| Class | Mean Error | Derivation |
|---|---:|---|
| B1 | **+0.196** | mean(pred_B1 − gt_B1) across 953 pohon |
| B2 | **+0.141** | mean(pred_B2 − gt_B2) across 953 pohon |
| B3 | **+0.217** | mean(pred_B3 − gt_B3) across 953 pohon |
| B4 | **-0.186** | mean(pred_B4 − gt_B4) across 953 pohon |

## Kecepatan (pelengkap)

Sumber: [`speed_summary.csv`](../benchmark_multidim/speed_summary.csv) (30 repetisi × 953 pohon)

- Mean: **0.0372 ms/pohon** (26,885 pohon/detik)
- Median: 0.0372 ms
- Std: 0.0002 ms

## Robustness terhadap Noise Koordinat (pelengkap)

Sumber: [`robustness_summary.csv`](../benchmark_multidim/robustness_summary.csv)

| σ (noise_pct) | Acc ±1 | MAE | n_fail | Acc drop vs σ=0 |
|---:|---:|---:|---:|---:|
| 0% | 85.73% | 0.3864 | 136 | +0.00% |
| 5% | 85.41% | 0.3885 | 139 | +0.32% |
| 10% | 85.31% | 0.3922 | 140 | +0.42% |
| 20% | 82.79% | 0.4213 | 164 | +2.94% |

## Pohon yang Gagal (Acc±1 fail = 136)

| tree_id | split | domain | MAE | err_B1 | err_B2 | err_B3 | err_B4 |
|---|---|---|---:|---:|---:|---:|---:|
| `DAMIMAS_A21B_0034` | train | DAMIMAS | 0.50 | 0 | 0 | 0 | 2 |
| `DAMIMAS_A21B_0035` | train | DAMIMAS | 0.50 | 0 | 2 | 0 | 0 |
| `DAMIMAS_A21B_0073` | test | DAMIMAS | 0.50 | 0 | 0 | 0 | 2 |
| `DAMIMAS_A21B_0102` | train | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0133` | train | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0139` | train | DAMIMAS | 0.75 | 0 | 0 | 2 | 1 |
| `DAMIMAS_A21B_0141` | train | DAMIMAS | 0.75 | 0 | 0 | 2 | 1 |
| `DAMIMAS_A21B_0143` | train | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0147` | val | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0151` | val | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0152` | val | DAMIMAS | 0.75 | 0 | 1 | 2 | 0 |
| `DAMIMAS_A21B_0162` | val | DAMIMAS | 1.00 | 0 | 0 | 3 | 1 |
| `DAMIMAS_A21B_0175` | val | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0176` | val | DAMIMAS | 0.75 | 1 | 0 | 0 | 2 |
| `DAMIMAS_A21B_0177` | val | DAMIMAS | 1.00 | 0 | 1 | 3 | 0 |
| `DAMIMAS_A21B_0201` | test | DAMIMAS | 0.75 | 1 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0208` | test | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0214` | test | DAMIMAS | 0.50 | 0 | 0 | 0 | 2 |
| `DAMIMAS_A21B_0221` | test | DAMIMAS | 1.00 | 0 | 1 | 2 | 1 |
| `DAMIMAS_A21B_0234` | test | DAMIMAS | 0.75 | 0 | 0 | 1 | 2 |
| `DAMIMAS_A21B_0237` | test | DAMIMAS | 1.00 | 1 | 1 | 2 | 0 |
| `DAMIMAS_A21B_0243` | val | DAMIMAS | 0.75 | 1 | 2 | 0 | 0 |
| `DAMIMAS_A21B_0246` | test | DAMIMAS | 1.25 | 0 | 2 | 2 | 1 |
| `DAMIMAS_A21B_0257` | train | DAMIMAS | 0.75 | 0 | 1 | 2 | 0 |
| `DAMIMAS_A21B_0258` | train | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0259` | train | DAMIMAS | 0.50 | 0 | 2 | 0 | 0 |
| `DAMIMAS_A21B_0268` | train | DAMIMAS | 1.00 | 0 | 2 | 1 | 1 |
| `DAMIMAS_A21B_0278` | val | DAMIMAS | 0.75 | 1 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0281` | train | DAMIMAS | 0.75 | 0 | 0 | 2 | 1 |
| `DAMIMAS_A21B_0289` | val | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0291` | train | DAMIMAS | 1.00 | 0 | 1 | 2 | 1 |
| `DAMIMAS_A21B_0293` | train | DAMIMAS | 1.00 | 0 | 1 | 2 | 1 |
| `DAMIMAS_A21B_0309` | train | DAMIMAS | 0.75 | 2 | 1 | 0 | 0 |
| `DAMIMAS_A21B_0311` | val | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0315` | train | DAMIMAS | 0.75 | 0 | 2 | 1 | 0 |
| `DAMIMAS_A21B_0318` | train | DAMIMAS | 0.50 | 0 | 0 | 0 | 2 |
| `DAMIMAS_A21B_0320` | train | DAMIMAS | 0.75 | 0 | 1 | 0 | 2 |
| `DAMIMAS_A21B_0321` | test | DAMIMAS | 0.75 | 1 | 2 | 0 | 0 |
| `DAMIMAS_A21B_0323` | test | DAMIMAS | 1.00 | 0 | 0 | 4 | 0 |
| `DAMIMAS_A21B_0324` | train | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0331` | test | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0334` | train | DAMIMAS | 0.75 | 0 | 0 | 1 | 2 |
| `DAMIMAS_A21B_0335` | train | DAMIMAS | 0.75 | 2 | 0 | 1 | 0 |
| `DAMIMAS_A21B_0336` | train | DAMIMAS | 1.00 | 2 | 0 | 1 | 1 |
| `DAMIMAS_A21B_0341` | train | DAMIMAS | 1.00 | 0 | 0 | 4 | 0 |
| `DAMIMAS_A21B_0342` | val | DAMIMAS | 0.75 | 0 | 2 | 1 | 0 |
| `DAMIMAS_A21B_0343` | train | DAMIMAS | 0.75 | 0 | 2 | 1 | 0 |
| `DAMIMAS_A21B_0348` | test | DAMIMAS | 0.75 | 1 | 0 | 0 | 2 |
| `DAMIMAS_A21B_0358` | train | DAMIMAS | 0.75 | 0 | 0 | 2 | 1 |
| `DAMIMAS_A21B_0361` | train | DAMIMAS | 1.00 | 1 | 0 | 1 | 2 |
| `DAMIMAS_A21B_0362` | train | DAMIMAS | 1.75 | 0 | 0 | 7 | 0 |
| `DAMIMAS_A21B_0382` | train | DAMIMAS | 0.50 | 0 | 0 | 0 | 2 |
| `DAMIMAS_A21B_0388` | val | DAMIMAS | 1.00 | 1 | 0 | 3 | 0 |
| `DAMIMAS_A21B_0389` | train | DAMIMAS | 0.75 | 0 | 0 | 2 | 1 |
| `DAMIMAS_A21B_0395` | train | DAMIMAS | 0.75 | 0 | 2 | 1 | 0 |
| `DAMIMAS_A21B_0396` | train | DAMIMAS | 0.75 | 0 | 0 | 1 | 2 |
| `DAMIMAS_A21B_0398` | val | DAMIMAS | 1.00 | 1 | 0 | 2 | 1 |
| `DAMIMAS_A21B_0410` | train | DAMIMAS | 0.75 | 1 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0414` | train | DAMIMAS | 0.75 | 0 | 0 | 3 | 0 |
| `DAMIMAS_A21B_0418` | train | DAMIMAS | 0.50 | 0 | 0 | 0 | 2 |
| `DAMIMAS_A21B_0430` | train | DAMIMAS | 1.25 | 0 | 0 | 3 | 2 |
| `DAMIMAS_A21B_0437` | train | DAMIMAS | 1.25 | 0 | 1 | 2 | 2 |
| `DAMIMAS_A21B_0438` | val | DAMIMAS | 0.75 | 0 | 0 | 1 | 2 |
| `DAMIMAS_A21B_0440` | train | DAMIMAS | 1.00 | 0 | 1 | 2 | 1 |
| `DAMIMAS_A21B_0442` | test | DAMIMAS | 0.75 | 0 | 0 | 2 | 1 |
| `DAMIMAS_A21B_0447` | train | DAMIMAS | 1.25 | 0 | 2 | 2 | 1 |
| `DAMIMAS_A21B_0450` | train | DAMIMAS | 1.00 | 1 | 0 | 2 | 1 |
| `DAMIMAS_A21B_0454` | train | DAMIMAS | 1.00 | 0 | 1 | 1 | 2 |
| `DAMIMAS_A21B_0471` | val | DAMIMAS | 0.75 | 0 | 2 | 1 | 0 |
| `DAMIMAS_A21B_0472` | val | DAMIMAS | 0.75 | 0 | 0 | 2 | 1 |
| `DAMIMAS_A21B_0494` | train | DAMIMAS | 0.75 | 0 | 0 | 1 | 2 |
| `DAMIMAS_A21B_0497` | test | DAMIMAS | 1.00 | 0 | 1 | 2 | 1 |
| `DAMIMAS_A21B_0522` | train | DAMIMAS | 0.50 | 0 | 2 | 0 | 0 |
| `DAMIMAS_A21B_0523` | train | DAMIMAS | 0.50 | 0 | 2 | 0 | 0 |
| `DAMIMAS_A21B_0524` | train | DAMIMAS | 0.50 | 0 | 0 | 0 | 2 |
| `DAMIMAS_A21B_0545` | train | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0558` | train | DAMIMAS | 0.50 | 0 | 2 | 0 | 0 |
| `DAMIMAS_A21B_0559` | train | DAMIMAS | 0.75 | 0 | 0 | 0 | 3 |
| `DAMIMAS_A21B_0571` | train | DAMIMAS | 0.50 | 0 | 2 | 0 | 0 |
| `DAMIMAS_A21B_0632` | train | DAMIMAS | 0.75 | 0 | 0 | 2 | 1 |
| `DAMIMAS_A21B_0656` | train | DAMIMAS | 0.75 | 0 | 0 | 1 | 2 |
| `DAMIMAS_A21B_0702` | train | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0732` | train | DAMIMAS | 0.75 | 0 | 0 | 1 | 2 |
| `DAMIMAS_A21B_0737` | test | DAMIMAS | 0.50 | 0 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0759` | val | DAMIMAS | 0.75 | 0 | 0 | 2 | 1 |
| `DAMIMAS_A21B_0774` | train | DAMIMAS | 0.75 | 0 | 0 | 3 | 0 |
| `DAMIMAS_A21B_0778` | test | DAMIMAS | 0.75 | 0 | 0 | 2 | 1 |
| `DAMIMAS_A21B_0810` | val | DAMIMAS | 3.00 | 1 | 2 | 8 | 1 |
| `DAMIMAS_A21B_0811` | train | DAMIMAS | 2.25 | 3 | 1 | 5 | 0 |
| `DAMIMAS_A21B_0812` | val | DAMIMAS | 3.00 | 5 | 3 | 4 | 0 |
| `DAMIMAS_A21B_0813` | test | DAMIMAS | 1.25 | 1 | 0 | 3 | 1 |
| `DAMIMAS_A21B_0814` | test | DAMIMAS | 1.50 | 0 | 0 | 5 | 1 |
| `DAMIMAS_A21B_0815` | train | DAMIMAS | 3.50 | 2 | 3 | 9 | 0 |
| `DAMIMAS_A21B_0816` | train | DAMIMAS | 1.50 | 2 | 3 | 1 | 0 |
| `DAMIMAS_A21B_0817` | val | DAMIMAS | 2.00 | 5 | 2 | 1 | 0 |
| `DAMIMAS_A21B_0818` | train | DAMIMAS | 3.00 | 0 | 6 | 6 | 0 |
| `DAMIMAS_A21B_0819` | train | DAMIMAS | 2.50 | 0 | 3 | 6 | 1 |
| `DAMIMAS_A21B_0820` | train | DAMIMAS | 4.00 | 0 | 5 | 11 | 0 |
| `DAMIMAS_A21B_0821` | val | DAMIMAS | 1.00 | 3 | 1 | 0 | 0 |
| `DAMIMAS_A21B_0822` | test | DAMIMAS | 2.25 | 1 | 1 | 7 | 0 |
| `DAMIMAS_A21B_0823` | train | DAMIMAS | 4.25 | 2 | 2 | 11 | 2 |
| `DAMIMAS_A21B_0824` | val | DAMIMAS | 3.00 | 3 | 7 | 2 | 0 |
| `DAMIMAS_A21B_0825` | val | DAMIMAS | 1.50 | 2 | 0 | 3 | 1 |
| `DAMIMAS_A21B_0826` | val | DAMIMAS | 1.25 | 0 | 3 | 2 | 0 |
| `DAMIMAS_A21B_0827` | train | DAMIMAS | 3.50 | 2 | 1 | 9 | 2 |
| `DAMIMAS_A21B_0828` | train | DAMIMAS | 2.75 | 1 | 6 | 4 | 0 |
| `DAMIMAS_A21B_0829` | val | DAMIMAS | 1.75 | 2 | 2 | 3 | 0 |
| `DAMIMAS_A21B_0830` | train | DAMIMAS | 2.00 | 4 | 2 | 2 | 0 |
| `DAMIMAS_A21B_0831` | train | DAMIMAS | 3.75 | 1 | 3 | 9 | 2 |
| `DAMIMAS_A21B_0832` | train | DAMIMAS | 2.75 | 0 | 5 | 5 | 1 |
| `DAMIMAS_A21B_0833` | train | DAMIMAS | 1.25 | 3 | 0 | 2 | 0 |
| `DAMIMAS_A21B_0834` | train | DAMIMAS | 2.50 | 1 | 4 | 3 | 2 |
| `DAMIMAS_A21B_0835` | train | DAMIMAS | 1.50 | 0 | 0 | 5 | 1 |
| `DAMIMAS_A21B_0836` | train | DAMIMAS | 2.50 | 2 | 3 | 5 | 0 |
| `DAMIMAS_A21B_0837` | train | DAMIMAS | 1.50 | 3 | 0 | 3 | 0 |
| `DAMIMAS_A21B_0838` | train | DAMIMAS | 2.25 | 0 | 1 | 4 | 4 |
| `DAMIMAS_A21B_0839` | val | DAMIMAS | 2.50 | 2 | 2 | 6 | 0 |
| `DAMIMAS_A21B_0840` | train | DAMIMAS | 2.25 | 2 | 5 | 2 | 0 |
| `DAMIMAS_A21B_0842` | train | DAMIMAS | 2.50 | 2 | 1 | 5 | 2 |
| `DAMIMAS_A21B_0843` | test | DAMIMAS | 1.50 | 0 | 2 | 4 | 0 |
| `DAMIMAS_A21B_0844` | train | DAMIMAS | 2.00 | 1 | 4 | 3 | 0 |
| `DAMIMAS_A21B_0846` | train | DAMIMAS | 1.75 | 0 | 4 | 3 | 0 |
| `DAMIMAS_A21B_0847` | test | DAMIMAS | 2.25 | 2 | 1 | 6 | 0 |
| `DAMIMAS_A21B_0848` | val | DAMIMAS | 2.00 | 0 | 2 | 6 | 0 |
| `DAMIMAS_A21B_0849` | train | DAMIMAS | 1.00 | 2 | 1 | 1 | 0 |
| `DAMIMAS_A21B_0850` | train | DAMIMAS | 2.75 | 2 | 3 | 6 | 0 |
| `DAMIMAS_A21B_0851` | train | DAMIMAS | 2.25 | 0 | 4 | 4 | 1 |
| `DAMIMAS_A21B_0852` | train | DAMIMAS | 0.50 | 0 | 2 | 0 | 0 |
| `DAMIMAS_A21B_0853` | train | DAMIMAS | 1.00 | 2 | 1 | 1 | 0 |
| `DAMIMAS_A21B_0854` | train | DAMIMAS | 2.00 | 3 | 2 | 3 | 0 |
| `LONSUM_A21A_0001` | train | LONSUM | 0.50 | 0 | 0 | 2 | 0 |
| `LONSUM_A21A_0003` | val | LONSUM | 0.50 | 0 | 0 | 0 | 2 |
| `LONSUM_A21A_0091` | val | LONSUM | 0.50 | 0 | 0 | 2 | 0 |
| `LONSUM_A21A_0092` | train | LONSUM | 0.50 | 0 | 0 | 2 | 0 |
| `LONSUM_A21A_0096` | train | LONSUM | 0.50 | 0 | 0 | 2 | 0 |
| `LONSUM_A21A_0099` | test | LONSUM | 0.50 | 0 | 0 | 2 | 0 |

## Sample 10 Baris Per-Tree

Kolom penuh tersedia di per-method CSV di atas. Preview:

| tree_id | split | ok | gt_B1 | gt_B2 | gt_B3 | gt_B4 | pred_B1 | pred_B2 | pred_B3 | pred_B4 |
|---|---|---|---|---|---|---|---|---|---|---|
| DAMIMAS_A21B_0001 | train | True | 1 | 2 | 5 | 0 | 1 | 3 | 5 | 0 |
| DAMIMAS_A21B_0002 | train | True | 1 | 0 | 7 | 4 | 2 | 0 | 6 | 5 |
| DAMIMAS_A21B_0003 | train | True | 1 | 2 | 5 | 1 | 1 | 3 | 6 | 1 |
| DAMIMAS_A21B_0004 | train | True | 0 | 0 | 8 | 0 | 0 | 0 | 8 | 0 |
| DAMIMAS_A21B_0005 | test | True | 1 | 3 | 3 | 2 | 1 | 3 | 3 | 1 |
| DAMIMAS_A21B_0006 | train | True | 1 | 0 | 5 | 3 | 1 | 0 | 4 | 3 |
| DAMIMAS_A21B_0007 | train | True | 3 | 3 | 8 | 3 | 3 | 2 | 8 | 3 |
| DAMIMAS_A21B_0008 | train | True | 2 | 3 | 4 | 0 | 2 | 3 | 4 | 0 |
| DAMIMAS_A21B_0009 | train | True | 0 | 0 | 3 | 3 | 0 | 0 | 3 | 2 |
| DAMIMAS_A21B_0010 | train | True | 0 | 2 | 6 | 3 | 0 | 3 | 6 | 3 |
