# Laporan GeoLinker — GT-Based Dedup (228 pohon JSON)
**Config:** `{'T_y': 0.05, 'T_s': 2.0, 'lam_s': 0.0, 'lam_adj': 0.08, 'iou_intra': 0.5, 'T_cost': 0.130001, 'adjacent_only': False, 'T_y_opp': 0.03, 'mutual_best': False, 'per_class_T_y': {}}`

## 1. Metrik Headline (Seluruh 228 pohon)

| Metrik | Nilai |
|---|---:|
| n_pohon | 228 |
| `pct_exact` (pred == gt) | **20.2%** |
| `pct_within_1` (≤ 1 error) | **52.2%** |
| `pct_within_2` (≤ 2 error) | **71.9%** |
| MAE total (tree-level) | 1.811 |
| MAE B1 | 0.461 |
| MAE B2 | 0.566 |
| MAE B3 | 1.000 |
| MAE B4 | 0.513 |
| Σ GT bunches | 2,466 |
| Σ PRED bunches | 2,507 |
| Σ NAIVE bunches | 4,408 |
| Aggregate pred/gt ratio | 1.0166 |
| Aggregate naive/gt ratio | 1.7875 |

## 2. Perbandingan vs Baseline

| Metode | pct_within_1 | pct_within_2 | MAE total | ratio pred/gt |
|---|---:|---:|---:|---:|
| **Naive sum** | 0.4% | 2.2% | 8.518 | 1.788 |
| **GeoLinker (ini)** | **52.2%** | **71.9%** | **1.811** | 1.017 |

## 3. Breakdown per Split

| Split | n | pct_exact | pct_within_1 | pct_within_2 | MAE total |
|---|---:|---:|---:|---:|---:|
| train | 196 | 20.4% | 55.1% | 75.0% | 1.704 |
| val | 1 | 0.0% | 0.0% | 0.0% | 3.000 |
| test | 31 | 19.4% | 35.5% | 54.8% | 2.452 |

## 4. Distribusi Signed Error (pred_total − gt_total)

| Error | n_pohon | % |
|---:|---:|---:|
| -8 | 2 | 0.9% |
| -7 | 1 | 0.4% |
| -6 | 1 | 0.4% |
| -4 | 6 | 2.6% |
| -3 | 15 | 6.6% |
| -2 | 24 | 10.5% |
| -1 | 40 | 17.5% |
| +0 | 46 | 20.2% |
| +1 | 33 | 14.5% |
| +2 | 21 | 9.2% |
| +3 | 18 | 7.9% |
| +4 | 10 | 4.4% |
| +5 | 9 | 3.9% |
| +6 | 1 | 0.4% |
| +7 | 1 | 0.4% |

## 5. Top-10 Pohon dengan Error Terbesar

| tree_id | split | GT total | PRED total | NAIVE total | err |
|---|---|---:|---:|---:|---:|
| DAMIMAS_A21B_0268 | train | 19 | 11 | 28 | 8 |
| DAMIMAS_A21B_0576 | test | 16 | 8 | 25 | 8 |
| DAMIMAS_A21B_0275 | train | 17 | 10 | 26 | 7 |
| DAMIMAS_A21B_0554 | test | 14 | 21 | 31 | 7 |
| DAMIMAS_A21B_0625 | train | 15 | 21 | 27 | 6 |
| DAMIMAS_A21B_0571 | test | 16 | 10 | 25 | 6 |
| DAMIMAS_A21B_0003 | train | 9 | 14 | 20 | 5 |
| DAMIMAS_A21B_0579 | train | 15 | 20 | 28 | 5 |
| DAMIMAS_A21B_0247 | train | 10 | 15 | 20 | 5 |
| DAMIMAS_A21B_0014 | train | 9 | 14 | 18 | 5 |

## 6. Catatan

- Algoritma murni geometri (tanpa ML/embedding).
- Input = GT bbox + class (label bersih per JSON-01 verdict).
- Fitur: |Δcy_center|, log rasio area bbox.
- Constraint: intra-kelas saja, one-per-side per cluster, adjacency-aware (sisi opposite threshold lebih ketat).
- Ceiling: pure geometri tidak bisa memisahkan dua tandan fisik yang berdekatan-ketinggian tanpa cue visual.
- Reproduce: `python scripts/tune_geo_linker.py && python scripts/eval_geo_linker.py`