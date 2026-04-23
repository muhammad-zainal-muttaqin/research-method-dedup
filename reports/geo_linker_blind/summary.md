# Blind Run GeoLinker — 725 pohon non-JSON

**Config:** `{'T_y': 0.05, 'T_s': 2.0, 'lam_s': 0.0, 'lam_adj': 0.08, 'iou_intra': 0.5, 'T_cost': 0.130001, 'adjacent_only': False, 'T_y_opp': 0.03, 'mutual_best': False, 'per_class_T_y': {}}`

## 1. Ringkasan

| Item | Nilai |
|---|---:|
| Total pohon blind | 725 |
| Pohon 4-sisi | 680 |
| Pohon 8-sisi | 45 |
| Domain DAMIMAS | 626 |
| Domain LONSUM | 99 |
| Σ PRED bunches | 7,832 |
| Σ NAIVE bunches | 13,665 |
| Aggregate PRED/NAIVE | 0.573 |
| Referensi JSON-228 (unique/naive) | 0.559 |

## 2. Per-Kelas Total

| Kelas | PRED | NAIVE | PRED/NAIVE | Referensi JSON-228 ratio |
|---|---:|---:|---:|---:|
| B1 | 1,003 | 1,618 | 0.620 | 0.559 |
| B2 | 1,718 | 2,974 | 0.578 | 0.559 |
| B3 | 3,453 | 6,417 | 0.538 | 0.559 |
| B4 | 1,658 | 2,656 | 0.624 | 0.559 |

## 3. Distribusi naive_ratio (per pohon)

- min   : 0.000
- median: 0.591
- mean  : 0.593
- max   : 1.000
- n_outlier (>25 pred atau ratio <0.30 atau >0.90): **39**

## 4. Per Split

| Split | n | Σ PRED | Σ NAIVE | ratio |
|---|---:|---:|---:|---:|
| train | 505 | 5,513 | 9,577 | 0.576 |
| val | 112 | 1,182 | 2,137 | 0.553 |
| test | 108 | 1,137 | 1,951 | 0.583 |

## 5. Per Domain

| Domain | n | Σ PRED | Σ NAIVE | ratio |
|---|---:|---:|---:|---:|
| DAMIMAS | 626 | 7,197 | 12,643 | 0.569 |
| LONSUM | 99 | 635 | 1,022 | 0.621 |

## 6. Top-10 Outlier (manual review)

| tree_id | split | n_sides | total_pred | total_naive | ratio |
|---|---|---:|---:|---:|---:|
| DAMIMAS_A21B_0429 | train | 4 | 28 | 36 | 0.778 |
| DAMIMAS_A21B_0243 | val | 4 | 20 | 22 | 0.909 |
| DAMIMAS_A21B_0291 | train | 4 | 20 | 22 | 0.909 |
| DAMIMAS_A21B_0488 | train | 4 | 16 | 17 | 0.941 |
| DAMIMAS_A21B_0851 | train | 8 | 15 | 51 | 0.294 |
| DAMIMAS_A21B_0811 | train | 8 | 14 | 48 | 0.292 |
| DAMIMAS_A21B_0840 | train | 8 | 13 | 45 | 0.289 |
| DAMIMAS_A21B_0296 | test | 4 | 12 | 12 | 1.000 |
| DAMIMAS_A21B_0300 | train | 4 | 12 | 13 | 0.923 |
| DAMIMAS_A21B_0812 | val | 8 | 11 | 39 | 0.282 |

## 7. Catatan

- **Tidak ada GT untuk 725 pohon ini** → metrik akurasi tidak bisa dihitung langsung.
- Sanity check: ratio PRED/NAIVE ~0.56 (referensi JSON-228) menandakan algoritma berperilaku konsisten.
- Drift per-kelas besar (terutama B3) mengindikasikan distribusi pohon non-JSON memang beda dari ber-JSON (mis. LONSUM lebih B3-heavy).
- `naive_ratio` per pohon dipakai sebagai sanity proxy; outlier flagging untuk manual review.