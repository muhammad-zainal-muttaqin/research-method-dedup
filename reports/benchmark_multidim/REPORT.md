# Benchmark Multi-Dimensi: 11 Algoritma Dedup

**Dataset:** 727 pohon JSON (228 GT)  
**Tanggal:** 2026-04-24  
**Metrik utama:** Acc ±1 (semua kelas dalam 1 error), MAE, ms/pohon

---

## Dimensi 1: Akurasi (Acc ±1 per kelas)

Pohon dianggap **benar** jika semua 4 kelas masing-masing dalam ±1 dari GT.

| Rank | Method | Gen | Acc ±1 | MAE | MTE | Gagal |
|---:|---|---|---:|---:|---:|---:|
| 1 | `v2_visibility` | v2 | **89.41%** | 0.3116 | 1.2462 | 77 |
| 2 | `v5_best_visibility` | v5 | **89.41%** | 0.3116 | 1.2462 | 77 |
| 3 | `v9_selector` | v9 | **89.27%** | 0.3267 | 1.3067 | 78 |
| 4 | `v9_b2_median_v6` | v9 | **89.00%** | 0.3219 | 1.2875 | 80 |
| 5 | `v6_selector` | v6 | **88.86%** | 0.3287 | 1.3150 | 81 |
| 6 | `v8_entropy_modulated` | v8 | **88.86%** | 0.3394 | 1.3576 | 81 |
| 7 | `v7_stacking_bracketed` | v7 | **88.45%** | 0.3181 | 1.2724 | 84 |
| 8 | `v7_stacking_density` | v7 | **88.45%** | 0.3232 | 1.2930 | 84 |
| 9 | `v1_corrected` | v1 | **87.90%** | 0.3291 | 1.3164 | 88 |
| 10 | `v8_b2_b4_boosted` | v8 | **87.62%** | 0.3067 | 1.2270 | 90 |
| 11 | `v5_adaptive_corrected` | v5 | **86.11%** | 0.3494 | 1.3975 | 101 |

> MTE = Mean Total Error (jumlah absolut error semua kelas, rata-rata per pohon)

### Akurasi Per Kelas (% pohon dalam ±1)

| Method | B1 | B2 | B3 | B4 |
|---|---:|---:|---:|---:|
| `v2_visibility` | 99.3% | 97.8% | 94.1% | 97.4% |
| `v5_best_visibility` | 99.3% | 97.8% | 94.1% | 97.4% |
| `v9_selector` | 99.6% | 97.7% | 92.4% | 98.2% |
| `v9_b2_median_v6` | 99.6% | 97.9% | 92.2% | 98.1% |
| `v6_selector` | 99.6% | 97.7% | 92.2% | 98.1% |
| `v8_entropy_modulated` | 99.6% | 97.9% | 91.8% | 98.1% |
| `v7_stacking_bracketed` | 99.7% | 97.9% | 91.6% | 97.7% |
| `v7_stacking_density` | 99.7% | 97.9% | 91.6% | 97.7% |
| `v1_corrected` | 99.3% | 97.8% | 91.8% | 97.8% |
| `v8_b2_b4_boosted` | 99.7% | 97.7% | 91.6% | 97.2% |
| `v5_adaptive_corrected` | 99.6% | 97.7% | 89.7% | 97.5% |

### Pola Error Per Kelas (over >1 / under <-1, jumlah pohon)

| Method | B1↑ | B1↓ | B2↑ | B2↓ | B3↑ | B3↓ | B4↑ | B4↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `v2_visibility` | 4 | 1 | 4 | 12 | 23 | 20 | 1 | 18 |
| `v5_best_visibility` | 4 | 1 | 4 | 12 | 23 | 20 | 1 | 18 |
| `v9_selector` | 2 | 1 | 10 | 7 | 45 | 10 | 6 | 7 |
| `v9_b2_median_v6` | 2 | 1 | 2 | 13 | 47 | 10 | 7 | 7 |
| `v6_selector` | 2 | 1 | 11 | 6 | 47 | 10 | 7 | 7 |
| `v8_entropy_modulated` | 2 | 1 | 9 | 6 | 50 | 10 | 9 | 5 |
| `v7_stacking_bracketed` | 1 | 1 | 9 | 6 | 49 | 12 | 9 | 8 |
| `v7_stacking_density` | 1 | 1 | 9 | 6 | 49 | 12 | 9 | 8 |
| `v1_corrected` | 4 | 1 | 6 | 10 | 44 | 16 | 7 | 9 |
| `v8_b2_b4_boosted` | 1 | 1 | 2 | 15 | 49 | 12 | 6 | 14 |
| `v5_adaptive_corrected` | 2 | 1 | 11 | 6 | 66 | 9 | 12 | 6 |

---

## Dimensi 2: Kecepatan (ms/pohon)

Diukur dengan 30 repetisi per metode, 727 pohon per repetisi.

| Rank | Method | Mean ms | Median ms | Std ms | pohon/detik |
|---:|---|---:|---:|---:|---:|
| 1 | `v1_corrected` | 0.0035 | 0.0034 | 0.0004 | 284792 |
| 2 | `v5_adaptive_corrected` | 0.0075 | 0.0073 | 0.0006 | 133213 |
| 3 | `v7_stacking_density` | 0.0154 | 0.0155 | 0.0009 | 64853 |
| 4 | `v2_visibility` | 0.0242 | 0.0242 | 0.0004 | 41356 |
| 5 | `v5_best_visibility` | 0.0243 | 0.0241 | 0.0017 | 41165 |
| 6 | `v8_b2_b4_boosted` | 0.0481 | 0.0479 | 0.0006 | 20808 |
| 7 | `v7_stacking_bracketed` | 0.0483 | 0.0481 | 0.0015 | 20723 |
| 8 | `v9_selector` | 0.0792 | 0.0788 | 0.0013 | 12623 |
| 9 | `v6_selector` | 0.0996 | 0.0992 | 0.0016 | 10041 |
| 10 | `v8_entropy_modulated` | 0.1039 | 0.1036 | 0.0013 | 9626 |
| 11 | `v9_b2_median_v6` | 0.4200 | 0.4196 | 0.0036 | 2381 |

---

## Dimensi 3: Robustness terhadap Noise Koordinat

Simulasi: tambah Gaussian noise σ=N% ke x_norm dan y_norm setiap bbox.  
Mengukur seberapa cepat akurasi turun ketika koordinat detector tidak sempurna.

| Method | σ=0% | σ=5% | σ=10% | σ=20% | Drop@20% |
|---|---:|---:|---:|---:|---:|
| `v2_visibility` | 89.41% | 89.13% | 87.62% | 85.97% | 3.44% |
| `v5_best_visibility` | 89.41% | 89.13% | 87.62% | 85.97% | 3.44% |
| `v9_selector` | 89.27% | 88.03% | 87.48% | 86.80% | 2.47% |
| `v9_b2_median_v6` | 89.00% | 87.90% | 87.35% | 86.66% | 2.34% |
| `v6_selector` | 88.86% | 87.76% | 87.21% | 86.52% | 2.34% |
| `v8_entropy_modulated` | 88.86% | 86.80% | 86.11% | 85.97% | 2.89% |
| `v7_stacking_bracketed` | 88.45% | 86.80% | 86.24% | 86.11% | 2.34% |
| `v7_stacking_density` | 88.45% | 86.80% | 86.24% | 86.11% | 2.34% |
| `v1_corrected` | 87.90% | 87.90% | 87.90% | 87.90% | 0.00% |
| `v8_b2_b4_boosted` | 87.62% | 86.24% | 85.69% | 85.69% | 1.93% |
| `v5_adaptive_corrected` | 86.11% | 86.11% | 86.11% | 86.11% | 0.00% |

> Drop@20% = selisih Acc antara noise=0% dan noise=20% (lebih kecil = lebih robust)

---

## Dimensi 4: Domain Breakdown (DAMIMAS vs LONSUM)

### Domain: DAMIMAS (n=727)

| Rank | Method | Acc ±1 | MAE | Gagal |
|---:|---|---:|---:|---:|
| 1 | `v2_visibility` | 89.41% | 0.3116 | 77 |
| 2 | `v5_best_visibility` | 89.41% | 0.3116 | 77 |
| 3 | `v9_selector` | 89.27% | 0.3267 | 78 |
| 4 | `v9_b2_median_v6` | 89.00% | 0.3219 | 80 |
| 5 | `v6_selector` | 88.86% | 0.3287 | 81 |
| 6 | `v8_entropy_modulated` | 88.86% | 0.3394 | 81 |
| 7 | `v7_stacking_bracketed` | 88.45% | 0.3181 | 84 |
| 8 | `v7_stacking_density` | 88.45% | 0.3232 | 84 |
| 9 | `v1_corrected` | 87.90% | 0.3291 | 88 |
| 10 | `v8_b2_b4_boosted` | 87.62% | 0.3067 | 90 |
| 11 | `v5_adaptive_corrected` | 86.11% | 0.3494 | 101 |

### Breakdown Per Split (train / val / test)

| Method | test Acc | train Acc | val Acc |
|---|---:|---:|---:|
| `v2_visibility` | 86.00% | 90.10% | 85.92% |
| `v5_best_visibility` | 86.00% | 90.10% | 85.92% |
| `v9_selector` | 90.00% | 89.60% | 85.92% |
| `v9_b2_median_v6` | 88.00% | 89.44% | 85.92% |
| `v6_selector` | 86.00% | 89.44% | 85.92% |
| `v8_entropy_modulated` | 88.00% | 89.44% | 84.51% |
| `v7_stacking_bracketed` | 88.00% | 88.78% | 85.92% |
| `v7_stacking_density` | 88.00% | 88.78% | 85.92% |
| `v1_corrected` | 86.00% | 87.95% | 88.73% |
| `v8_b2_b4_boosted` | 88.00% | 87.95% | 84.51% |
| `v5_adaptive_corrected` | 84.00% | 86.63% | 83.10% |

---

## Ringkasan: Tradeoff Antar Dimensi

| Method | Acc ±1 | Rank Acc | ms/pohon | Rank Speed | Drop@20% | Rank Robust |
|---|---:|---:|---:|---:|---:|---:|
| `v2_visibility` | 89.41% | #1 | 0.024 | #4 | 3.44% | #11 |
| `v5_best_visibility` | 89.41% | #2 | 0.024 | #5 | 3.44% | #10 |
| `v9_selector` | 89.27% | #3 | 0.079 | #8 | 2.47% | #8 |
| `v9_b2_median_v6` | 89.00% | #4 | 0.420 | #11 | 2.34% | #7 |
| `v6_selector` | 88.86% | #5 | 0.100 | #9 | 2.34% | #6 |
| `v8_entropy_modulated` | 88.86% | #6 | 0.104 | #10 | 2.89% | #9 |
| `v7_stacking_bracketed` | 88.45% | #7 | 0.048 | #7 | 2.34% | #4 |
| `v7_stacking_density` | 88.45% | #8 | 0.015 | #3 | 2.34% | #5 |
| `v1_corrected` | 87.90% | #9 | 0.004 | #1 | 0.00% | #1 |
| `v8_b2_b4_boosted` | 87.62% | #10 | 0.048 | #6 | 1.93% | #3 |
| `v5_adaptive_corrected` | 86.11% | #11 | 0.007 | #2 | 0.00% | #2 |

> **Rekomendasi final:** `v9_selector` untuk akurasi maksimal. Untuk pipeline real-time atau inference massal, pertimbangkan `v6_selector` atau `v5_adaptive_corrected` (lebih cepat, Acc masih >93%).
