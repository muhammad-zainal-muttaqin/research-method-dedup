# Benchmark Multi-Dimensi: 11 Algoritma Dedup

**Dataset:** 478 pohon JSON  
**Tanggal:** 2026-04-24  
**Metrik utama:** Acc ±1 (semua kelas dalam 1 error), MAE, ms/pohon

---

## Dimensi 1: Akurasi (Acc ±1 per kelas)

Pohon dianggap **benar** jika semua 4 kelas masing-masing dalam ±1 dari GT.

| Rank | Method | Gen | Acc ±1 | MAE | MTE | Gagal |
|---:|---|---|---:|---:|---:|---:|
| 1 | `v9_b2_median_v6` | v9 | **92.68%** | 0.2840 | 1.1360 | 35 |
| 2 | `v9_selector` | v9 | **92.68%** | 0.2892 | 1.1569 | 35 |
| 3 | `v7_stacking_bracketed` | v7 | **91.84%** | 0.2887 | 1.1548 | 39 |
| 4 | `v6_selector` | v6 | **91.84%** | 0.2929 | 1.1715 | 39 |
| 5 | `v7_stacking_density` | v7 | **91.84%** | 0.2939 | 1.1757 | 39 |
| 6 | `v8_entropy_modulated` | v8 | **91.63%** | 0.3060 | 1.2238 | 40 |
| 7 | `v8_b2_b4_boosted` | v8 | **91.00%** | 0.2762 | 1.1046 | 43 |
| 8 | `v2_visibility` | v2 | **90.38%** | 0.2856 | 1.1423 | 46 |
| 9 | `v5_best_visibility` | v5 | **90.38%** | 0.2856 | 1.1423 | 46 |
| 10 | `v5_adaptive_corrected` | v5 | **89.96%** | 0.3081 | 1.2322 | 48 |
| 11 | `v1_corrected` | v1 | **89.12%** | 0.3039 | 1.2155 | 52 |

> MTE = Mean Total Error (jumlah absolut error semua kelas, rata-rata per pohon)

### Akurasi Per Kelas (% pohon dalam ±1)

| Method | B1 | B2 | B3 | B4 |
|---|---:|---:|---:|---:|
| `v9_b2_median_v6` | 99.6% | 97.9% | 96.0% | 98.7% |
| `v9_selector` | 99.6% | 97.3% | 96.4% | 99.0% |
| `v7_stacking_bracketed` | 99.8% | 97.3% | 95.6% | 98.5% |
| `v6_selector` | 99.6% | 97.1% | 96.0% | 98.7% |
| `v7_stacking_density` | 99.8% | 97.3% | 95.6% | 98.5% |
| `v8_entropy_modulated` | 99.6% | 97.3% | 95.6% | 98.5% |
| `v8_b2_b4_boosted` | 99.8% | 97.7% | 95.6% | 97.3% |
| `v2_visibility` | 99.2% | 97.5% | 95.6% | 97.5% |
| `v5_best_visibility` | 99.2% | 97.5% | 95.6% | 97.5% |
| `v5_adaptive_corrected` | 99.6% | 97.1% | 94.6% | 98.1% |
| `v1_corrected` | 99.2% | 97.3% | 93.9% | 97.9% |

### Pola Error Per Kelas (over >1 / under <-1, jumlah pohon)

| Method | B1↑ | B1↓ | B2↑ | B2↓ | B3↑ | B3↓ | B4↑ | B4↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `v9_b2_median_v6` | 2 | 0 | 0 | 10 | 14 | 5 | 1 | 5 |
| `v9_selector` | 2 | 0 | 6 | 7 | 12 | 5 | 0 | 5 |
| `v7_stacking_bracketed` | 1 | 0 | 6 | 7 | 15 | 6 | 2 | 5 |
| `v6_selector` | 2 | 0 | 7 | 7 | 14 | 5 | 1 | 5 |
| `v7_stacking_density` | 1 | 0 | 6 | 7 | 15 | 6 | 2 | 5 |
| `v8_entropy_modulated` | 2 | 0 | 6 | 7 | 16 | 5 | 2 | 5 |
| `v8_b2_b4_boosted` | 1 | 0 | 0 | 11 | 15 | 6 | 2 | 11 |
| `v2_visibility` | 4 | 0 | 1 | 11 | 9 | 12 | 0 | 12 |
| `v5_best_visibility` | 4 | 0 | 1 | 11 | 9 | 12 | 0 | 12 |
| `v5_adaptive_corrected` | 2 | 0 | 7 | 7 | 22 | 4 | 4 | 5 |
| `v1_corrected` | 4 | 0 | 3 | 10 | 19 | 10 | 4 | 6 |

---

## Dimensi 2: Kecepatan (ms/pohon)

Diukur dengan 30 repetisi per metode, 478 pohon per repetisi.

| Rank | Method | Mean ms | Median ms | Std ms | pohon/detik |
|---:|---|---:|---:|---:|---:|
| 1 | `v1_corrected` | 0.0036 | 0.0034 | 0.0004 | 280081 |
| 2 | `v5_adaptive_corrected` | 0.0073 | 0.0072 | 0.0002 | 136388 |
| 3 | `v7_stacking_density` | 0.0138 | 0.0136 | 0.0006 | 72286 |
| 4 | `v5_best_visibility` | 0.0230 | 0.0226 | 0.0015 | 43468 |
| 5 | `v2_visibility` | 0.0231 | 0.0229 | 0.0008 | 43343 |
| 6 | `v7_stacking_bracketed` | 0.0489 | 0.0487 | 0.0009 | 20442 |
| 7 | `v8_b2_b4_boosted` | 0.0494 | 0.0491 | 0.0013 | 20232 |
| 8 | `v9_selector` | 0.0791 | 0.0788 | 0.0010 | 12637 |
| 9 | `v6_selector` | 0.1006 | 0.1004 | 0.0012 | 9941 |
| 10 | `v8_entropy_modulated` | 0.1045 | 0.1044 | 0.0013 | 9568 |
| 11 | `v9_b2_median_v6` | 0.4329 | 0.4313 | 0.0046 | 2310 |

---

## Dimensi 3: Robustness terhadap Noise Koordinat

Simulasi: tambah Gaussian noise σ=N% ke x_norm dan y_norm setiap bbox.  
Mengukur seberapa cepat akurasi turun ketika koordinat detector tidak sempurna.

| Method | σ=0% | σ=5% | σ=10% | σ=20% | Drop@20% |
|---|---:|---:|---:|---:|---:|
| `v9_b2_median_v6` | 92.68% | 91.42% | 91.42% | 90.79% | 1.89% |
| `v9_selector` | 92.68% | 91.21% | 91.21% | 90.59% | 2.09% |
| `v7_stacking_bracketed` | 91.84% | 90.38% | 90.17% | 89.96% | 1.88% |
| `v6_selector` | 91.84% | 90.79% | 90.79% | 90.17% | 1.67% |
| `v7_stacking_density` | 91.84% | 90.38% | 90.17% | 89.96% | 1.88% |
| `v8_entropy_modulated` | 91.63% | 90.79% | 90.17% | 89.96% | 1.67% |
| `v8_b2_b4_boosted` | 91.00% | 90.17% | 90.17% | 89.96% | 1.04% |
| `v2_visibility` | 90.38% | 90.17% | 89.54% | 89.33% | 1.05% |
| `v5_best_visibility` | 90.38% | 90.17% | 89.54% | 89.33% | 1.05% |
| `v5_adaptive_corrected` | 89.96% | 89.96% | 89.96% | 89.96% | 0.00% |
| `v1_corrected` | 89.12% | 89.12% | 89.12% | 89.12% | 0.00% |

> Drop@20% = selisih Acc antara noise=0% dan noise=20% (lebih kecil = lebih robust)

---

## Dimensi 4: Domain Breakdown (DAMIMAS vs LONSUM)

### Domain: DAMIMAS (n=478)

| Rank | Method | Acc ±1 | MAE | Gagal |
|---:|---|---:|---:|---:|
| 1 | `v9_b2_median_v6` | 92.68% | 0.2840 | 35 |
| 2 | `v9_selector` | 92.68% | 0.2892 | 35 |
| 3 | `v6_selector` | 91.84% | 0.2929 | 39 |
| 4 | `v7_stacking_bracketed` | 91.84% | 0.2887 | 39 |
| 5 | `v7_stacking_density` | 91.84% | 0.2939 | 39 |
| 6 | `v8_entropy_modulated` | 91.63% | 0.3060 | 40 |
| 7 | `v8_b2_b4_boosted` | 91.00% | 0.2762 | 43 |
| 8 | `v2_visibility` | 90.38% | 0.2856 | 46 |
| 9 | `v5_best_visibility` | 90.38% | 0.2856 | 46 |
| 10 | `v5_adaptive_corrected` | 89.96% | 0.3081 | 48 |
| 11 | `v1_corrected` | 89.12% | 0.3039 | 52 |

### Breakdown Per Split (train / val / test)

| Method | test Acc | train Acc | val Acc |
|---|---:|---:|---:|
| `v9_b2_median_v6` | 89.29% | 93.52% | 92.31% |
| `v9_selector` | 86.90% | 94.08% | 92.31% |
| `v7_stacking_bracketed` | 85.71% | 93.24% | 92.31% |
| `v6_selector` | 84.52% | 93.52% | 92.31% |
| `v7_stacking_density` | 85.71% | 93.24% | 92.31% |
| `v8_entropy_modulated` | 85.71% | 93.24% | 89.74% |
| `v8_b2_b4_boosted` | 89.29% | 91.83% | 87.18% |
| `v2_visibility` | 86.90% | 91.55% | 87.18% |
| `v5_best_visibility` | 86.90% | 91.55% | 87.18% |
| `v5_adaptive_corrected` | 83.33% | 91.55% | 89.74% |
| `v1_corrected` | 85.71% | 89.58% | 92.31% |

---

## Ringkasan: Tradeoff Antar Dimensi

| Method | Acc ±1 | Rank Acc | ms/pohon | Rank Speed | Drop@20% | Rank Robust |
|---|---:|---:|---:|---:|---:|---:|
| `v9_b2_median_v6` | 92.68% | #1 | 0.433 | #11 | 1.89% | #10 |
| `v9_selector` | 92.68% | #2 | 0.079 | #8 | 2.09% | #11 |
| `v7_stacking_bracketed` | 91.84% | #3 | 0.049 | #6 | 1.88% | #8 |
| `v6_selector` | 91.84% | #4 | 0.101 | #9 | 1.67% | #6 |
| `v7_stacking_density` | 91.84% | #5 | 0.014 | #3 | 1.88% | #9 |
| `v8_entropy_modulated` | 91.63% | #6 | 0.104 | #10 | 1.67% | #7 |
| `v8_b2_b4_boosted` | 91.00% | #7 | 0.049 | #7 | 1.04% | #3 |
| `v2_visibility` | 90.38% | #8 | 0.023 | #5 | 1.05% | #4 |
| `v5_best_visibility` | 90.38% | #9 | 0.023 | #4 | 1.05% | #5 |
| `v5_adaptive_corrected` | 89.96% | #10 | 0.007 | #2 | 0.00% | #2 |
| `v1_corrected` | 89.12% | #11 | 0.004 | #1 | 0.00% | #1 |

> **Rekomendasi final:** `v9_selector` untuk akurasi maksimal. Untuk pipeline real-time atau inference massal, pertimbangkan `v6_selector` atau `v5_adaptive_corrected` (lebih cepat, Acc masih >93%).
