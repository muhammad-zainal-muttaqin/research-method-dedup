# Benchmark Multi-Dimensi: 11 Algoritma Dedup

**Dataset:** 882 pohon JSON (228 GT)  
**Tanggal:** 2026-04-24  
**Metrik utama:** Acc ±1 (semua kelas dalam 1 error), MAE, ms/pohon

---

## Dimensi 1: Akurasi (Acc ±1 per kelas)

Pohon dianggap **benar** jika semua 4 kelas masing-masing dalam ±1 dari GT.

| Rank | Method | Gen | Acc ±1 | MAE | MTE | Gagal |
|---:|---|---|---:|---:|---:|---:|
| 1 | `v2_visibility` | v2 | **89.34%** | 0.3061 | 1.2245 | 94 |
| 2 | `v5_best_visibility` | v5 | **89.34%** | 0.3061 | 1.2245 | 94 |
| 3 | `v9_b2_median_v6` | v9 | **88.78%** | 0.3115 | 1.2460 | 99 |
| 4 | `v9_selector` | v9 | **88.78%** | 0.3163 | 1.2653 | 99 |
| 5 | `v8_entropy_modulated` | v8 | **88.78%** | 0.3282 | 1.3129 | 99 |
| 6 | `v6_selector` | v6 | **88.55%** | 0.3172 | 1.2687 | 101 |
| 7 | `v7_stacking_bracketed` | v7 | **88.44%** | 0.3078 | 1.2313 | 102 |
| 8 | `v7_stacking_density` | v7 | **88.44%** | 0.3141 | 1.2562 | 102 |
| 9 | `v8_b2_b4_boosted` | v8 | **88.21%** | 0.2939 | 1.1757 | 104 |
| 10 | `v1_corrected` | v1 | **88.21%** | 0.3192 | 1.2766 | 104 |
| 11 | `v5_adaptive_corrected` | v5 | **86.28%** | 0.3342 | 1.3367 | 121 |

> MTE = Mean Total Error (jumlah absolut error semua kelas, rata-rata per pohon)

### Akurasi Per Kelas (% pohon dalam ±1)

| Method | B1 | B2 | B3 | B4 |
|---|---:|---:|---:|---:|
| `v2_visibility` | 99.7% | 98.2% | 93.5% | 97.3% |
| `v5_best_visibility` | 99.7% | 98.2% | 93.5% | 97.3% |
| `v9_b2_median_v6` | 99.8% | 98.3% | 92.2% | 97.7% |
| `v9_selector` | 99.8% | 98.1% | 92.3% | 97.8% |
| `v8_entropy_modulated` | 99.8% | 98.0% | 91.8% | 98.4% |
| `v6_selector` | 99.8% | 98.1% | 92.2% | 97.7% |
| `v7_stacking_bracketed` | 99.9% | 98.0% | 91.8% | 98.0% |
| `v7_stacking_density` | 99.9% | 98.0% | 91.8% | 98.0% |
| `v8_b2_b4_boosted` | 99.9% | 98.2% | 91.8% | 97.5% |
| `v1_corrected` | 99.7% | 98.2% | 91.8% | 98.1% |
| `v5_adaptive_corrected` | 99.8% | 98.1% | 89.9% | 97.7% |

### Pola Error Per Kelas (over >1 / under <-1, jumlah pohon)

| Method | B1↑ | B1↓ | B2↑ | B2↓ | B3↑ | B3↓ | B4↑ | B4↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `v2_visibility` | 3 | 0 | 3 | 13 | 27 | 30 | 0 | 24 |
| `v5_best_visibility` | 3 | 0 | 3 | 13 | 27 | 30 | 0 | 24 |
| `v9_b2_median_v6` | 2 | 0 | 1 | 14 | 51 | 18 | 8 | 12 |
| `v9_selector` | 2 | 0 | 10 | 7 | 50 | 18 | 7 | 12 |
| `v8_entropy_modulated` | 2 | 0 | 9 | 9 | 55 | 17 | 8 | 6 |
| `v6_selector` | 2 | 0 | 10 | 7 | 51 | 18 | 8 | 12 |
| `v7_stacking_bracketed` | 1 | 0 | 9 | 9 | 53 | 19 | 8 | 10 |
| `v7_stacking_density` | 1 | 0 | 9 | 9 | 53 | 19 | 8 | 10 |
| `v8_b2_b4_boosted` | 1 | 0 | 1 | 15 | 53 | 19 | 4 | 18 |
| `v1_corrected` | 3 | 0 | 6 | 10 | 51 | 21 | 6 | 11 |
| `v5_adaptive_corrected` | 2 | 0 | 10 | 7 | 73 | 16 | 12 | 8 |

---

## Dimensi 2: Kecepatan (ms/pohon)

Diukur dengan 30 repetisi per metode, 882 pohon per repetisi.

| Rank | Method | Mean ms | Median ms | Std ms | pohon/detik |
|---:|---|---:|---:|---:|---:|
| 1 | `v1_corrected` | 0.0035 | 0.0034 | 0.0003 | 284238 |
| 2 | `v5_adaptive_corrected` | 0.0081 | 0.0082 | 0.0008 | 124025 |
| 3 | `v7_stacking_density` | 0.0140 | 0.0138 | 0.0010 | 71400 |
| 4 | `v5_best_visibility` | 0.0229 | 0.0231 | 0.0009 | 43586 |
| 5 | `v2_visibility` | 0.0231 | 0.0233 | 0.0010 | 43253 |
| 6 | `v7_stacking_bracketed` | 0.0474 | 0.0473 | 0.0007 | 21101 |
| 7 | `v8_b2_b4_boosted` | 0.0475 | 0.0473 | 0.0011 | 21034 |
| 8 | `v9_selector` | 0.0787 | 0.0786 | 0.0012 | 12704 |
| 9 | `v6_selector` | 0.0991 | 0.0990 | 0.0011 | 10094 |
| 10 | `v8_entropy_modulated` | 0.1014 | 0.1010 | 0.0012 | 9864 |
| 11 | `v9_b2_median_v6` | 0.4150 | 0.4149 | 0.0033 | 2410 |

---

## Dimensi 3: Robustness terhadap Noise Koordinat

Simulasi: tambah Gaussian noise σ=N% ke x_norm dan y_norm setiap bbox.  
Mengukur seberapa cepat akurasi turun ketika koordinat detector tidak sempurna.

| Method | σ=0% | σ=5% | σ=10% | σ=20% | Drop@20% |
|---|---:|---:|---:|---:|---:|
| `v2_visibility` | 89.34% | 88.89% | 88.66% | 86.85% | 2.49% |
| `v5_best_visibility` | 89.34% | 88.89% | 88.66% | 86.85% | 2.49% |
| `v9_b2_median_v6` | 88.78% | 88.44% | 87.76% | 87.41% | 1.37% |
| `v9_selector` | 88.78% | 88.32% | 87.64% | 87.30% | 1.48% |
| `v8_entropy_modulated` | 88.78% | 86.62% | 86.39% | 86.28% | 2.50% |
| `v6_selector` | 88.55% | 88.32% | 87.64% | 87.30% | 1.25% |
| `v7_stacking_bracketed` | 88.44% | 87.07% | 86.39% | 86.28% | 2.16% |
| `v7_stacking_density` | 88.44% | 87.07% | 86.39% | 86.28% | 2.16% |
| `v8_b2_b4_boosted` | 88.21% | 86.62% | 85.94% | 86.05% | 2.16% |
| `v1_corrected` | 88.21% | 88.21% | 88.21% | 88.21% | 0.00% |
| `v5_adaptive_corrected` | 86.28% | 86.28% | 86.28% | 86.28% | 0.00% |

> Drop@20% = selisih Acc antara noise=0% dan noise=20% (lebih kecil = lebih robust)

---

## Dimensi 4: Domain Breakdown (DAMIMAS vs LONSUM)

### Domain: DAMIMAS (n=802)

| Rank | Method | Acc ±1 | MAE | Gagal |
|---:|---|---:|---:|---:|
| 1 | `v2_visibility` | 89.15% | 0.3108 | 87 |
| 2 | `v5_best_visibility` | 89.15% | 0.3108 | 87 |
| 3 | `v9_b2_median_v6` | 88.53% | 0.3173 | 92 |
| 4 | `v9_selector` | 88.53% | 0.3223 | 92 |
| 5 | `v6_selector` | 88.28% | 0.3233 | 94 |
| 6 | `v8_entropy_modulated` | 88.28% | 0.3354 | 94 |
| 7 | `v1_corrected` | 88.15% | 0.3242 | 95 |
| 8 | `v7_stacking_bracketed` | 88.03% | 0.3136 | 96 |
| 9 | `v7_stacking_density` | 88.03% | 0.3195 | 96 |
| 10 | `v8_b2_b4_boosted` | 87.78% | 0.3011 | 98 |
| 11 | `v5_adaptive_corrected` | 85.79% | 0.3416 | 114 |

### Domain: LONSUM (n=80)

| Rank | Method | Acc ±1 | MAE | Gagal |
|---:|---|---:|---:|---:|
| 1 | `v8_entropy_modulated` | 93.75% | 0.2562 | 5 |
| 2 | `v7_stacking_bracketed` | 92.50% | 0.2500 | 6 |
| 3 | `v7_stacking_density` | 92.50% | 0.2594 | 6 |
| 4 | `v8_b2_b4_boosted` | 92.50% | 0.2219 | 6 |
| 5 | `v2_visibility` | 91.25% | 0.2594 | 7 |
| 6 | `v5_adaptive_corrected` | 91.25% | 0.2594 | 7 |
| 7 | `v5_best_visibility` | 91.25% | 0.2594 | 7 |
| 8 | `v6_selector` | 91.25% | 0.2562 | 7 |
| 9 | `v9_b2_median_v6` | 91.25% | 0.2531 | 7 |
| 10 | `v9_selector` | 91.25% | 0.2562 | 7 |
| 11 | `v1_corrected` | 88.75% | 0.2688 | 9 |

### Breakdown Per Split (train / val / test)

| Method | test Acc | train Acc | val Acc |
|---|---:|---:|---:|
| `v2_visibility` | 88.10% | 89.27% | 90.67% |
| `v5_best_visibility` | 88.10% | 89.27% | 90.67% |
| `v9_b2_median_v6` | 87.30% | 89.27% | 88.00% |
| `v9_selector` | 84.92% | 89.77% | 88.00% |
| `v8_entropy_modulated` | 86.51% | 89.11% | 89.33% |
| `v6_selector` | 84.92% | 89.44% | 88.00% |
| `v7_stacking_bracketed` | 86.51% | 88.61% | 89.33% |
| `v7_stacking_density` | 86.51% | 88.61% | 89.33% |
| `v8_b2_b4_boosted` | 88.89% | 87.79% | 89.33% |
| `v1_corrected` | 87.30% | 87.95% | 90.00% |
| `v5_adaptive_corrected` | 84.13% | 86.80% | 86.00% |

---

## Ringkasan: Tradeoff Antar Dimensi

| Method | Acc ±1 | Rank Acc | ms/pohon | Rank Speed | Drop@20% | Rank Robust |
|---|---:|---:|---:|---:|---:|---:|
| `v2_visibility` | 89.34% | #1 | 0.023 | #5 | 2.49% | #9 |
| `v5_best_visibility` | 89.34% | #2 | 0.023 | #4 | 2.49% | #10 |
| `v9_b2_median_v6` | 88.78% | #3 | 0.415 | #11 | 1.37% | #4 |
| `v9_selector` | 88.78% | #4 | 0.079 | #8 | 1.48% | #5 |
| `v8_entropy_modulated` | 88.78% | #5 | 0.101 | #10 | 2.50% | #11 |
| `v6_selector` | 88.55% | #6 | 0.099 | #9 | 1.25% | #3 |
| `v7_stacking_bracketed` | 88.44% | #7 | 0.047 | #6 | 2.16% | #8 |
| `v7_stacking_density` | 88.44% | #8 | 0.014 | #3 | 2.16% | #7 |
| `v8_b2_b4_boosted` | 88.21% | #9 | 0.048 | #7 | 2.16% | #6 |
| `v1_corrected` | 88.21% | #10 | 0.004 | #1 | 0.00% | #1 |
| `v5_adaptive_corrected` | 86.28% | #11 | 0.008 | #2 | 0.00% | #2 |

> **Rekomendasi final:** `v9_selector` untuk akurasi maksimal. Untuk pipeline real-time atau inference massal, pertimbangkan `v6_selector` atau `v5_adaptive_corrected` (lebih cepat, Acc masih >93%).
