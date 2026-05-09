# Benchmark Multi-Dimensi: 11 Algoritma Dedup

**Dataset:** 953 pohon JSON (228 GT)  
**Tanggal:** 2026-04-24  
**Metrik utama:** Acc ±1 (semua kelas dalam 1 error), MAE, ms/pohon

---

## Dimensi 1: Akurasi (Acc ±1 per kelas)

Pohon dianggap **benar** jika semua 4 kelas masing-masing dalam ±1 dari GT.

| Rank | Method | Gen | Acc ±1 | MAE | MTE | Gagal |
|---:|---|---|---:|---:|---:|---:|
| 1 | `v2_visibility` | v2 | **85.52%** | 0.3993 | 1.5971 | 138 |
| 2 | `v5_best_visibility` | v5 | **85.52%** | 0.3993 | 1.5971 | 138 |
| 3 | `v9_b2_median_v6` | v9 | **84.89%** | 0.4268 | 1.7072 | 144 |
| 4 | `v9_selector` | v9 | **84.89%** | 0.4357 | 1.7429 | 144 |
| 5 | `v8_entropy_modulated` | v8 | **84.78%** | 0.4507 | 1.8027 | 145 |
| 6 | `v6_selector` | v6 | **84.68%** | 0.4365 | 1.7461 | 146 |
| 7 | `v7_stacking_bracketed` | v7 | **84.58%** | 0.4284 | 1.7135 | 147 |
| 8 | `v7_stacking_density` | v7 | **84.58%** | 0.4347 | 1.7387 | 147 |
| 9 | `v8_b2_b4_boosted` | v8 | **84.37%** | 0.4111 | 1.6443 | 149 |
| 10 | `v1_corrected` | v1 | **84.37%** | 0.4158 | 1.6632 | 149 |
| 11 | `v5_adaptive_corrected` | v5 | **82.58%** | 0.4599 | 1.8395 | 166 |

> MTE = Mean Total Error (jumlah absolut error semua kelas, rata-rata per pohon)

### Akurasi Per Kelas (% pohon dalam ±1)

| Method | B1 | B2 | B3 | B4 |
|---|---:|---:|---:|---:|
| `v2_visibility` | 97.3% | 95.3% | 90.1% | 96.8% |
| `v5_best_visibility` | 97.3% | 95.3% | 90.1% | 96.8% |
| `v9_b2_median_v6` | 97.4% | 95.1% | 88.6% | 97.2% |
| `v9_selector` | 97.4% | 94.8% | 88.7% | 97.3% |
| `v8_entropy_modulated` | 97.2% | 94.7% | 88.2% | 97.6% |
| `v6_selector` | 97.4% | 94.8% | 88.6% | 97.2% |
| `v7_stacking_bracketed` | 97.5% | 94.7% | 88.2% | 97.4% |
| `v7_stacking_density` | 97.5% | 94.7% | 88.2% | 97.4% |
| `v8_b2_b4_boosted` | 97.5% | 95.0% | 88.2% | 96.8% |
| `v1_corrected` | 97.3% | 95.3% | 88.5% | 97.5% |
| `v5_adaptive_corrected` | 97.4% | 94.8% | 86.5% | 97.2% |

### Pola Error Per Kelas (over >1 / under <-1, jumlah pohon)

| Method | B1↑ | B1↓ | B2↑ | B2↓ | B3↑ | B3↓ | B4↑ | B4↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `v2_visibility` | 25 | 1 | 30 | 15 | 61 | 33 | 5 | 26 |
| `v5_best_visibility` | 25 | 1 | 30 | 15 | 61 | 33 | 5 | 26 |
| `v9_b2_median_v6` | 24 | 1 | 31 | 16 | 88 | 21 | 14 | 13 |
| `v9_selector` | 24 | 1 | 41 | 9 | 87 | 21 | 13 | 13 |
| `v8_entropy_modulated` | 26 | 1 | 40 | 11 | 92 | 20 | 16 | 7 |
| `v6_selector` | 24 | 1 | 41 | 9 | 88 | 21 | 14 | 13 |
| `v7_stacking_bracketed` | 23 | 1 | 40 | 11 | 90 | 22 | 14 | 11 |
| `v7_stacking_density` | 23 | 1 | 40 | 11 | 90 | 22 | 14 | 11 |
| `v8_b2_b4_boosted` | 23 | 1 | 31 | 17 | 90 | 22 | 10 | 20 |
| `v1_corrected` | 25 | 1 | 33 | 12 | 86 | 24 | 11 | 13 |
| `v5_adaptive_corrected` | 24 | 1 | 41 | 9 | 110 | 19 | 18 | 9 |

---

## Dimensi 2: Kecepatan (ms/pohon)

Diukur dengan 30 repetisi per metode, 953 pohon per repetisi.

| Rank | Method | Mean ms | Median ms | Std ms | pohon/detik |
|---:|---|---:|---:|---:|---:|
| 1 | `v1_corrected` | 0.0049 | 0.0045 | 0.0021 | 202066 |
| 2 | `v5_adaptive_corrected` | 0.0083 | 0.0082 | 0.0006 | 121194 |
| 3 | `v7_stacking_density` | 0.0160 | 0.0160 | 0.0004 | 62530 |
| 4 | `v5_best_visibility` | 0.0248 | 0.0243 | 0.0014 | 40244 |
| 5 | `v2_visibility` | 0.0265 | 0.0259 | 0.0023 | 37720 |
| 6 | `v7_stacking_bracketed` | 0.0512 | 0.0491 | 0.0074 | 19516 |
| 7 | `v8_b2_b4_boosted` | 0.0515 | 0.0488 | 0.0077 | 19413 |
| 8 | `v9_selector` | 0.0896 | 0.0863 | 0.0077 | 11162 |
| 9 | `v8_entropy_modulated` | 0.1048 | 0.1028 | 0.0053 | 9545 |
| 10 | `v6_selector` | 0.1050 | 0.1008 | 0.0087 | 9523 |
| 11 | `v9_b2_median_v6` | 0.4451 | 0.4432 | 0.0216 | 2246 |

---

## Dimensi 3: Robustness terhadap Noise Koordinat

Simulasi: tambah Gaussian noise σ=N% ke x_norm dan y_norm setiap bbox.  
Mengukur seberapa cepat akurasi turun ketika koordinat detector tidak sempurna.

| Method | σ=0% | σ=5% | σ=10% | σ=20% | Drop@20% |
|---|---:|---:|---:|---:|---:|
| `v2_visibility` | 85.52% | 85.10% | 84.78% | 83.00% | 2.52% |
| `v5_best_visibility` | 85.52% | 85.10% | 84.78% | 83.00% | 2.52% |
| `v9_b2_median_v6` | 84.89% | 84.47% | 84.05% | 83.32% | 1.57% |
| `v9_selector` | 84.89% | 84.26% | 83.95% | 83.21% | 1.68% |
| `v8_entropy_modulated` | 84.78% | 83.00% | 82.48% | 82.48% | 2.30% |
| `v6_selector` | 84.68% | 84.26% | 83.95% | 83.21% | 1.47% |
| `v7_stacking_bracketed` | 84.58% | 83.00% | 82.58% | 82.58% | 2.00% |
| `v7_stacking_density` | 84.58% | 83.00% | 82.58% | 82.58% | 2.00% |
| `v8_b2_b4_boosted` | 84.37% | 82.79% | 82.37% | 82.37% | 2.00% |
| `v1_corrected` | 84.37% | 84.37% | 84.37% | 84.37% | 0.00% |
| `v5_adaptive_corrected` | 82.58% | 82.58% | 82.58% | 82.58% | 0.00% |

> Drop@20% = selisih Acc antara noise=0% dan noise=20% (lebih kecil = lebih robust)

---

## Dimensi 4: Domain Breakdown (DAMIMAS vs LONSUM)

### Domain: DAMIMAS (n=854)

| Rank | Method | Acc ±1 | MAE | Gagal |
|---:|---|---:|---:|---:|
| 1 | `v2_visibility` | 84.66% | 0.4157 | 131 |
| 2 | `v5_best_visibility` | 84.66% | 0.4157 | 131 |
| 3 | `v9_b2_median_v6` | 83.96% | 0.4467 | 137 |
| 4 | `v9_selector` | 83.96% | 0.4564 | 137 |
| 5 | `v6_selector` | 83.72% | 0.4573 | 139 |
| 6 | `v1_corrected` | 83.61% | 0.4330 | 140 |
| 7 | `v8_entropy_modulated` | 83.61% | 0.4734 | 140 |
| 8 | `v7_stacking_bracketed` | 83.49% | 0.4494 | 141 |
| 9 | `v7_stacking_density` | 83.49% | 0.4549 | 141 |
| 10 | `v8_b2_b4_boosted` | 83.26% | 0.4330 | 143 |
| 11 | `v5_adaptive_corrected` | 81.38% | 0.4830 | 159 |

### Domain: LONSUM (n=99)

| Rank | Method | Acc ±1 | MAE | Gagal |
|---:|---|---:|---:|---:|
| 1 | `v8_entropy_modulated` | 94.95% | 0.2551 | 5 |
| 2 | `v7_stacking_bracketed` | 93.94% | 0.2475 | 6 |
| 3 | `v7_stacking_density` | 93.94% | 0.2601 | 6 |
| 4 | `v8_b2_b4_boosted` | 93.94% | 0.2222 | 6 |
| 5 | `v2_visibility` | 92.93% | 0.2576 | 7 |
| 6 | `v5_adaptive_corrected` | 92.93% | 0.2601 | 7 |
| 7 | `v5_best_visibility` | 92.93% | 0.2576 | 7 |
| 8 | `v6_selector` | 92.93% | 0.2576 | 7 |
| 9 | `v9_b2_median_v6` | 92.93% | 0.2551 | 7 |
| 10 | `v9_selector` | 92.93% | 0.2576 | 7 |
| 11 | `v1_corrected` | 90.91% | 0.2677 | 9 |

### Breakdown Per Split (train / val / test)

| Method | test Acc | train Acc | val Acc |
|---|---:|---:|---:|
| `v2_visibility` | 86.23% | 86.02% | 83.15% |
| `v5_best_visibility` | 86.23% | 86.02% | 83.15% |
| `v9_b2_median_v6` | 85.03% | 86.18% | 80.34% |
| `v9_selector` | 85.63% | 85.86% | 80.90% |
| `v8_entropy_modulated` | 86.83% | 85.53% | 80.34% |
| `v6_selector` | 85.63% | 85.53% | 80.90% |
| `v7_stacking_bracketed` | 86.83% | 85.03% | 80.90% |
| `v7_stacking_density` | 86.83% | 85.03% | 80.90% |
| `v8_b2_b4_boosted` | 85.03% | 85.03% | 81.46% |
| `v1_corrected` | 86.83% | 84.54% | 81.46% |
| `v5_adaptive_corrected` | 85.03% | 82.89% | 79.21% |

---

## Ringkasan: Tradeoff Antar Dimensi

| Method | Acc ±1 | Rank Acc | ms/pohon | Rank Speed | Drop@20% | Rank Robust |
|---|---:|---:|---:|---:|---:|---:|
| `v2_visibility` | 85.52% | #1 | 0.026 | #5 | 2.52% | #11 |
| `v5_best_visibility` | 85.52% | #2 | 0.025 | #4 | 2.52% | #10 |
| `v9_b2_median_v6` | 84.89% | #3 | 0.445 | #11 | 1.57% | #4 |
| `v9_selector` | 84.89% | #4 | 0.090 | #8 | 1.68% | #5 |
| `v8_entropy_modulated` | 84.78% | #5 | 0.105 | #9 | 2.30% | #9 |
| `v6_selector` | 84.68% | #6 | 0.105 | #10 | 1.47% | #3 |
| `v7_stacking_bracketed` | 84.58% | #7 | 0.051 | #6 | 2.00% | #8 |
| `v7_stacking_density` | 84.58% | #8 | 0.016 | #3 | 2.00% | #7 |
| `v8_b2_b4_boosted` | 84.37% | #9 | 0.051 | #7 | 2.00% | #6 |
| `v1_corrected` | 84.37% | #10 | 0.005 | #1 | 0.00% | #1 |
| `v5_adaptive_corrected` | 82.58% | #11 | 0.008 | #2 | 0.00% | #2 |

> **Rekomendasi final:** `v9_selector` untuk akurasi maksimal. Untuk pipeline real-time atau inference massal, pertimbangkan `v6_selector` atau `v5_adaptive_corrected` (lebih cepat, Acc masih >93%).
