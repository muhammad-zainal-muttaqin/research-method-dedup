# Benchmark Multi-Dimensi: 11 Algoritma Dedup

**Dataset:** 228 pohon JSON (228 GT)  
**Tanggal:** 2026-04-24  
**Metrik utama:** Acc ±1 (semua kelas dalam 1 error), MAE, ms/pohon

---

## Dimensi 1: Akurasi (Acc ±1 per kelas)

Pohon dianggap **benar** jika semua 4 kelas masing-masing dalam ±1 dari GT.

| Rank | Method | Gen | Acc ±1 | MAE | MTE | Gagal |
|---:|---|---|---:|---:|---:|---:|
| 1 | `v9_selector` | v9 | **97.37%** | 0.2533 | 1.0132 | 6 |
| 2 | `v9_b2_median_v6` | v9 | **96.05%** | 0.2577 | 1.0307 | 9 |
| 3 | `v6_selector` | v6 | **96.05%** | 0.2599 | 1.0395 | 9 |
| 4 | `v7_stacking_bracketed` | v7 | **94.30%** | 0.2643 | 1.0570 | 13 |
| 5 | `v7_stacking_density` | v7 | **94.30%** | 0.2708 | 1.0833 | 13 |
| 6 | `v8_entropy_modulated` | v8 | **94.30%** | 0.2763 | 1.1053 | 13 |
| 7 | `v5_adaptive_corrected` | v5 | **93.86%** | 0.2774 | 1.1096 | 14 |
| 8 | `v8_b2_b4_boosted` | v8 | **92.54%** | 0.2632 | 1.0526 | 17 |
| 9 | `v2_visibility` | v2 | **92.54%** | 0.2664 | 1.0658 | 17 |
| 10 | `v5_best_visibility` | v5 | **92.54%** | 0.2664 | 1.0658 | 17 |
| 11 | `v1_corrected` | v1 | **90.79%** | 0.2851 | 1.1404 | 21 |

> MTE = Mean Total Error (jumlah absolut error semua kelas, rata-rata per pohon)

### Akurasi Per Kelas (% pohon dalam ±1)

| Method | B1 | B2 | B3 | B4 |
|---|---:|---:|---:|---:|
| `v9_selector` | 100.0% | 98.7% | 99.1% | 99.6% |
| `v9_b2_median_v6` | 100.0% | 98.2% | 98.7% | 99.1% |
| `v6_selector` | 100.0% | 98.2% | 98.7% | 99.1% |
| `v7_stacking_bracketed` | 100.0% | 98.7% | 96.5% | 98.7% |
| `v7_stacking_density` | 100.0% | 98.7% | 96.5% | 98.7% |
| `v8_entropy_modulated` | 100.0% | 98.7% | 96.5% | 98.7% |
| `v5_adaptive_corrected` | 100.0% | 98.2% | 96.9% | 98.2% |
| `v8_b2_b4_boosted` | 100.0% | 97.8% | 96.5% | 97.8% |
| `v2_visibility` | 100.0% | 97.8% | 96.0% | 98.2% |
| `v5_best_visibility` | 100.0% | 97.8% | 96.0% | 98.2% |
| `v1_corrected` | 100.0% | 97.8% | 94.3% | 97.8% |

### Pola Error Per Kelas (over >1 / under <-1, jumlah pohon)

| Method | B1↑ | B1↓ | B2↑ | B2↓ | B3↑ | B3↓ | B4↑ | B4↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `v9_selector` | 0 | 0 | 1 | 2 | 1 | 1 | 0 | 1 |
| `v9_b2_median_v6` | 0 | 0 | 0 | 4 | 2 | 1 | 1 | 1 |
| `v6_selector` | 0 | 0 | 2 | 2 | 2 | 1 | 1 | 1 |
| `v7_stacking_bracketed` | 0 | 0 | 1 | 2 | 6 | 2 | 2 | 1 |
| `v7_stacking_density` | 0 | 0 | 1 | 2 | 6 | 2 | 2 | 1 |
| `v8_entropy_modulated` | 0 | 0 | 1 | 2 | 6 | 2 | 2 | 1 |
| `v5_adaptive_corrected` | 0 | 0 | 2 | 2 | 6 | 1 | 3 | 1 |
| `v8_b2_b4_boosted` | 0 | 0 | 0 | 5 | 6 | 2 | 2 | 3 |
| `v2_visibility` | 0 | 0 | 0 | 5 | 2 | 7 | 0 | 4 |
| `v5_best_visibility` | 0 | 0 | 0 | 5 | 2 | 7 | 0 | 4 |
| `v1_corrected` | 0 | 0 | 0 | 5 | 7 | 6 | 4 | 1 |

---

## Dimensi 2: Kecepatan (ms/pohon)

Diukur dengan 30 repetisi per metode, 228 pohon per repetisi.

| Rank | Method | Mean ms | Median ms | Std ms | pohon/detik |
|---:|---|---:|---:|---:|---:|
| 1 | `v1_corrected` | 0.0036 | 0.0034 | 0.0005 | 279830 |
| 2 | `v5_adaptive_corrected` | 0.0073 | 0.0073 | 0.0003 | 136242 |
| 3 | `v7_stacking_density` | 0.0142 | 0.0138 | 0.0013 | 70585 |
| 4 | `v2_visibility` | 0.0222 | 0.0218 | 0.0009 | 45146 |
| 5 | `v5_best_visibility` | 0.0228 | 0.0227 | 0.0015 | 43833 |
| 6 | `v8_b2_b4_boosted` | 0.0477 | 0.0480 | 0.0021 | 20951 |
| 7 | `v7_stacking_bracketed` | 0.0480 | 0.0481 | 0.0018 | 20830 |
| 8 | `v9_selector` | 0.0792 | 0.0794 | 0.0023 | 12619 |
| 9 | `v6_selector` | 0.0993 | 0.0986 | 0.0035 | 10074 |
| 10 | `v8_entropy_modulated` | 0.1046 | 0.1044 | 0.0023 | 9558 |
| 11 | `v9_b2_median_v6` | 0.4291 | 0.4250 | 0.0140 | 2330 |

---

## Dimensi 3: Robustness terhadap Noise Koordinat

Simulasi: tambah Gaussian noise σ=N% ke x_norm dan y_norm setiap bbox.  
Mengukur seberapa cepat akurasi turun ketika koordinat detector tidak sempurna.

| Method | σ=0% | σ=5% | σ=10% | σ=20% | Drop@20% |
|---|---:|---:|---:|---:|---:|
| `v9_selector` | 97.37% | 95.18% | 95.18% | 94.74% | 2.63% |
| `v9_b2_median_v6` | 96.05% | 94.30% | 94.30% | 93.86% | 2.19% |
| `v6_selector` | 96.05% | 94.30% | 94.30% | 93.86% | 2.19% |
| `v7_stacking_bracketed` | 94.30% | 93.86% | 93.86% | 93.86% | 0.44% |
| `v7_stacking_density` | 94.30% | 93.86% | 93.86% | 93.86% | 0.44% |
| `v8_entropy_modulated` | 94.30% | 93.86% | 92.98% | 92.98% | 1.32% |
| `v5_adaptive_corrected` | 93.86% | 93.86% | 93.86% | 93.86% | 0.00% |
| `v8_b2_b4_boosted` | 92.54% | 93.42% | 93.42% | 93.42% | -0.88% |
| `v2_visibility` | 92.54% | 92.11% | 91.67% | 91.67% | 0.87% |
| `v5_best_visibility` | 92.54% | 92.11% | 91.67% | 91.67% | 0.87% |
| `v1_corrected` | 90.79% | 90.79% | 90.79% | 90.79% | 0.00% |

> Drop@20% = selisih Acc antara noise=0% dan noise=20% (lebih kecil = lebih robust)

---

## Dimensi 4: Domain Breakdown (DAMIMAS vs LONSUM)

### Domain: DAMIMAS (n=228)

| Rank | Method | Acc ±1 | MAE | Gagal |
|---:|---|---:|---:|---:|
| 1 | `v9_selector` | 97.37% | 0.2533 | 6 |
| 2 | `v6_selector` | 96.05% | 0.2599 | 9 |
| 3 | `v9_b2_median_v6` | 96.05% | 0.2577 | 9 |
| 4 | `v7_stacking_bracketed` | 94.30% | 0.2643 | 13 |
| 5 | `v7_stacking_density` | 94.30% | 0.2708 | 13 |
| 6 | `v8_entropy_modulated` | 94.30% | 0.2763 | 13 |
| 7 | `v5_adaptive_corrected` | 93.86% | 0.2774 | 14 |
| 8 | `v2_visibility` | 92.54% | 0.2664 | 17 |
| 9 | `v5_best_visibility` | 92.54% | 0.2664 | 17 |
| 10 | `v8_b2_b4_boosted` | 92.54% | 0.2632 | 17 |
| 11 | `v1_corrected` | 90.79% | 0.2851 | 21 |

### Breakdown Per Split (train / val / test)

| Method | test Acc | train Acc | val Acc |
|---|---:|---:|---:|
| `v9_selector` | 90.32% | 98.47% | 100.00% |
| `v9_b2_median_v6` | 87.10% | 97.45% | 100.00% |
| `v6_selector` | 83.87% | 97.96% | 100.00% |
| `v7_stacking_bracketed` | 83.87% | 95.92% | 100.00% |
| `v7_stacking_density` | 83.87% | 95.92% | 100.00% |
| `v8_entropy_modulated` | 83.87% | 95.92% | 100.00% |
| `v5_adaptive_corrected` | 80.65% | 95.92% | 100.00% |
| `v8_b2_b4_boosted` | 83.87% | 93.88% | 100.00% |
| `v2_visibility` | 80.65% | 94.39% | 100.00% |
| `v5_best_visibility` | 80.65% | 94.39% | 100.00% |
| `v1_corrected` | 80.65% | 92.35% | 100.00% |

---

## Ringkasan: Tradeoff Antar Dimensi

| Method | Acc ±1 | Rank Acc | ms/pohon | Rank Speed | Drop@20% | Rank Robust |
|---|---:|---:|---:|---:|---:|---:|
| `v9_selector` | 97.37% | #1 | 0.079 | #8 | 2.63% | #11 |
| `v9_b2_median_v6` | 96.05% | #2 | 0.429 | #11 | 2.19% | #10 |
| `v6_selector` | 96.05% | #3 | 0.099 | #9 | 2.19% | #9 |
| `v7_stacking_bracketed` | 94.30% | #4 | 0.048 | #7 | 0.44% | #5 |
| `v7_stacking_density` | 94.30% | #5 | 0.014 | #3 | 0.44% | #4 |
| `v8_entropy_modulated` | 94.30% | #6 | 0.105 | #10 | 1.32% | #8 |
| `v5_adaptive_corrected` | 93.86% | #7 | 0.007 | #2 | 0.00% | #3 |
| `v8_b2_b4_boosted` | 92.54% | #8 | 0.048 | #6 | -0.88% | #1 |
| `v2_visibility` | 92.54% | #9 | 0.022 | #4 | 0.87% | #7 |
| `v5_best_visibility` | 92.54% | #10 | 0.023 | #5 | 0.87% | #6 |
| `v1_corrected` | 90.79% | #11 | 0.004 | #1 | 0.00% | #2 |

> **Rekomendasi final:** `v9_selector` untuk akurasi maksimal. Untuk pipeline real-time atau inference massal, pertimbangkan `v6_selector` atau `v5_adaptive_corrected` (lebih cepat, Acc masih >93%).
