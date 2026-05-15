# Benchmark Multi-Dimensi: 11 Algoritma Dedup

**Dataset:** 953 pohon JSON (228 GT)  
**Tanggal:** 2026-04-24  
**Metrik utama:** Acc ±1 (semua kelas dalam 1 error), MAE, ms/pohon

---

## Dimensi 1: Akurasi (Acc ±1 per kelas)

Pohon dianggap **benar** jika semua 4 kelas masing-masing dalam ±1 dari GT.

| Rank | Method | Gen | Acc ±1 | MAE | MTE | Gagal |
|---:|---|---|---:|---:|---:|---:|
| 1 | `M06_weight_visibility` | ? | **86.36%** | 0.3743 | 1.4974 | 130 |
| 2 | `M20_weight_visibility_grid` | ? | **86.36%** | 0.3743 | 1.4974 | 130 |
| 3 | `M11_median_b2` | ? | **86.04%** | 0.4111 | 1.6443 | 133 |
| 4 | `M12_selector_overrides` | ? | **86.04%** | 0.4200 | 1.6800 | 133 |
| 5 | `M15_divide_global` | ? | **85.94%** | 0.3909 | 1.5635 | 134 |
| 6 | `M17_selector_regime` | ? | **85.94%** | 0.4208 | 1.6831 | 134 |
| 7 | `M10_entropy_divide` | ? | **85.83%** | 0.4328 | 1.7314 | 135 |
| 8 | `M13_stack_bracket` | ? | **85.62%** | 0.4103 | 1.6411 | 137 |
| 9 | `M14_stack_density` | ? | **85.62%** | 0.4166 | 1.6663 | 137 |
| 10 | `M16_boost_b2b4` | ? | **85.41%** | 0.3932 | 1.5729 | 139 |
| 11 | `M19_divide_adaptive` | ? | **83.95%** | 0.4441 | 1.7765 | 153 |

> MTE = Mean Total Error (jumlah absolut error semua kelas, rata-rata per pohon)

### Akurasi Per Kelas (% pohon dalam ±1)

| Method | B1 | B2 | B3 | B4 |
|---|---:|---:|---:|---:|
| `M06_weight_visibility` | 97.6% | 95.6% | 90.5% | 97.2% |
| `M20_weight_visibility_grid` | 97.6% | 95.6% | 90.5% | 97.2% |
| `M11_median_b2` | 97.6% | 95.4% | 89.3% | 97.6% |
| `M12_selector_overrides` | 97.6% | 95.2% | 89.4% | 97.6% |
| `M15_divide_global` | 97.6% | 95.6% | 89.4% | 98.1% |
| `M17_selector_regime` | 97.6% | 95.2% | 89.3% | 97.6% |
| `M10_entropy_divide` | 97.3% | 95.0% | 89.0% | 97.8% |
| `M13_stack_bracket` | 97.6% | 95.1% | 89.1% | 97.6% |
| `M14_stack_density` | 97.6% | 95.1% | 89.1% | 97.6% |
| `M16_boost_b2b4` | 97.6% | 95.3% | 89.1% | 97.2% |
| `M19_divide_adaptive` | 97.6% | 95.2% | 87.3% | 97.6% |

### Pola Error Per Kelas (over >1 / under <-1, jumlah pohon)

| Method | B1↑ | B1↓ | B2↑ | B2↓ | B3↑ | B3↓ | B4↑ | B4↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `M06_weight_visibility` | 23 | 0 | 30 | 12 | 60 | 31 | 4 | 23 |
| `M20_weight_visibility_grid` | 23 | 0 | 30 | 12 | 60 | 31 | 4 | 23 |
| `M11_median_b2` | 23 | 0 | 31 | 13 | 84 | 18 | 12 | 11 |
| `M12_selector_overrides` | 23 | 0 | 40 | 6 | 83 | 18 | 12 | 11 |
| `M15_divide_global` | 23 | 0 | 33 | 9 | 80 | 21 | 8 | 10 |
| `M17_selector_regime` | 23 | 0 | 40 | 6 | 84 | 18 | 12 | 11 |
| `M10_entropy_divide` | 26 | 0 | 40 | 8 | 88 | 17 | 16 | 5 |
| `M13_stack_bracket` | 23 | 0 | 39 | 8 | 85 | 19 | 14 | 9 |
| `M14_stack_density` | 23 | 0 | 39 | 8 | 85 | 19 | 14 | 9 |
| `M16_boost_b2b4` | 23 | 0 | 31 | 14 | 85 | 19 | 10 | 17 |
| `M19_divide_adaptive` | 23 | 0 | 40 | 6 | 105 | 16 | 16 | 7 |

---

## Dimensi 2: Kecepatan (ms/pohon)

Diukur dengan 30 repetisi per metode, 953 pohon per repetisi.

| Rank | Method | Mean ms | Median ms | Std ms | pohon/detik |
|---:|---|---:|---:|---:|---:|
| 1 | `M15_divide_global` | 0.0067 | 0.0067 | 0.0009 | 150154 |
| 2 | `M19_divide_adaptive` | 0.0116 | 0.0106 | 0.0024 | 86322 |
| 3 | `M14_stack_density` | 0.0206 | 0.0197 | 0.0028 | 48616 |
| 4 | `M06_weight_visibility` | 0.0327 | 0.0330 | 0.0048 | 30564 |
| 5 | `M20_weight_visibility_grid` | 0.0345 | 0.0322 | 0.0068 | 28970 |
| 6 | `M16_boost_b2b4` | 0.0588 | 0.0585 | 0.0055 | 17011 |
| 7 | `M13_stack_bracket` | 0.0730 | 0.0745 | 0.0124 | 13693 |
| 8 | `M12_selector_overrides` | 0.0992 | 0.0961 | 0.0101 | 10081 |
| 9 | `M10_entropy_divide` | 0.1301 | 0.1226 | 0.0211 | 7684 |
| 10 | `M17_selector_regime` | 0.1796 | 0.1700 | 0.0497 | 5569 |
| 11 | `M11_median_b2` | 0.5531 | 0.5500 | 0.0556 | 1808 |

---

## Dimensi 3: Robustness terhadap Noise Koordinat

Simulasi: tambah Gaussian noise σ=N% ke x_norm dan y_norm setiap bbox.  
Mengukur seberapa cepat akurasi turun ketika koordinat detector tidak sempurna.

| Method | σ=0% | σ=5% | σ=10% | σ=20% | Drop@20% |
|---|---:|---:|---:|---:|---:|
| `M06_weight_visibility` | 86.36% | 86.15% | 85.83% | 83.95% | 2.41% |
| `M20_weight_visibility_grid` | 86.36% | 86.15% | 85.83% | 83.95% | 2.41% |
| `M11_median_b2` | 86.04% | 85.10% | 85.31% | 84.78% | 1.26% |
| `M12_selector_overrides` | 86.04% | 85.31% | 85.31% | 84.78% | 1.26% |
| `M15_divide_global` | 85.94% | 85.94% | 85.94% | 85.94% | 0.00% |
| `M17_selector_regime` | 85.94% | 85.31% | 85.31% | 84.78% | 1.16% |
| `M10_entropy_divide` | 85.83% | 84.78% | 84.05% | 83.95% | 1.88% |
| `M13_stack_bracket` | 85.62% | 84.47% | 83.95% | 83.95% | 1.67% |
| `M14_stack_density` | 85.62% | 84.47% | 83.95% | 83.95% | 1.67% |
| `M16_boost_b2b4` | 85.41% | 83.84% | 83.53% | 83.53% | 1.88% |
| `M19_divide_adaptive` | 83.95% | 83.95% | 83.95% | 83.95% | 0.00% |

> Drop@20% = selisih Acc antara noise=0% dan noise=20% (lebih kecil = lebih robust)

---

## Dimensi 4: Domain Breakdown (DAMIMAS vs LONSUM)

### Domain: DAMIMAS (n=854)

| Rank | Method | Acc ±1 | MAE | Gagal |
|---:|---|---:|---:|---:|
| 1 | `M06_weight_visibility` | 85.48% | 0.3888 | 124 |
| 2 | `M20_weight_visibility_grid` | 85.48% | 0.3888 | 124 |
| 3 | `M15_divide_global` | 85.25% | 0.4063 | 126 |
| 4 | `M11_median_b2` | 85.13% | 0.4300 | 127 |
| 5 | `M12_selector_overrides` | 85.13% | 0.4397 | 127 |
| 6 | `M17_selector_regime` | 85.01% | 0.4406 | 128 |
| 7 | `M10_entropy_divide` | 84.66% | 0.4543 | 131 |
| 8 | `M13_stack_bracket` | 84.54% | 0.4300 | 132 |
| 9 | `M14_stack_density` | 84.54% | 0.4356 | 132 |
| 10 | `M16_boost_b2b4` | 84.31% | 0.4139 | 134 |
| 11 | `M19_divide_adaptive` | 82.79% | 0.4663 | 147 |

### Domain: LONSUM (n=99)

| Rank | Method | Acc ±1 | MAE | Gagal |
|---:|---|---:|---:|---:|
| 1 | `M10_entropy_divide` | 95.96% | 0.2475 | 4 |
| 2 | `M13_stack_bracket` | 94.95% | 0.2399 | 5 |
| 3 | `M14_stack_density` | 94.95% | 0.2525 | 5 |
| 4 | `M16_boost_b2b4` | 94.95% | 0.2146 | 5 |
| 5 | `M06_weight_visibility` | 93.94% | 0.2500 | 6 |
| 6 | `M19_divide_adaptive` | 93.94% | 0.2525 | 6 |
| 7 | `M20_weight_visibility_grid` | 93.94% | 0.2500 | 6 |
| 8 | `M17_selector_regime` | 93.94% | 0.2500 | 6 |
| 9 | `M11_median_b2` | 93.94% | 0.2475 | 6 |
| 10 | `M12_selector_overrides` | 93.94% | 0.2500 | 6 |
| 11 | `M15_divide_global` | 91.92% | 0.2576 | 8 |

### Breakdown Per Split (train / val / test)

| Method | test Acc | train Acc | unknown Acc | val Acc |
|---|---:|---:|---:|---:|
| `M06_weight_visibility` | 87.95% | 86.62% | 57.14% | 86.29% |
| `M20_weight_visibility_grid` | 87.95% | 86.62% | 57.14% | 86.29% |
| `M11_median_b2` | 87.35% | 87.29% | 57.14% | 82.86% |
| `M12_selector_overrides` | 87.95% | 86.96% | 57.14% | 83.43% |
| `M15_divide_global` | 89.16% | 86.29% | 57.14% | 84.00% |
| `M17_selector_regime` | 87.95% | 86.79% | 57.14% | 83.43% |
| `M10_entropy_divide` | 89.16% | 86.45% | 57.14% | 82.86% |
| `M13_stack_bracket` | 89.16% | 85.95% | 57.14% | 83.43% |
| `M14_stack_density` | 89.16% | 85.95% | 57.14% | 83.43% |
| `M16_boost_b2b4` | 87.35% | 85.95% | 57.14% | 84.00% |
| `M19_divide_adaptive` | 87.35% | 84.28% | 57.14% | 81.71% |

---

## Ringkasan: Tradeoff Antar Dimensi

| Method | Acc ±1 | Rank Acc | ms/pohon | Rank Speed | Drop@20% | Rank Robust |
|---|---:|---:|---:|---:|---:|---:|
| `M06_weight_visibility` | 86.36% | #1 | 0.033 | #4 | 2.41% | #11 |
| `M20_weight_visibility_grid` | 86.36% | #2 | 0.035 | #5 | 2.41% | #10 |
| `M11_median_b2` | 86.04% | #3 | 0.553 | #11 | 1.26% | #4 |
| `M12_selector_overrides` | 86.04% | #4 | 0.099 | #8 | 1.26% | #5 |
| `M15_divide_global` | 85.94% | #5 | 0.007 | #1 | 0.00% | #1 |
| `M17_selector_regime` | 85.94% | #6 | 0.180 | #10 | 1.16% | #3 |
| `M10_entropy_divide` | 85.83% | #7 | 0.130 | #9 | 1.88% | #8 |
| `M13_stack_bracket` | 85.62% | #8 | 0.073 | #7 | 1.67% | #6 |
| `M14_stack_density` | 85.62% | #9 | 0.021 | #3 | 1.67% | #7 |
| `M16_boost_b2b4` | 85.41% | #10 | 0.059 | #6 | 1.88% | #9 |
| `M19_divide_adaptive` | 83.95% | #11 | 0.012 | #2 | 0.00% | #2 |

> **Rekomendasi final:** `v9_selector` untuk akurasi maksimal. Untuk pipeline real-time atau inference massal, pertimbangkan `v6_selector` atau `v5_adaptive_corrected` (lebih cepat, Acc masih >93%).
