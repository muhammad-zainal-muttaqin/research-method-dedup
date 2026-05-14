# Benchmark Multi-Dimensi: 11 Algoritma Dedup

**Dataset:** 953 pohon JSON (228 GT)  
**Tanggal:** 2026-04-24  
**Metrik utama:** Acc ±1 (semua kelas dalam 1 error), MAE, ms/pohon

---

## Dimensi 1: Akurasi (Acc ±1 per kelas)

Pohon dianggap **benar** jika semua 4 kelas masing-masing dalam ±1 dari GT.

| Rank | Method | Gen | Acc ±1 | MAE | MTE | Gagal |
|---:|---|---|---:|---:|---:|---:|
| 1 | `M06_weight_visibility` | ? | **85.73%** | 0.3864 | 1.5456 | 136 |
| 2 | `M20_weight_visibility_grid` | ? | **85.73%** | 0.3864 | 1.5456 | 136 |
| 3 | `M11_median_b2` | ? | **84.99%** | 0.4242 | 1.6967 | 143 |
| 4 | `M12_selector_overrides` | ? | **84.99%** | 0.4342 | 1.7366 | 143 |
| 5 | `M10_entropy_divide` | ? | **84.89%** | 0.4470 | 1.7880 | 144 |
| 6 | `M13_stack_bracket` | ? | **84.78%** | 0.4231 | 1.6925 | 145 |
| 7 | `M14_stack_density` | ? | **84.78%** | 0.4294 | 1.7177 | 145 |
| 8 | `M17_selector_regime` | ? | **84.78%** | 0.4349 | 1.7398 | 145 |
| 9 | `M15_divide_global` | ? | **84.58%** | 0.4042 | 1.6170 | 147 |
| 10 | `M16_boost_b2b4` | ? | **84.58%** | 0.4048 | 1.6191 | 147 |
| 11 | `M19_divide_adaptive` | ? | **82.69%** | 0.4583 | 1.8332 | 165 |

> MTE = Mean Total Error (jumlah absolut error semua kelas, rata-rata per pohon)

### Akurasi Per Kelas (% pohon dalam ±1)

| Method | B1 | B2 | B3 | B4 |
|---|---:|---:|---:|---:|
| `M06_weight_visibility` | 97.3% | 95.4% | 90.3% | 97.0% |
| `M20_weight_visibility_grid` | 97.3% | 95.4% | 90.3% | 97.0% |
| `M11_median_b2` | 97.4% | 95.2% | 88.7% | 97.3% |
| `M12_selector_overrides` | 97.4% | 94.9% | 88.8% | 97.4% |
| `M10_entropy_divide` | 97.1% | 94.8% | 88.3% | 97.7% |
| `M13_stack_bracket` | 97.5% | 94.8% | 88.5% | 97.5% |
| `M14_stack_density` | 97.5% | 94.8% | 88.5% | 97.5% |
| `M17_selector_regime` | 97.4% | 94.9% | 88.7% | 97.3% |
| `M15_divide_global` | 97.3% | 95.4% | 88.7% | 97.7% |
| `M16_boost_b2b4` | 97.5% | 95.1% | 88.5% | 97.0% |
| `M19_divide_adaptive` | 97.4% | 94.9% | 86.6% | 97.3% |

### Pola Error Per Kelas (over >1 / under <-1, jumlah pohon)

| Method | B1↑ | B1↓ | B2↑ | B2↓ | B3↑ | B3↓ | B4↑ | B4↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `M06_weight_visibility` | 26 | 0 | 32 | 12 | 64 | 28 | 5 | 24 |
| `M20_weight_visibility_grid` | 26 | 0 | 32 | 12 | 64 | 28 | 5 | 24 |
| `M11_median_b2` | 25 | 0 | 33 | 13 | 92 | 16 | 15 | 11 |
| `M12_selector_overrides` | 25 | 0 | 43 | 6 | 91 | 16 | 14 | 11 |
| `M10_entropy_divide` | 28 | 0 | 42 | 8 | 96 | 15 | 17 | 5 |
| `M13_stack_bracket` | 24 | 0 | 42 | 8 | 93 | 17 | 15 | 9 |
| `M14_stack_density` | 24 | 0 | 42 | 8 | 93 | 17 | 15 | 9 |
| `M17_selector_regime` | 25 | 0 | 43 | 6 | 92 | 16 | 15 | 11 |
| `M15_divide_global` | 26 | 0 | 35 | 9 | 89 | 19 | 11 | 11 |
| `M16_boost_b2b4` | 24 | 0 | 33 | 14 | 93 | 17 | 11 | 18 |
| `M19_divide_adaptive` | 25 | 0 | 43 | 6 | 114 | 14 | 19 | 7 |

---

## Dimensi 2: Kecepatan (ms/pohon)

Diukur dengan 30 repetisi per metode, 953 pohon per repetisi.

| Rank | Method | Mean ms | Median ms | Std ms | pohon/detik |
|---:|---|---:|---:|---:|---:|
| 1 | `M15_divide_global` | 0.0061 | 0.0050 | 0.0020 | 164089 |
| 2 | `M19_divide_adaptive` | 0.0102 | 0.0101 | 0.0002 | 97930 |
| 3 | `M14_stack_density` | 0.0166 | 0.0153 | 0.0022 | 60138 |
| 4 | `M06_weight_visibility` | 0.0372 | 0.0372 | 0.0002 | 26885 |
| 5 | `M20_weight_visibility_grid` | 0.0374 | 0.0372 | 0.0005 | 26766 |
| 6 | `M16_boost_b2b4` | 0.0486 | 0.0485 | 0.0006 | 20557 |
| 7 | `M13_stack_bracket` | 0.0524 | 0.0483 | 0.0095 | 19072 |
| 8 | `M12_selector_overrides` | 0.0994 | 0.0995 | 0.0007 | 10060 |
| 9 | `M10_entropy_divide` | 0.1045 | 0.1043 | 0.0010 | 9566 |
| 10 | `M17_selector_regime` | 0.1458 | 0.1515 | 0.0122 | 6859 |
| 11 | `M11_median_b2` | 0.4503 | 0.4501 | 0.0030 | 2220 |

---

## Dimensi 3: Robustness terhadap Noise Koordinat

Simulasi: tambah Gaussian noise σ=N% ke x_norm dan y_norm setiap bbox.  
Mengukur seberapa cepat akurasi turun ketika koordinat detector tidak sempurna.

| Method | σ=0% | σ=5% | σ=10% | σ=20% | Drop@20% |
|---|---:|---:|---:|---:|---:|
| `M06_weight_visibility` | 85.73% | 85.41% | 85.31% | 82.79% | 2.94% |
| `M20_weight_visibility_grid` | 85.73% | 85.41% | 85.31% | 82.79% | 2.94% |
| `M11_median_b2` | 84.99% | 84.58% | 84.37% | 83.63% | 1.36% |
| `M12_selector_overrides` | 84.99% | 84.58% | 84.26% | 83.53% | 1.46% |
| `M10_entropy_divide` | 84.89% | 83.53% | 82.69% | 82.58% | 2.31% |
| `M13_stack_bracket` | 84.78% | 83.42% | 82.79% | 82.69% | 2.09% |
| `M14_stack_density` | 84.78% | 83.42% | 82.79% | 82.69% | 2.09% |
| `M17_selector_regime` | 84.78% | 84.58% | 84.26% | 83.53% | 1.25% |
| `M15_divide_global` | 84.58% | 84.58% | 84.58% | 84.58% | 0.00% |
| `M16_boost_b2b4` | 84.58% | 83.00% | 82.48% | 82.48% | 2.10% |
| `M19_divide_adaptive` | 82.69% | 82.69% | 82.69% | 82.69% | 0.00% |

> Drop@20% = selisih Acc antara noise=0% dan noise=20% (lebih kecil = lebih robust)

---

## Dimensi 4: Domain Breakdown (DAMIMAS vs LONSUM)

### Domain: DAMIMAS (n=854)

| Rank | Method | Acc ±1 | MAE | Gagal |
|---:|---|---:|---:|---:|
| 1 | `M06_weight_visibility` | 84.78% | 0.4022 | 130 |
| 2 | `M20_weight_visibility_grid` | 84.78% | 0.4022 | 130 |
| 3 | `M11_median_b2` | 83.96% | 0.4447 | 137 |
| 4 | `M12_selector_overrides` | 83.96% | 0.4555 | 137 |
| 5 | `M15_divide_global` | 83.72% | 0.4213 | 139 |
| 6 | `M17_selector_regime` | 83.72% | 0.4564 | 139 |
| 7 | `M13_stack_bracket` | 83.61% | 0.4444 | 140 |
| 8 | `M14_stack_density` | 83.61% | 0.4499 | 140 |
| 9 | `M10_entropy_divide` | 83.61% | 0.4701 | 140 |
| 10 | `M16_boost_b2b4` | 83.37% | 0.4268 | 142 |
| 11 | `M19_divide_adaptive` | 81.38% | 0.4821 | 159 |

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

| Method | test Acc | train Acc | val Acc |
|---|---:|---:|---:|
| `M06_weight_visibility` | 86.83% | 86.16% | 83.24% |
| `M20_weight_visibility_grid` | 86.83% | 86.16% | 83.24% |
| `M11_median_b2` | 85.63% | 86.16% | 80.45% |
| `M12_selector_overrides` | 86.23% | 85.83% | 81.01% |
| `M10_entropy_divide` | 87.43% | 85.50% | 80.45% |
| `M13_stack_bracket` | 87.43% | 85.17% | 81.01% |
| `M14_stack_density` | 87.43% | 85.17% | 81.01% |
| `M17_selector_regime` | 86.23% | 85.50% | 81.01% |
| `M15_divide_global` | 87.43% | 84.68% | 81.56% |
| `M16_boost_b2b4` | 85.63% | 85.17% | 81.56% |
| `M19_divide_adaptive` | 85.63% | 82.87% | 79.33% |

---

## Ringkasan: Tradeoff Antar Dimensi

| Method | Acc ±1 | Rank Acc | ms/pohon | Rank Speed | Drop@20% | Rank Robust |
|---|---:|---:|---:|---:|---:|---:|
| `M06_weight_visibility` | 85.73% | #1 | 0.037 | #4 | 2.94% | #11 |
| `M20_weight_visibility_grid` | 85.73% | #2 | 0.037 | #5 | 2.94% | #10 |
| `M11_median_b2` | 84.99% | #3 | 0.450 | #11 | 1.36% | #4 |
| `M12_selector_overrides` | 84.99% | #4 | 0.099 | #8 | 1.46% | #5 |
| `M10_entropy_divide` | 84.89% | #5 | 0.104 | #9 | 2.31% | #9 |
| `M13_stack_bracket` | 84.78% | #6 | 0.052 | #7 | 2.09% | #6 |
| `M14_stack_density` | 84.78% | #7 | 0.017 | #3 | 2.09% | #7 |
| `M17_selector_regime` | 84.78% | #8 | 0.146 | #10 | 1.25% | #3 |
| `M15_divide_global` | 84.58% | #9 | 0.006 | #1 | 0.00% | #1 |
| `M16_boost_b2b4` | 84.58% | #10 | 0.049 | #6 | 2.10% | #8 |
| `M19_divide_adaptive` | 82.69% | #11 | 0.010 | #2 | 0.00% | #2 |

> **Rekomendasi final:** `v9_selector` untuk akurasi maksimal. Untuk pipeline real-time atau inference massal, pertimbangkan `v6_selector` atau `v5_adaptive_corrected` (lebih cepat, Acc masih >93%).
