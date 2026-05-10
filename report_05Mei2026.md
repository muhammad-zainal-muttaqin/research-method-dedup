# Benchmark Report — Dataset json_05 mei 2026 (882 Pohon)

Deduplikasi multi-view untuk menghitung tandan sawit unik per pohon.
**Dataset:** 882 JSON pohon (DAMIMAS 802 + LONSUM 80) — kanonik 1 file per `tree_name`, hasil dedup dari raw export `05 Mei 2026/`.
**Angka di bawah berasal dari run `scripts/benchmark_multidim.py` dengan `JSON_DIR=json_05 mei 2026/`.** Output mentah di [`reports/benchmark_05mei2026/`](reports/benchmark_05mei2026/).

---

## Dataset

| Item | Jumlah |
|---|---:|
| Total pohon JSON | **882** |
| DAMIMAS | 802 |
| LONSUM | 80 |
| Belum punya JSON GT | 71 (lihat `json_05 mei 2026/_MISSING.md`) |
| Sisi per pohon | 4 (mayoritas), 45 pohon 8-sisi |

---

## Divisor / Factor (882 Pohon)

```
factor[C] = total_naive[C] / total_gt[C]
```

| Kelas | GT (unik) | Naive (jumlah) | factor | Keterangan |
|---:|---:|---:|---:|---|
| B1 | 876 | 1809 | **2.065** | paling besar, merah, posisi bawah — terlihat dari banyak sisi |
| B2 | 1651 | 3058 | **1.852** | transisi |
| B3 | 4629 | 8606 | **1.859** | hitam |
| B4 | 1909 | 3131 | **1.640** | paling kecil, terhalang pelepah, sering kelewat |
| **Total** | **9065** | **16604** | **1.832** | keseluruhan |

Naive overcount: **83.2%** (vs 83.4% pada 727, 80.7% pada 478, 78.8% pada 228).

Faktor B1/B4 sangat dekat dengan dataset 727 — divisor universal sudah stabil di skala ini.

---

## Metrik Primer

Pohon dianggap **benar** jika prediksi semua 4 kelas (B1/B2/B3/B4) berada dalam ±1 dari GT.

| Metrik | Arah | Definisi |
|---|:---:|---|
| **Acc ±1** | ↑ | % pohon dengan semua kelas dalam ±1 |
| **MAE** | ↓ | rata-rata \|pred − GT\| per kelas, dirata-rata lintas pohon |
| **MTE** | ↓ | mean total error = Σ\|pred − GT\| per pohon, rata-rata |
| **n_fail** | ↓ | jumlah pohon gagal (≥1 kelas meleset >1) |

---

## Hasil Utama — 11 Algoritma (882 Pohon)

Urut berdasarkan **Acc ±1** (kemudian MAE).

| Rank | Method | Gen | Acc ±1 ↑ | MAE ↓ | MTE ↓ | Gagal ↓ |
|---:|---|---|---:|---:|---:|---:|
| 1 | `M06_weight_visibility` | v2 | **89.34%** | 0.3061 | 1.2245 | 94 |
| 2 | `M20_weight_visibility_grid` | v5 | **89.34%** | 0.3061 | 1.2245 | 94 |
| 3 | `M11_median_b2` | v9 | 88.78% | 0.3115 | 1.2460 | 99 |
| 4 | `M12_selector_overrides` | v9 | 88.78% | 0.3163 | 1.2653 | 99 |
| 5 | `M10_entropy_divide` | v8 | 88.78% | 0.3282 | 1.3129 | 99 |
| 6 | `M17_selector_regime` | v6 | 88.55% | 0.3172 | 1.2687 | 101 |
| 7 | `M13_stack_bracket` | v7 | 88.44% | 0.3078 | 1.2313 | 102 |
| 8 | `M14_stack_density` | v7 | 88.44% | 0.3141 | 1.2562 | 102 |
| 9 | `M16_boost_b2b4` | v8 | 88.21% | **0.2939** | **1.1757** | 104 |
| 10 | `M15_divide_global` | v1 | 88.21% | 0.3192 | 1.2766 | 104 |
| 11 | `M19_divide_adaptive` | v5 | 86.28% | 0.3342 | 1.3367 | 121 |

Sumber: [`reports/benchmark_05mei2026/accuracy_summary.csv`](reports/benchmark_05mei2026/accuracy_summary.csv).

### Akurasi Per Kelas (% pohon dalam ±1)

| Method | B1 | B2 | B3 | B4 |
|---|---:|---:|---:|---:|
| `M06_weight_visibility` | 99.7% | 98.2% | 93.5% | 97.3% |
| `M20_weight_visibility_grid` | 99.7% | 98.2% | 93.5% | 97.3% |
| `M11_median_b2` | 99.8% | 98.3% | 92.2% | 97.7% |
| `M12_selector_overrides` | 99.8% | 98.1% | 92.3% | 97.9% |
| `M10_entropy_divide` | 99.8% | 98.0% | 91.8% | 98.4% |
| `M17_selector_regime` | 99.8% | 98.1% | 92.2% | 97.7% |
| `M13_stack_bracket` | 99.9% | 98.0% | 91.8% | 98.0% |
| `M14_stack_density` | 99.9% | 98.0% | 91.8% | 98.0% |
| `M16_boost_b2b4` | 99.9% | 98.2% | 91.8% | 97.5% |
| `M15_divide_global` | 99.7% | 98.2% | 91.8% | 98.1% |
| `M19_divide_adaptive` | 99.8% | 98.1% | 89.9% | 97.7% |

B1 hampir sempurna untuk semua metode. **B3 tetap bottleneck** (89.9%–93.5%) — ambiguitas B2↔B3 masih jadi ceiling irreducible.

### Per-Class MAE (↓)

| Method | B1 | B2 | B3 | B4 |
|---|---:|---:|---:|---:|
| `M15_divide_global` | 0.141 | 0.223 | 0.618 | 0.295 |
| `M06_weight_visibility` | 0.141 | 0.226 | 0.569 | 0.289 |
| `M19_divide_adaptive` | 0.127 | 0.238 | 0.658 | 0.314 |
| `M20_weight_visibility_grid` | 0.141 | 0.226 | 0.569 | 0.289 |
| `M17_selector_regime` | 0.125 | 0.238 | 0.603 | 0.303 |
| `M13_stack_bracket` | **0.095** | 0.237 | 0.600 | 0.299 |
| `M14_stack_density` | 0.110 | 0.237 | 0.602 | 0.307 |
| `M10_entropy_divide` | 0.117 | 0.266 | 0.615 | 0.315 |
| `M16_boost_b2b4` | **0.095** | **0.210** | 0.600 | **0.271** |
| `M11_median_b2` | 0.125 | 0.215 | 0.603 | 0.303 |
| `M12_selector_overrides` | 0.125 | 0.236 | **0.603** | 0.302 |

`M16_boost_b2b4` dominan di B1, B2, B4. `M06_weight_visibility` paling baik di B3 (0.569).

### Pola Error Per Kelas (over >1 vs under <-1)

Sumber: [`accuracy_per_class.csv`](reports/benchmark_05mei2026/accuracy_per_class.csv).

| Method | B1↑ | B1↓ | B2↑ | B2↓ | B3↑ | B3↓ | B4↑ | B4↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `M06_weight_visibility` | 3 | 0 | 3 | 13 | 27 | 30 | 0 | 24 |
| `M20_weight_visibility_grid` | 3 | 0 | 3 | 13 | 27 | 30 | 0 | 24 |
| `M11_median_b2` | 2 | 0 | 1 | 14 | 51 | 18 | 8 | 12 |
| `M12_selector_overrides` | 2 | 0 | 10 | 7 | 50 | 18 | 7 | 12 |
| `M10_entropy_divide` | 2 | 0 | 9 | 9 | 55 | 17 | 8 | 6 |
| `M17_selector_regime` | 2 | 0 | 10 | 7 | 51 | 18 | 8 | 12 |
| `M13_stack_bracket` | 1 | 0 | 9 | 9 | 53 | 19 | 8 | 10 |
| `M14_stack_density` | 1 | 0 | 9 | 9 | 53 | 19 | 8 | 10 |
| `M16_boost_b2b4` | 1 | 0 | 1 | 15 | 53 | 19 | 4 | 18 |
| `M15_divide_global` | 3 | 0 | 6 | 10 | 51 | 21 | 6 | 11 |
| `M19_divide_adaptive` | 2 | 0 | 10 | 7 | 73 | 16 | 12 | 8 |

`M06_weight_visibility` & `M20_weight_visibility_grid` punya 0 overcount B4 — paling konservatif. `M16_boost_b2b4` underprediksi B2 (15 pohon) dan B4 (18 pohon) — boost-nya terlalu agresif untuk dataset besar ini.

---

## Kecepatan (ms/pohon, 30 repetisi × 882 pohon)

Sumber: [`speed_summary.csv`](reports/benchmark_05mei2026/speed_summary.csv).

| Rank | Method | ms ↓ | pohon/detik ↑ |
|---:|---|---:|---:|
| 1 | `M15_divide_global` | 0.004 | 240,248 |
| 2 | `M19_divide_adaptive` | 0.012 | 85,471 |
| 3 | `M14_stack_density` | 0.018 | 57,135 |
| 4 | `M06_weight_visibility` | 0.028 | 35,222 |
| 5 | `M20_weight_visibility_grid` | 0.030 | 33,846 |
| 6 | `M16_boost_b2b4` | 0.050 | 20,167 |
| 7 | `M13_stack_bracket` | 0.055 | 18,229 |
| 8 | `M12_selector_overrides` | 0.077 | 12,936 |
| 9 | `M17_selector_regime` | 0.104 | 9,584 |
| 10 | `M10_entropy_divide` | 0.111 | 9,046 |
| 11 | `M11_median_b2` | 0.419 | 2,387 |

`M15_divide_global` tetap paling cepat (~240k pohon/detik) dengan akurasi 88.21% — tradeoff terbaik untuk inference massal.

---

## Robustness terhadap Noise Koordinat

Gaussian noise σ=N% pada `x_norm` dan `y_norm`. Sumber: [`robustness_summary.csv`](reports/benchmark_05mei2026/robustness_summary.csv).

| Method | σ=0% | σ=5% | σ=10% | σ=20% | Drop@20% |
|---|---:|---:|---:|---:|---:|
| `M15_divide_global` | 88.21% | 88.21% | 88.21% | 88.21% | **0.00%** |
| `M19_divide_adaptive` | 86.28% | 86.28% | 86.28% | 86.28% | **0.00%** |
| `M17_selector_regime` | 88.55% | 88.32% | 87.64% | 87.30% | 1.25% |
| `M11_median_b2` | 88.78% | 88.44% | 87.76% | 87.41% | 1.37% |
| `M12_selector_overrides` | 88.78% | 88.32% | 87.64% | 87.30% | 1.48% |
| `M13_stack_bracket` | 88.44% | 87.07% | 86.39% | 86.28% | 2.16% |
| `M14_stack_density` | 88.44% | 87.07% | 86.39% | 86.28% | 2.16% |
| `M16_boost_b2b4` | 88.21% | 86.62% | 85.94% | 86.05% | 2.16% |
| `M06_weight_visibility` | 89.34% | 88.89% | 88.66% | 86.85% | 2.49% |
| `M20_weight_visibility_grid` | 89.34% | 88.89% | 88.66% | 86.85% | 2.49% |
| `M10_entropy_divide` | 88.78% | 86.62% | 86.39% | 86.28% | 2.50% |

`M15_divide_global` & `M19_divide_adaptive` 0.00% (tidak pakai koordinat). `M06_weight_visibility` paling sensitif noise (−2.49%) — tradeoff dari ketergantungan pada posisi bbox.

---

## Domain Breakdown (DAMIMAS vs LONSUM)

### DAMIMAS (n=802)

| Rank | Method | Acc ±1 | MAE | Gagal |
|---:|---|---:|---:|---:|
| 1 | `M06_weight_visibility` | 89.15% | 0.3108 | 87 |
| 2 | `M20_weight_visibility_grid` | 89.15% | 0.3108 | 87 |
| 3 | `M11_median_b2` | 88.53% | 0.3173 | 92 |
| 4 | `M12_selector_overrides` | 88.53% | 0.3223 | 92 |
| 5 | `M17_selector_regime` | 88.28% | 0.3233 | 94 |
| 6 | `M10_entropy_divide` | 88.28% | 0.3354 | 94 |
| 7 | `M15_divide_global` | 88.15% | 0.3242 | 95 |
| 8 | `M13_stack_bracket` | 88.03% | 0.3136 | 96 |
| 9 | `M14_stack_density` | 88.03% | 0.3195 | 96 |
| 10 | `M16_boost_b2b4` | 87.78% | 0.3011 | 98 |
| 11 | `M19_divide_adaptive` | 85.79% | 0.3416 | 114 |

### LONSUM (n=80) — Pertama Kali Punya GT

| Rank | Method | Acc ±1 | MAE | Gagal |
|---:|---|---:|---:|---:|
| 1 | `M10_entropy_divide` | **93.75%** | 0.2562 | 5 |
| 2 | `M13_stack_bracket` | 92.50% | 0.2500 | 6 |
| 3 | `M14_stack_density` | 92.50% | 0.2594 | 6 |
| 4 | `M16_boost_b2b4` | 92.50% | **0.2219** | 6 |
| 5 | `M06_weight_visibility` | 91.25% | 0.2594 | 7 |
| 6 | `M19_divide_adaptive` | 91.25% | 0.2594 | 7 |
| 7 | `M20_weight_visibility_grid` | 91.25% | 0.2594 | 7 |
| 8 | `M17_selector_regime` | 91.25% | 0.2562 | 7 |
| 9 | `M11_median_b2` | 91.25% | 0.2531 | 7 |
| 10 | `M12_selector_overrides` | 91.25% | 0.2562 | 7 |
| 11 | `M15_divide_global` | 88.75% | 0.2688 | 9 |

**Temuan utama domain:** Akurasi LONSUM (88.75%–93.75%) **lebih tinggi** dari DAMIMAS (85.79%–89.15%) di hampir semua metode. Hipotesis: LONSUM punya struktur kanopi lebih terbuka → bbox lebih konsisten antar sisi → noise koordinat efektif lebih rendah. Pemenang DAMIMAS (`M06_weight_visibility`) bukan pemenang LONSUM (`M10_entropy_divide`) — pasangan rekomendasi tergantung varietas target.

### Per Split (train / val / test)

| Method | test | train | val |
|---|---:|---:|---:|
| `M06_weight_visibility` | 88.10% | 89.27% | **90.67%** |
| `M20_weight_visibility_grid` | 88.10% | 89.27% | **90.67%** |
| `M16_boost_b2b4` | **88.89%** | 87.79% | 89.33% |
| `M11_median_b2` | 87.30% | 89.27% | 88.00% |
| `M12_selector_overrides` | 84.92% | **89.77%** | 88.00% |
| `M10_entropy_divide` | 86.51% | 89.11% | 89.33% |
| `M17_selector_regime` | 84.92% | 89.44% | 88.00% |
| `M13_stack_bracket` | 86.51% | 88.61% | 89.33% |
| `M14_stack_density` | 86.51% | 88.61% | 89.33% |
| `M15_divide_global` | 87.30% | 87.95% | 90.00% |
| `M19_divide_adaptive` | 84.13% | 86.80% | 86.00% |

`M16_boost_b2b4` unggul di test set (88.89%). Tidak ada satu metode yang dominan di semua split.

---

## Tradeoff Antar Dimensi

| Method | Acc | Rank Acc | ms | Rank Speed | Drop@20% | Rank Robust |
|---|---:|---:|---:|---:|---:|---:|
| `M06_weight_visibility` | 89.34% | #1 | 0.028 | #4 | 2.49% | #9 |
| `M20_weight_visibility_grid` | 89.34% | #2 | 0.030 | #5 | 2.49% | #10 |
| `M11_median_b2` | 88.78% | #3 | 0.419 | #11 | 1.37% | #4 |
| `M12_selector_overrides` | 88.78% | #4 | 0.077 | #8 | 1.48% | #5 |
| `M10_entropy_divide` | 88.78% | #5 | 0.111 | #10 | 2.50% | #11 |
| `M17_selector_regime` | 88.55% | #6 | 0.104 | #9 | 1.25% | #3 |
| `M13_stack_bracket` | 88.44% | #7 | 0.055 | #7 | 2.16% | #8 |
| `M14_stack_density` | 88.44% | #8 | 0.018 | #3 | 2.16% | #7 |
| `M16_boost_b2b4` | 88.21% | #9 | 0.050 | #6 | 2.16% | #6 |
| `M15_divide_global` | 88.21% | #10 | 0.004 | #1 | 0.00% | #1 |
| `M19_divide_adaptive` | 86.28% | #11 | 0.012 | #2 | 0.00% | #2 |

---

## Rekomendasi (882 Pohon)

| Kebutuhan | Pilihan |
|---|---|
| Acc ±1 tertinggi | `M06_weight_visibility` / `M20_weight_visibility_grid` (89.34%) |
| MAE/MTE terendah | `M16_boost_b2b4` (MAE 0.2939, MTE 1.1757) |
| Tercepat + akurasi layak | `M15_divide_global` (0.004 ms, 88.21%, drop 0%) |
| Paling robust noise | `M15_divide_global` / `M19_divide_adaptive` (drop 0.00%) |
| Pipeline LONSUM | `M10_entropy_divide` (93.75%) |
| Pipeline DAMIMAS | `M06_weight_visibility` / `M20_weight_visibility_grid` (89.15%) |
| Test set unseen | `M16_boost_b2b4` (88.89%) |
| Tidak butuh koordinat bbox | `M15_divide_global` |

---

## Perbandingan Lintas-Dataset

Acc ±1 (semua kelas dalam ±1) untuk metode kunci:

| Method | 228 (asli) | 478 | 727 (30 Apr) | **882 (5 Mei)** |
|---|---:|---:|---:|---:|
| `M06_weight_visibility` | 92.11% | 90.38% | 77.30% | **89.34%** |
| `M20_weight_visibility_grid` | 92.54% | 90.38% | 77.30% | **89.34%** |
| `M16_boost_b2b4` | 92.54% | 91.00% | 75.52% | **88.21%** |
| `M12_selector_overrides` | **98.68%** | 92.68% | 71.39% | **88.78%** |
| `M17_selector_regime` | 96.49% | 91.84% | 70.98% | **88.55%** |
| `M15_divide_global` | 90.79% | 89.12% | 72.35% | **88.21%** |
| `M19_divide_adaptive` | 93.86% | 89.96% | 67.54% | **86.28%** |

**Pengamatan kunci:**

1. **Recovery dari 727 → 882.** Semua metode naik 11–17 poin persen. Penyebab utama: GT dataset 882 adalah hasil re-anotasi dedup'd via `tools_sawit/` schema v2 — varietas LONSUM benar, 1 file per `tree_name`, label noise jauh lebih rendah dibanding snapshot 30 April.
2. **Ranking tetap stabil dengan dataset 727.** `M06_weight_visibility` & `M20_weight_visibility_grid` tetap di puncak Acc; `M12_selector_overrides` masih di tengah, **tidak** kembali ke 98.68%. Konfirmasi temuan sebelumnya: 98.68% pada subset 228 adalah overfit, bukan generalisasi.
3. **Metode sederhana (v1, v2) generalisasi paling baik.** Selisih kecil dengan metode kompleks di dataset besar.
4. **Penurunan teori 727 (akibat noisy GT) terkonfirmasi.** Setelah GT bersih (882), semua metode kembali ke level >86%, sangat dekat dengan dataset asli.

---

## Catatan

- 71 pohon belum punya JSON GT. Setelah re-annotation lengkap (953 pohon), benchmark akan dijalankan ulang.
- Parameter v6/v9 masih diturunkan dari subset 228. Re-tuning di 882 berpotensi menaikkan ranking kembali.
- B3 tetap bottleneck universal — ambiguitas B2↔B3 visual irreducible tanpa cross-view embedding (dilarang oleh constraint algorithmic-only).
- Reproduce: `JSON_DIR="json_05 mei 2026" OUT_DIR="reports/benchmark_05mei2026" python scripts/benchmark_multidim.py`.

---
