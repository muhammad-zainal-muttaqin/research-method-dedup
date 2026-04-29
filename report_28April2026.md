# Benchmark Report — Dataset json_28 April 2026 (478 Pohon)

Deduplikasi multi-view untuk menghitung tandan sawit unik per pohon.  
**Dataset:** 478 JSON pohon DAMIMAS (0 LONSUM).  
**Angka benchmark di bawah sudah diperbarui dengan dataset baru.**  
Perbandingan dengan hasil sebelumnya (228 pohon) bisa dilihat dari perbedaan nilai di setiap tabel.

---

## Dataset

| Item | Jumlah |
|---|---:|
| Total pohon JSON | 478 (DAMIMAS) |
| Sisi per pohon | 4 (mayoritas), 8 (~45 terbaru) |

---

## Divisor / Factor

Dari 478 pohon dengan ground truth:

```
factor[C] = total_naive[C] / total_gt[C]
```

| Kelas | GT (unik) | Naive (jumlah) | factor | Keterangan |
|---:|---:|---:|---:|---|
| B1 | 573 | 1166 | **2.035** | paling besar, merah, posisi bawah — terlihat dari banyak sisi |
| B2 | 1030 | 1866 | **1.812** | transisi, masih besar |
| B3 | 2490 | 4540 | **1.823** | hitam, masih besar |
| B4 | 1103 | 1817 | **1.647** | paling kecil, posisi atas, terhalang pelepah — sering terlewat |
| **Total** | 5196 | 9389 | **1.807** | keseluruhan |

Naive overcount: **80.7%** (vs 78.8% pada dataset 228).

---

## Metrik Primer

Seluruh hasil di bawah dihitung pada 478 pohon JSON.

| Metrik | Arah | Definisi |
|---|:---:|---|
| **Per-class MAE** | ↓ | rata-rata \|pred − GT\| tiap kelas (B1/B2/B3/B4), dirata-rata lintas pohon |
| **Macro class-MAE** | ↓ | rata-rata dari 4 per-class MAE (bobot sama antar kelas) |
| **Exact accuracy** | ↑ | % pohon dengan prediksi tepat sama GT di **semua** kelas |
| **Total count MAE** | ↓ | rata-rata \|Σpred − ΣGT\| per pohon |
| **Total ±1 accuracy** | ↑ | % pohon dengan total count dalam ±1 dari total GT |
| **Per-class mean error** | →0 | rata-rata (pred − GT) per kelas — mengukur **bias arah** (+ overcount, − undercount) |

> Legenda: **↓** makin kecil makin baik, **↑** makin besar makin baik, **→0** ideal mendekati nol.

---

## Hasil Utama — 11 Algoritma

Urut berdasarkan **macro class-MAE**.

| Rank | Method | Macro MAE ↓ | Exact % ↑ | Total MAE ↓ | Total ±1 % ↑ |
|---:|---|---:|---:|---:|---:|
| 1 | `v8_b2_b4_boosted` | **0.2762** | **29.71%** | **1.1046** | 91.00% |
| 2 | `v9_b2_median_v6` | 0.2840 | 26.36% | 1.1360 | **92.68%** |
| 3 | `v2_visibility` | 0.2856 | 28.03% | 1.1423 | 90.38% |
| 4 | `v5_best_visibility` | 0.2856 | 28.03% | 1.1423 | 90.38% |
| 5 | `v7_stacking_bracketed` | 0.2887 | 28.24% | 1.1548 | 91.84% |
| 6 | `v9_selector` | 0.2892 | 26.36% | 1.1569 | 92.68% |
| 7 | `v6_selector` | 0.2929 | 25.52% | 1.1715 | 91.84% |
| 8 | `v7_stacking_density` | 0.2939 | 27.20% | 1.1757 | 91.84% |
| 9 | `v1_corrected` | 0.3039 | 26.57% | 1.2155 | 89.12% |
| 10 | `v8_entropy_modulated` | 0.3060 | 27.62% | 1.2238 | 91.63% |
| 11 | `v5_adaptive_corrected` | 0.3081 | 23.85% | 1.2322 | 89.96% |

Sumber: `reports/benchmark_multidim/accuracy_summary.csv` (478 pohon × 11 metode).

### Per-Class MAE (↓ lebih kecil lebih baik)

| Method | B1 ↓ | B2 ↓ | B3 ↓ | B4 ↓ |
|---|---:|---:|---:|---:|
| `v1_corrected` | 0.146 | 0.226 | 0.527 | 0.316 |
| `v2_visibility` | 0.146 | 0.226 | **0.471** | 0.299 |
| `v5_adaptive_corrected` | 0.136 | 0.255 | 0.506 | 0.335 |
| `v5_best_visibility` | 0.146 | 0.226 | 0.471 | 0.299 |
| `v6_selector` | 0.134 | 0.255 | 0.467 | 0.316 |
| `v7_stacking_bracketed` | **0.109** | 0.255 | 0.473 | 0.318 |
| `v7_stacking_density` | 0.121 | 0.255 | 0.475 | 0.324 |
| `v8_entropy_modulated` | 0.130 | 0.280 | 0.485 | 0.329 |
| `v8_b2_b4_boosted` | **0.109** | 0.234 | 0.473 | **0.289** |
| `v9_b2_median_v6` | 0.134 | **0.220** | 0.467 | 0.316 |
| `v9_selector` | 0.134 | 0.247 | 0.462 | 0.314 |

B1 termudah untuk semua metode. **B3 adalah bottleneck** universal.

### Per-Class Mean Error (Bias) (→0 ideal)

Nilai positif = overcount, negatif = undercount, nol = tidak bias.

| Method | B1 →0 | B2 →0 | B3 →0 | B4 →0 |
|---|---:|---:|---:|---:|
| `v1_corrected` | +0.134 | −0.008 | +0.059 | −0.019 |
| `v2_visibility` | +0.134 | −0.021 | −0.098 | −0.228 |
| `v5_adaptive_corrected` | +0.082 | +0.092 | +0.172 | +0.059 |
| `v5_best_visibility` | +0.134 | −0.021 | −0.098 | −0.228 |
| `v6_selector` | +0.084 | +0.084 | +0.111 | +0.019 |
| `v7_stacking_bracketed` | +0.067 | +0.067 | +0.100 | +0.038 |
| `v7_stacking_density` | +0.054 | +0.067 | +0.098 | +0.031 |
| `v8_entropy_modulated` | +0.109 | +0.109 | +0.138 | +0.090 |
| `v8_b2_b4_boosted` | +0.067 | −0.105 | +0.100 | −0.159 |
| `v9_b2_median_v6` | +0.084 | −0.056 | +0.111 | +0.019 |
| `v9_selector` | +0.084 | +0.075 | +0.103 | +0.017 |

`v8_b2_b4_boosted` punya bias B1 dan B4 paling kecil, tapi underprediksi B2 (−0.105) dan B4 (−0.159). `v2_visibility` underprediksi B4 (−0.228) paling besar.

---

## Metrik Pelengkap

### Acc ±1 per kelas per pohon

Sumber: `accuracy_summary.csv`.

| Rank | Method | Acc ±1 ↑ | Gagal ↓ |
|---:|---|---:|---:|
| 1 | `v9_b2_median_v6` | **92.68%** | 35 |
| 2 | `v9_selector` | **92.68%** | 35 |
| 3 | `v7_stacking_bracketed` | 91.84% | 39 |
| 4 | `v6_selector` | 91.84% | 39 |
| 5 | `v7_stacking_density` | 91.84% | 39 |
| 6 | `v8_entropy_modulated` | 91.63% | 40 |
| 7 | `v8_b2_b4_boosted` | 91.00% | 43 |
| 8 | `v2_visibility` | 90.38% | 46 |
| 9 | `v5_best_visibility` | 90.38% | 46 |
| 10 | `v5_adaptive_corrected` | 89.96% | 48 |
| 11 | `v1_corrected` | 89.12% | 52 |

### Kecepatan (ms/pohon, 30 repetisi × 478 pohon)

| Method | ms ↓ | pohon/detik ↑ |
|---|---:|---:|
| `v1_corrected` | 0.004 | 280,081 |
| `v5_adaptive_corrected` | 0.007 | 136,388 |
| `v7_stacking_density` | 0.014 | 72,286 |
| `v2_visibility` | 0.023 | 43,343 |
| `v5_best_visibility` | 0.023 | 43,468 |
| `v7_stacking_bracketed` | 0.049 | 20,442 |
| `v8_b2_b4_boosted` | 0.049 | 20,232 |
| `v9_selector` | 0.079 | 12,637 |
| `v6_selector` | 0.101 | 9,941 |
| `v8_entropy_modulated` | 0.105 | 9,568 |
| `v9_b2_median_v6` | 0.433 | 2,310 |

### Robustness (Gaussian noise σ=20% pada koordinat)

| Method | Drop Acc @ σ=20% ↓ |
|---|---:|
| `v1_corrected`, `v5_adaptive_corrected` | 0.00% (tidak pakai koordinat) |
| `v8_b2_b4_boosted` | −1.04% |
| `v2_visibility`, `v5_best_visibility` | −1.05% |
| `v8_entropy_modulated` | −1.67% |
| `v6_selector` | −1.67% |
| `v7_stacking_bracketed`, `v7_stacking_density` | −1.88% |
| `v9_b2_median_v6` | −1.89% |
| `v9_selector` | −2.09% (paling sensitif) |

### Per Split (train=355, val=39, test=84)

| Method | test ↑ | train ↑ | val ↑ |
|---|---:|---:|---:|
| `v9_b2_median_v6` | **89.29%** | 93.52% | 92.31% |
| `v9_selector` | 86.90% | **94.08%** | 92.31% |
| `v6_selector` | 84.52% | 93.52% | 92.31% |
| `v7_stacking_bracketed` | 85.71% | 93.24% | 92.31% |
| `v7_stacking_density` | 85.71% | 93.24% | 92.31% |
| `v8_entropy_modulated` | 85.71% | 93.24% | 89.74% |
| `v8_b2_b4_boosted` | **89.29%** | 91.83% | 87.18% |
| `v2_visibility` | 86.90% | 91.55% | 87.18% |
| `v5_best_visibility` | 86.90% | 91.55% | 87.18% |
| `v1_corrected` | 85.71% | 89.58% | 92.31% |
| `v5_adaptive_corrected` | 83.33% | 91.55% | 89.74% |

`v9_b2_median_v6` unggul di test set (89.29%), sementara `v9_selector` unggul di train set (94.08%).

---

## Rekomendasi

| Kebutuhan | Pilihan |
|---|---|
| Akurasi Total ±1 tertinggi | `v9_b2_median_v6` / `v9_selector` (92.68%) |
| Macro MAE terendah | `v8_b2_b4_boosted` (0.2762) |
| Throughput tinggi + Acc >90% | `v7_stacking_density` (0.014 ms, 91.84%) |
| Tradeoff akurasi/kecepatan | `v7_stacking_bracketed` (0.049 ms, 91.84%) |
| Pipeline TXT noisy (tanpa GT) | `v5_adaptive_corrected` atau `v1_corrected` |
| Tidak butuh koordinat bbox | `v1_corrected` |

---

## Evolusi Metode

| Gen | Method | Macro MAE ↓ | Catatan |
|---|---|---|---:|
| naive | — | — | overcount ~80.7% |
| v1 | `v1_corrected` | 0.3039 | divisor global |
| v2 | `v2_visibility` | 0.2856 | geometri sederhana |
| v5 | `v5_adaptive_corrected` | 0.3081 | adaptive divisor |
| v6 | `v6_selector` | 0.2929 | **titik balik** — routing per regime |
| v7 | `v7_stacking_bracketed` | 0.2887 | stacking density family |
| v8 | `v8_b2_b4_boosted` | **0.2762** | per-kelas boosting |
| v9 | `v9_selector` | 0.2892 | narrow overrides di atas v6 |

**Catatan penting:** Di dataset 478 pohon, `v8_b2_b4_boosted` justru unggul secara Macro MAE, sementara `v9_selector` dan `v9_b2_median_v6` unggul di Total ±1 accuracy. Pola ini berbeda dari dataset 228 pohon sebelumnya — menunjukkan bahwa dataset yang lebih besar dan lebih beragam mengubah tradeoff antar metode.

---

## Dataset

| Item | Jumlah |
|---|---:|
| Total pohon JSON | 478 (DAMIMAS) |
| Sisi per pohon | 4 (mayoritas), 8 (~45 terbaru) |

**Sumber folder:** `json_28 April 2026/`

---

## Catatan

- Dataset 478 pohon ini **hanya DAMIMAS** (tidak ada LONSUM), berbeda dari dataset 228 sebelumnya yang memiliki DAMIMAS + LONSUM.
- Akurasi Top-1 turun dari 98.68% (228 trees) menjadi 92.68% (478 trees) — wajar karena dataset lebih besar dan beragam.
- `v8_b2_b4_boosted` menunjukkan Macro MAE terendah (0.2762), mengindikasikan bahwa boosting per-kelas B2/B4 efektif secara rata-rata meskipun akurasi total ±1 sedikit lebih rendah.

---

Sumber: `reports/benchmark_multidim/` — regenerate dengan:
```bash
# modifikasi JSON_DIR di benchmark_multidim.py ke "json_28 April 2026"
python scripts/benchmark_multidim.py
```
