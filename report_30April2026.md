# Benchmark Report — Dataset json_30 April 2026 (727 Pohon)

Deduplikasi multi-view untuk menghitung tandan sawit unik per pohon.  
**Dataset:** 727 JSON pohon DAMIMAS (0 LONSUM).  
**Angka benchmark di bawah sudah diperbarui dengan dataset baru.**

---

## Dataset

| Item | Jumlah |
|---|---:|
| Total pohon JSON | 727 (DAMIMAS) |
| Sisi per pohon | 4 (mayoritas), 8 (~45 terbaru) |

---

## Divisor / Factor

Dari 727 pohon dengan ground truth:

```
factor[C] = total_naive[C] / total_gt[C]
```

| Kelas | GT (unik) | Naive (jumlah) | factor | Keterangan |
|---:|---:|---:|---:|---|
| B1 | 841 | 1732 | **2.060** | paling besar, merah, posisi bawah — terlihat dari banyak sisi |
| B2 | 1477 | 2720 | **1.842** | transisi, masih besar |
| B3 | 3811 | 7093 | **1.861** | hitam, masih besar |
| B4 | 1686 | 2788 | **1.654** | paling kecil, posisi atas, terhalang pelepah — sering terlewat |
| **Total** | 7815 | 14333 | **1.834** | keseluruhan |

Naive overcount: **83.4%** (vs 80.7% pada dataset 478, vs 78.8% pada dataset 228).

---

## Metrik Primer

Seluruh hasil di bawah dihitung pada 727 pohon JSON.

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
| 1 | `v8_b2_b4_boosted` | **0.3067** | **27.24%** | **1.2270** | 75.52% |
| 2 | `v2_visibility` | 0.3116 | 26.27% | 1.2462 | **77.30%** |
| 3 | `v5_best_visibility` | 0.3116 | 26.27% | 1.2462 | **77.30%** |
| 4 | `v7_stacking_bracketed` | 0.3181 | 26.55% | 1.2724 | 71.25% |
| 5 | `v9_b2_median_v6` | 0.3219 | 23.93% | 1.2875 | 72.63% |
| 6 | `v7_stacking_density` | 0.3232 | 25.45% | 1.2930 | 70.56% |
| 7 | `v9_selector` | 0.3267 | 23.93% | 1.3067 | 71.39% |
| 8 | `v6_selector` | 0.3287 | 23.38% | 1.3150 | 70.98% |
| 9 | `v1_corrected` | 0.3291 | 24.35% | 1.3164 | 72.35% |
| 10 | `v8_entropy_modulated` | 0.3394 | 24.76% | 1.3576 | 69.05% |
| 11 | `v5_adaptive_corrected` | 0.3494 | 21.60% | 1.3975 | 67.54% |

Sumber: `reports/benchmark_multidim/accuracy_summary.csv` (727 pohon × 11 metode).

### Per-Class MAE (↓ lebih kecil lebih baik)

| Method | B1 ↓ | B2 ↓ | B3 ↓ | B4 ↓ |
|---|---:|---:|---:|---:|
| `v1_corrected` | 0.171 | 0.224 | 0.609 | 0.312 |
| `v2_visibility` | 0.169 | 0.222 | 0.554 | 0.301 |
| `v5_adaptive_corrected` | 0.143 | 0.253 | 0.655 | 0.347 |
| `v5_best_visibility` | 0.169 | 0.222 | 0.554 | 0.301 |
| `v6_selector` | 0.142 | 0.252 | 0.594 | 0.327 |
| `v7_stacking_bracketed` | **0.109** | 0.241 | 0.597 | 0.326 |
| `v7_stacking_density` | 0.122 | 0.241 | 0.598 | 0.332 |
| `v8_entropy_modulated` | 0.136 | 0.268 | 0.608 | 0.345 |
| `v8_b2_b4_boosted` | **0.109** | **0.226** | 0.597 | **0.296** |
| `v9_b2_median_v6` | 0.142 | 0.224 | 0.594 | 0.327 |
| `v9_selector` | 0.142 | 0.248 | **0.592** | 0.326 |

B1 termudah untuk semua metode. **B3 adalah bottleneck** universal.

### Per-Class Mean Error (Bias) (→0 ideal)

Nilai positif = overcount, negatif = undercount, nol = tidak bias.

| Method | B1 →0 | B2 →0 | B3 →0 | B4 →0 |
|---|---:|---:|---:|---:|
| `v1_corrected` | +0.146 | +0.029 | +0.172 | −0.015 |
| `v2_visibility` | +0.147 | +0.010 | +0.021 | −0.213 |
| `v5_adaptive_corrected` | +0.080 | +0.110 | +0.316 | +0.080 |
| `v5_best_visibility` | +0.147 | +0.010 | +0.021 | −0.213 |
| `v6_selector` | +0.081 | +0.103 | +0.237 | +0.033 |
| `v7_stacking_bracketed` | +0.067 | +0.081 | +0.220 | +0.043 |
| `v7_stacking_density` | +0.054 | +0.081 | +0.219 | +0.037 |
| `v8_entropy_modulated` | +0.109 | +0.125 | +0.261 | +0.106 |
| `v8_b2_b4_boosted` | +0.067 | −0.072 | +0.220 | −0.131 |
| `v9_b2_median_v6` | +0.081 | −0.023 | +0.237 | +0.033 |
| `v9_selector` | +0.081 | +0.096 | +0.231 | +0.032 |

`v8_b2_b4_boosted` punya MAE B1 dan B4 terendah, tapi underprediksi B2 (−0.072) dan B4 (−0.131). `v2_visibility` underprediksi B4 (−0.213) paling besar.

---

## Metrik Pelengkap

### Acc ±1 per kelas per pohon

Sumber: `accuracy_summary.csv`.

| Rank | Method | Acc ±1 ↑ | Gagal ↓ |
|---:|---|---:|---:|
| 1 | `v2_visibility` | **77.30%** | 165 |
| 2 | `v5_best_visibility` | **77.30%** | 165 |
| 3 | `v8_b2_b4_boosted` | 75.52% | 178 |
| 4 | `v9_b2_median_v6` | 72.63% | 199 |
| 5 | `v1_corrected` | 72.35% | 201 |
| 6 | `v9_selector` | 71.39% | 208 |
| 7 | `v7_stacking_bracketed` | 71.25% | 209 |
| 8 | `v6_selector` | 70.98% | 211 |
| 9 | `v7_stacking_density` | 70.56% | 214 |
| 10 | `v8_entropy_modulated` | 69.05% | 225 |
| 11 | `v5_adaptive_corrected` | 67.54% | 236 |

### Kecepatan (ms/pohon, 30 repetisi × 727 pohon)

| Method | ms ↓ | pohon/detik ↑ |
|---|---:|---:|
| `v1_corrected` | 0.004 | 284,792 |
| `v5_adaptive_corrected` | 0.008 | 133,213 |
| `v7_stacking_density` | 0.015 | 64,853 |
| `v2_visibility` | 0.024 | 41,356 |
| `v5_best_visibility` | 0.024 | 41,165 |
| `v8_b2_b4_boosted` | 0.048 | 20,808 |
| `v7_stacking_bracketed` | 0.048 | 20,723 |
| `v9_selector` | 0.079 | 12,623 |
| `v6_selector` | 0.100 | 10,041 |
| `v8_entropy_modulated` | 0.104 | 9,626 |
| `v9_b2_median_v6` | 0.420 | 2,381 |

### Robustness (Gaussian noise σ=20% pada koordinat)

| Method | Drop Acc @ σ=20% ↓ |
|---|---:|
| `v1_corrected`, `v5_adaptive_corrected` | 0.00% (tidak pakai koordinat) |
| `v8_b2_b4_boosted` | −1.93% |
| `v7_stacking_bracketed`, `v7_stacking_density` | −2.34% |
| `v6_selector` | −2.34% |
| `v9_b2_median_v6` | −2.34% |
| `v9_selector` | −2.47% |
| `v8_entropy_modulated` | −2.89% |
| `v2_visibility`, `v5_best_visibility` | −3.44% (paling sensitif) |

### Per Split (train=606, val=71, test=50)

| Method | test ↑ | train ↑ | val ↑ |
|---|---:|---:|---:|
| `v9_selector` | **90.00%** | 89.60% | 85.92% |
| `v7_stacking_bracketed` | 88.00% | 88.78% | 85.92% |
| `v7_stacking_density` | 88.00% | 88.78% | 85.92% |
| `v8_b2_b4_boosted` | 88.00% | 87.95% | 84.51% |
| `v9_b2_median_v6` | 88.00% | 89.44% | 85.92% |
| `v1_corrected` | 86.00% | 87.95% | **88.73%** |
| `v2_visibility` | 86.00% | **90.10%** | 85.92% |
| `v5_best_visibility` | 86.00% | **90.10%** | 85.92% |
| `v6_selector` | 86.00% | 89.44% | 85.92% |
| `v5_adaptive_corrected` | 84.00% | 86.63% | 83.10% |
| `v8_entropy_modulated` | 88.00% | 89.44% | 84.51% |

`v9_selector` unggul di test set (90.00%), sementara `v2_visibility` dan `v5_best_visibility` unggul di train set (90.10%).

---

## Rekomendasi

| Kebutuhan | Pilihan |
|---|---|
| Macro MAE terendah | `v8_b2_b4_boosted` (0.3067) |
| Total ±1 tertinggi | `v2_visibility` / `v5_best_visibility` (77.30%) |
| Throughput tinggi + Acc >71% | `v7_stacking_density` (0.015 ms, 70.56%) |
| Tradeoff akurasi/kecepatan | `v7_stacking_bracketed` (0.048 ms, 71.25%) |
| Pipeline TXT noisy (tanpa GT) | `v5_adaptive_corrected` atau `v1_corrected` |
| Tidak butuh koordinat bbox | `v1_corrected` |

---

## Evolusi Metode

| Gen | Method | Macro MAE ↓ | Catatan |
|---|---|---|---:|
| naive | — | — | overcount ~83.4% |
| v1 | `v1_corrected` | 0.3291 | divisor global |
| v2 | `v2_visibility` | 0.3116 | geometri sederhana |
| v5 | `v5_adaptive_corrected` | 0.3494 | adaptive divisor |
| v6 | `v6_selector` | 0.3287 | **titik balik** — routing per regime |
| v7 | `v7_stacking_bracketed` | 0.3181 | stacking density family |
| v8 | `v8_b2_b4_boosted` | **0.3067** | per-kelas boosting |
| v9 | `v9_selector` | 0.3267 | narrow overrides di atas v6 |

**Catatan penting:** Di dataset 727 pohon, `v8_b2_b4_boosted` unggul secara Macro MAE (0.3067), sementara `v2_visibility` unggul di Total ±1 accuracy (77.30%). Pola ini berbeda drastis dari dataset 228 dan 478 — menunjukkan bahwa dataset yang lebih besar dan lebih beragam mengubah tradeoff antar metode secara signifikan. v9_selector yang sebelumnya unggul di dataset kecil (98.68% di 228 pohon) turun ke peringkat 7 di dataset 727 pohon, mengindikasikan overfitting pada regime sempit dataset asli.

---

## Dataset

| Item | Jumlah |
|---|---:|
| Total pohon JSON | 727 (DAMIMAS) |
| Sisi per pohon | 4 (mayoritas), 8 (~45 terbaru) |

**Sumber folder:** `json_30 April 2026/`

> Catatan: Dataset 727 ini **DAMIMAS-only**. LONSUM hanya ada di subset TXT (tanpa JSON).

---

## Catatan

- Akurasi Top-1 turun dari 98.68% (228 trees) → 92.68% (478 trees) → 77.30% (727 trees) — wajar karena dataset terus membesar dan lebih beragam.
- `v8_b2_b4_boosted` menunjukkan Macro MAE terendah (0.3067), konsisten dengan temuan di dataset 478.
- `v2_visibility` dan `v5_best_visibility` memimpin Total ±1 — metode sederhana (weighted sum visibility) justru generalisasi paling baik di dataset besar.
- `v9_selector` kehilangan keunggulannya: dari rank 1 di 228 pohon (98.68%) menjadi rank 7 di 727 pohon (71.39%), mengindikasikan bahwa narrow overrides v9 overfit pada pola-pola spesifik di dataset 228.
- Peringatan penting: v9 dan v6 menggunakan parameter yang diturunkan dari dataset 228, sehingga performanya di dataset 727 adalah ukuran generalisasi — bukan ukuran absolut.

---
