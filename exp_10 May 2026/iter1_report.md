# Iterasi 1 — Ensemble Heuristik (10 Mei 2026)

Dataset: `Brand-New-Dataset-YOLO/json/` (953 pohon, GT lengkap).
Skrip: `exp_10 May 2026/iter1_ensemble.py`.
Hasil mentah: `exp_10 May 2026/iter1_results.csv`.

## Tujuan

Melampaui `hybrid_vis_corr` (Acc±1 = 86,04%, MAE = 0,4077) tanpa pelatihan model.

## Kandidat yang Diuji

1. `median_top5` — median per kelas dari lima estimator papan atas.
2. `trimmed_mean5` — rata-rata setelah membuang nilai min dan max.
3. `floor_clamped_hybrid` — `hybrid_vis_corr` di-*clamp* dengan batas bawah `max_per_side` (jumlah deteksi maksimum dalam satu sisi). Justifikasi: satu tandan tidak dapat tampak dua kali pada bingkai yang sama, sehingga jumlah unik tidak mungkin lebih kecil dari jumlah yang teramati pada satu sisi.
4. `floor_clamped_vis` — sama, tetapi alasnya `visibility_count`.
5. `triple_avg` — rata-rata setara dari `visibility`, `adaptive_corrected`, `side_coverage`.
6. `hybrid_w{45..80}` — sapuan bobot mixing untuk `hybrid_vis_corr`.

## Hasil

| Metode | Acc±1 | MAE | n_gagal |
|---|---:|---:|---:|
| `floor_clamped_hybrid` | **86,04%** | **0,4050** | 133 |
| `triple_avg` | 86,04% | 0,4063 | 133 |
| `hybrid_w70` | 86,04% | 0,4063 | 133 |
| `hybrid_w65` | 86,04% | 0,4069 | 133 |
| `hybrid_vis_corr` (baseline) | 86,04% | 0,4077 | 133 |
| `hybrid_w55` | 86,04% | 0,4077 | 133 |
| `hybrid_w60` | 86,04% | 0,4077 | 133 |
| `floor_clamped_vis` | 85,94% | **0,3930** | 134 |
| `trimmed_mean5` | 85,94% | 0,4019 | 134 |
| `baseline_visibility` | 85,94% | 0,3956 | 134 |
| `median_top5` | 85,94% | 0,4022 | 134 |
| `hybrid_w80` | 85,94% | 0,4008 | 134 |
| `hybrid_w50` | 83,63% | 0,4370 | 156 |
| `hybrid_w45` | 83,21% | 0,4436 | 160 |

## Temuan

1. **Plateau Acc±1 di 86,04%.** Lima kandidat baru terikat dengan baseline — *ceiling* sama.
2. **`floor_clamped_hybrid` adalah pemenang kecil:** Acc±1 identik, MAE turun 0,66% (0,4077 → 0,4050) dengan justifikasi struktural yang kokoh.
3. **`floor_clamped_vis` punya MAE terendah keseluruhan (0,3930)** tetapi mengorbankan satu pohon Acc±1.
4. **Bobot hybrid tidak sensitif** di rentang 0,55–0,70 — empat bobot terikat di 86,04%. Bobot ekstrem (0,45–0,50) merusak akurasi.
5. **Median dan trimmed mean tidak membantu** — keduanya sama dengan baseline visibility.

## Catatan Kejujuran (sesuai RULES.txt)

- `floor_clamped_hybrid` tidak mengandung *hack*. `max_per_side` adalah batas bawah fisik yang valid (satu tandan tidak mungkin terdeteksi dua kali pada satu bingkai yang sama).
- `triple_avg` adalah ensemble lurus tanpa parameter ad hoc.
- Sapuan `hybrid_w` murni eksploratif; tidak ada bobot yang di-*hard-code* hanya untuk mengejar dataset ini di luar yang sudah ada di baseline.
- Tidak ada metode yang melanggar batasan "tanpa training".

## Rekomendasi Lanjut (Iterasi 2)

133 pohon gagal pada semua kandidat 86,04%. Perlu inspeksi tree-level: apakah kegagalan terkonsentrasi pada satu kelas (kemungkinan B2↔B3) atau distribusi pohon tertentu. Iter 2 fokus pada karakterisasi `n_fail` dan koreksi terarah, bukan ensemble lebih lanjut.
