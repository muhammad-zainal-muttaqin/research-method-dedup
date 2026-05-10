# Iterasi 4 — Karakterisasi Split + Pencarian Koreksi Generalisir

Tanggal: 10 Mei 2026.
Skrip: `iter4_split_analysis.py`. Hasil: `iter4_results.csv`, `iter4_split_stats.csv`.

## Karakterisasi Split

| Split | n | DAMIMAS | LONSUM | n_dets med | GT total med | gt_B1 mean | gt_B4 mean | naive/GT ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 608 | 539 | 69 | 19,5 | 10 | 1,077 | 2,036 | 1,907 |
| val | 178 | 163 | 15 | 20,0 | 10 | **0,719** | **2,303** | **1,983** |
| test | 167 | 152 | 15 | 20,0 | 10 | 0,958 | 2,251 | 1,856 |

**Mengapa val 5,55pp lebih rendah dari train:**

1. **B1 jauh lebih jarang di val** (mean 0,72 vs 1,08 train). Median val = 0 — banyak pohon val tanpa B1 sama sekali. Setiap kelebihan prediksi B1 langsung melanggar Acc±1 (`pred=1, gt=0` masih lolos, `pred=2` gagal).
2. **B4 lebih banyak di val** (2,30 vs 2,04 train). Subset rentan under-prediction.
3. **naive/GT ratio val tertinggi 1,98** (test 1,86, train 1,91). Pohon val punya duplikasi lebih agresif — divisor yang dikalibrasi pada agregat 953 sedikit kurang untuk val.

Bukan noise random — perbedaan distribusi nyata. Plateau yang berbeda per split.

## Hasil Kandidat

| Metode | Acc all | MAE all | train | val | test | Δtrain | Δval | Δtest |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`geometric_mean_blend`** | **86,15%** | **0,3961** | 87,17 | 81,46 | 87,43 | +0,16 | 0,00 | 0,00 |
| `baseline_floor_clamped_hybrid` | 86,04% | 0,4050 | 87,01 | 81,46 | 87,43 | 0 | 0 | 0 |
| `hybrid_w70_floor` | 86,04% | 0,4037 | 87,01 | 81,46 | 87,43 | 0 | 0 | 0 |
| `naive_ceiling_clamp` | 86,04% | 0,4050 | 87,01 | 81,46 | 87,43 | 0 | 0 | 0 |
| `median3_floor` | 85,94% | **0,3930** | 86,35 | **83,15** | 87,43 | −0,66 | **+1,69** | 0 |
| `density_floor_clamp` | 85,20% | 0,4161 | 86,02 | 81,46 | 86,23 | −0,99 | 0 | −1,20 |

## Temuan Iter4

### `geometric_mean_blend` adalah perbaikan nyata namun kecil
- Acc±1 naik 0,11pp (86,04 → 86,15) — 1 pohon tambahan di train.
- MAE turun 2,2% (0,4050 → 0,3961).
- Tidak merusak val maupun test — strict equal di kedua split.
- Justifikasi struktural: rata-rata geometris dari `visibility` dan `adaptive_corrected` lebih konservatif terhadap divergensi estimator. Ketika kedua estimator sepakat, hasilnya identik dengan rata-rata aritmetika; ketika berbeda, geometris menarik ke nilai lebih kecil → mengurangi over-prediction (yang dominan menurut iter2).

### `median3_floor` punya trade-off menarik
- Val: **+1,69pp** (81,46 → 83,15). Sinyal kuat bahwa median 3 estimator memang membantu split sulit.
- Train: −0,66pp.
- MAE keseluruhan: **0,3930** (terendah di seluruh iter1–iter4).
- Tidak lolos gate "tidak ada split turun > 0,5pp" — train turun 0,66pp.
- **Honest take**: bukan kandidat produksi karena merusak train; tetapi memberi sinyal bahwa kombinasi 3 estimator (visibility + adaptive_corrected + side_coverage) tepat untuk val.

### Lainnya
- `naive_ceiling_clamp` no-op (semua estimator sudah ≤ naive secara bawaan).
- `hybrid_w` 0,55–0,75 dengan floor: semua mentok di 86,04% (sudah dibuktikan iter1).
- `density_floor_clamp` (sigma adaptive): turun 0,84pp — tidak prinsipil cukup.

## Rekomendasi Produksi

**Update kandidat juara: `geometric_mean_blend`** (Acc±1 86,15%, MAE 0,3961).
- Lolos gate generalisir (tidak ada split turun).
- Improvement kecil tetapi prinsipil dan reproducible.
- Implementasi sederhana: ganti `0.6*v + 0.4*c` dengan `sqrt(v * c)`.

## Iter5 Direction

Eksplorasi:
1. **Hybrid geometric**: bobot pada geometric mean (sqrt^k atau weighted geo mean). Possibly geo mean of 3 estimators (vis + adaptive + side_coverage).
2. **Per-split divisor**: jika perbedaan val terstrukturalisasi (naive/GT ratio 1,98 vs 1,91), maybe dup_factor adaptive ke `n_sides` per pohon dapat menutup sebagian gap. **Tetapi tidak boleh memandang split label** — harus murni ditentukan dari geometri pohon.
3. **B1 ceiling tightening prinsipil**: median val B1 = 0. Ada sinyal bahwa B1 murah hati. Cari kondisi struktural yang membatasi B1 tanpa membatasi keseluruhan.

## Catatan Kejujuran (RULES.txt)

- `geometric_mean_blend` peningkatan 0,11pp = 1 pohon dari 953. Statistik margin tipis. Honest framing: "marginal improvement, generalisir, MAE turun jelas 2,2%". Bukan breakthrough.
- `median3_floor` punya MAE terendah tapi mengorbankan train. Tidak boleh dijual sebagai produksi tanpa caveat.
- Plateau Acc±1 sekitar 86% tetap struktural — mendekati irreducible.
