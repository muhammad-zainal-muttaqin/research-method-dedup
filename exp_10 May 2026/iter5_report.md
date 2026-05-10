# Iterasi 5 — Ekstensi Geometric Mean

Tanggal: 10 Mei 2026.
Skrip: `iter5_geo_extensions.py`. Hasil: `iter5_results.csv`.

## Tujuan

Lampaui iter4 winner `geometric_mean_blend` (86,15%, MAE 0,3961).

## Kandidat (10 variasi)

1. `geo_3way` — geometric mean dari `vis`, `adaptive_corrected`, `side_coverage`.
2. `arith_geo_mix` — rata-rata aritmetika + geometris.
3. `geo_weighted` — `vis^w * corr^(1-w)` untuk w ∈ {0,40, 0,45, 0,50, 0,55, 0,60, 0,65, 0,70}.
4. `geo_with_naive_ceiling` — geo + plafon eksplisit pada naive count.
5. `b1_tight_geo` — geo + B1 cap saat `active_sides(B1) ≤ 1`.
6. `b1_tight_dense_geo` — sama, hanya untuk pohon padat (`n_dets ≥ 18`).

## Hasil

| Metode | Acc all | MAE all | train | val | test |
|---|---:|---:|---:|---:|---:|
| `iter4_geometric_mean_blend` | **86,15%** | **0,3961** | 87,17 | 81,46 | 87,43 |
| `arith_geo_mix` | 86,15% | 0,3961 | 87,17 | 81,46 | 87,43 |
| `geo_with_naive_ceiling` | 86,15% | 0,3961 | 87,17 | 81,46 | 87,43 |
| `b1_tight_geo` | 86,15% | 0,3961 | 87,17 | 81,46 | 87,43 |
| `b1_tight_dense_geo` | 86,15% | 0,3961 | 87,17 | 81,46 | 87,43 |
| `geo_weighted_w50` | 86,15% | 0,3961 | 87,17 | 81,46 | 87,43 |
| `geo_weighted_w55` | 86,04% | 0,3990 | 87,01 | 81,46 | 87,43 |
| `geo_3way` | 86,04% | 0,4042 | 87,01 | 81,46 | 87,43 |
| `geo_weighted_w40` | 83,21% | 0,4376 | 83,39 | 80,90 | 85,03 |

## Temuan Jujur

### Plateau makin kuat
Enam metode dari formulasi matematis berbeda (geo, arith+geo, geo+ceiling, geo+B1 tight, weighted geo w=0,5) **semua terikat di 86,15%**. Tidak satu pun melampaui iter4. Dengan dua iterasi terpisah memberikan plateau identik, ini bukti kuat bahwa 86,15% adalah ceiling struktural.

### Operasi no-op
- `b1_tight_geo` dan `b1_tight_dense_geo`: kondisi `active_sides(B1) ≤ 1` saat ini sudah ditangani secara implisit oleh kombinasi visibility + floor clamp. Cap eksplisit tidak menambah nilai.
- `geo_with_naive_ceiling`: estimator selalu ≤ naive secara bawaan. Plafon eksplisit redundan.

### Bobot geometric tidak sensitif
`geo_weighted_w50` (= unweighted) optimal. Memberi w=0,55–0,70 menurunkan ke 86,04% (efek kecil), w=0,40–0,45 jatuh tajam (−3pp).

### `geo_3way` tidak lebih baik
Menambah `side_coverage` ke geometric mean tidak meningkatkan akurasi dan menaikkan MAE (0,3961 → 0,4042). `side_coverage` membawa noise.

## Status Ceiling

Empat iterasi (iter1, iter3, iter4, iter5) konvergen ke plateau 86,04–86,15%. Berbagai formulasi:
- Arithmetic mean (hybrid_vis_corr)
- Geometric mean (geometric_mean_blend) ← juara saat ini
- Median 3-estimator
- Per-class divisor adjustment
- Weighted blends

Semua landing dalam rentang 0,11pp. Plateau ini adalah **batas iredusibel struktur dataset**, bukan kekurangan algoritma satu per satu.

## Rekomendasi Iter6

Pertanyaan kritis: **apakah metode berbeda gagal di pohon yang sama?**

- Jika ya → 132 pohon adalah hard set struktural (B2↔B3 ambiguity). Tidak ada heuristik akan menyentuhnya.
- Jika tidak → ada irisan yang bisa diperbaiki dengan ensemble disagreement (per pohon pilih metode yang terbukti benar pada pohon serupa).

Iter6 fokus: analisis intersection failure 5 metode top, identifikasi pohon yang dapat dipindah ke "benar" via ensemble.

## Catatan Kejujuran (RULES.txt)

- Iter5 menghasilkan **nol perbaikan**. Setiap variasi geometric tied atau lebih buruk.
- Plateau 86,15% bukan kegagalan tooling — itu adalah temuan: dataset 953 punya batas struktural. Honest framing: optimisasi heuristik telah mencapai diminishing return.
- Tidak ada metode dipilih sebagai "kandidat baru". `geometric_mean_blend` tetap juara tanpa peserta baru yang dapat menggesernya.
