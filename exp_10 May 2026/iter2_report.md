# Iterasi 2 — Analisis Kegagalan 133 Pohon

Dataset: 953 pohon (`Brand-New-Dataset-YOLO/json/`).
Metode acuan: `floor_clamped_hybrid` (Acc±1 = 86,04%, 133 gagal).
Skrip: `exp_10 May 2026/iter2_failure_analysis.py`.

## Frekuensi Kelas Pemicu (|err| > 1)

| Kelas | Total gagal | Over-pred | Under-pred | Mean signed err |
|---|---:|---:|---:|---:|
| B1 | 26 | 25 | 1 | +0,564 |
| B2 | 46 | 31 | 15 | +0,504 |
| **B3** | **92** | 58 | 34 | **+1,120** |
| **B4** | 29 | 6 | **23** | **−0,398** |

## Distribusi Arah

- 70 pohon: hanya over-pred (52,6%)
- 62 pohon: hanya under-pred (46,6%)
- 1 pohon: campur

Bias bidirectional — tidak ada divisor global tunggal yang dapat memperbaiki keduanya.

## Karakteristik Pohon Gagal

- `n_dets`: median 23, rata-rata 26 (populasi median sekitar 19) — kegagalan condong pada pohon padat.
- `abs_total_err`: 32 pohon meleset 2, 35 pohon meleset 3, 20 pohon meleset 4. Ekor tebal (≥10): 24 pohon — ini kasus terberat.
- Kombinasi pemicu paling umum: hanya B3 (53 pohon), hanya B4 (19), hanya B2 (15), B2+B3 (13), B1+B2+B3 (12).

## Temuan Utama

1. **B4 di-*under-predict* sistematis.** 23 dari 29 kegagalan B4 adalah under-pred. Mean signed error −0,398 → divisor B4 saat ini terlalu agresif untuk subset pohon padat.
2. **B1 dan B2 cenderung *over-predict*.** B1 hampir eksklusif over (25/26). Divisor B1 (1,986) tinggi tapi `visibility_count` masih murah hati pada B1 di pohon padat.
3. **B3 adalah penyebab utama kegagalan** (69% pohon). Bidirectional, mean +1,12 — campuran kasus over-confident pada deteksi pinggir frame dan kasus under-count saat satu tandan dilihat dari banyak sisi tanpa overlap visual yang jelas.
4. **Kegagalan ≠ pohon mudah salah tebak.** Distribusi `n_dets` failure menunjukkan pohon-pohon padat — bukan pohon dengan deteksi minim. Ini konsisten dengan B2↔B3 confusion sebagai *ceiling* iredusibel.

## Kesimpulan untuk Iter3

Tidak ada perbaikan universal melalui divisor tunggal — perlu koreksi **per-kelas dan terkondisi kepadatan**:

- **B4 floor lift**: untuk pohon padat (`n_dets` ≥ 20), naikkan estimasi B4 jika `max_per_side(B4)` mendekati prediksi atau ada deteksi B4 di banyak sisi.
- **B1 dan B2 trim**: pertimbangkan plafon berbasis konsensus per-sisi (jika deteksi terkonsentrasi di satu sisi saja, kemungkinan duplikasi ringan dan prediksi sudah cukup).
- **Hindari overfit**: setiap koreksi harus diuji tidak merusak Acc±1 dari 820 pohon yang sudah benar.

Iter3: implementasi koreksi prinsipil berbasis kepadatan + cross-side coverage signal, bukan boost/trim hard-coded.

## Catatan Kejujuran (RULES.txt)

- Analisis ini deskriptif. Tidak ada perubahan kode produksi.
- Pola yang ditemukan adalah agregat dari 133 sampel — sah sebagai sinyal arah, bukan sebagai parameter ad-hoc per-pohon.
- Iter3 harus berhati-hati: sederhana mengalikan B4 dengan 1,15 di seluruh dataset bukan koreksi prinsipil — itu *hack*. Yang benar: kondisi struktural yang membuat B4 under-pred, lalu fix kondisi itu.
