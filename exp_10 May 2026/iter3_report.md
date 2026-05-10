# Iterasi 3 — Koreksi Per-Kelas Tervalidasi (Held-Out)

Tanggal: 10 Mei 2026.
Dataset: 953 pohon. Split: train=608, val=178, test=167.
Skrip: `iter3_cv_corrections.py`. Hasil: `iter3_cv_results.csv`.

## Tujuan

Uji koreksi per-kelas turunan iter2 dengan validasi held-out, bukan agregat tunggal.
Mencegah overfit pada subset 133 pohon gagal.

## Hasil Lengkap (Semua Split)

| Metode | Acc all | Acc train | Acc val | Acc test |
|---|---:|---:|---:|---:|
| `baseline_floor_clamped_hybrid` | **86,04%** | 87,01% | 81,46% | 87,43% |
| `iter1_baseline_hybrid_vis_corr` | 86,04% | 87,01% | 81,46% | 87,43% |
| `b1_concentration_trim` | 86,04% | 87,01% | 81,46% | 87,43% |
| `b4_lift_d160` | 84,68% | 85,86% | 79,21% | 86,23% |
| `adaptive_b4_lift` (d=1,50) | 82,69% | 84,21% | 77,53% | 82,63% |
| `combined_corrections` | 82,69% | 84,21% | 77,53% | 82,63% |
| `b4_lift_d145` | 75,55% | 77,80% | 67,98% | 75,45% |
| `b4_lift_d135` | 71,77% | 75,33% | 60,67% | 70,66% |

## Temuan Jujur

### Koreksi B4 lift GAGAL
Setiap variasi divisor B4 (1,35–1,60) **menurunkan** akurasi di tiga split sekaligus:
- d=1,60 (paling konservatif): turun 1,36pp di all (86,04 → 84,68).
- d=1,50 (baseline iter2 saran): turun 3,35pp di all.
- d=1,35 (paling agresif): turun 14,27pp di all.

**Ini bukan overfit — ini intervensi salah arah.** Sinyal agregat iter2 "B4 mean signed err = −0,398" benar secara statistik untuk 29 pohon yang gagal di B4, **tetapi** menerapkan lift ke seluruh dataset merusak ratusan pohon yang B4-nya sudah benar.

### `b1_concentration_trim` no-op
Kondisi `active_sides(B1) ≤ 1` jarang terpicu pada pohon yang gagal. Hasil identik baseline. Tidak merugikan, tidak menguntungkan.

### Plateau 86,04% bertahan
Tidak ada metode yang melampaui baseline `floor_clamped_hybrid` di iter3.

### Gap train vs val mencolok
- Train: 87,01%
- Val: **81,46%**
- Test: 87,43%
- All: 86,04%

Selisih val ke train = **5,55 poin persen**. Test mirip train. Ini sinyal bahwa **split val mengandung pohon yang lebih sulit** (atau lebih banyak kasus B2↔B3 ambigu). Bukan distribusi acak.

## Pelajaran (RULES.txt)

1. **Sinyal agregat ≠ resep koreksi per-pohon.** Jika B4 mean error −0,398, itu tidak berarti "naikkan B4 di semua tree". Dataset dengan distribusi 23 under, 6 over berarti naikkan B4 → 6 yang benar jadi salah, 23 yang salah mungkin jadi benar. Net loss bila yang benar jauh lebih banyak (897/953 = 94% pohon B4-nya tepat).

2. **Validasi held-out wajib.** Tanpa split train/val/test, b4_lift d=1,60 mungkin terlihat "tied" dengan baseline pada tuning. Tetapi tiga split menunjukkan turun konsisten — sinyal nyata bahwa intervensi merugikan generalisasi.

3. **Per RULES.txt: tidak commit perbaikan yang gagal validasi.** Iter3 menghasilkan **nol perbaikan**. Laporan ini adalah produk iter3 — sama berharganya dengan perbaikan algoritmik karena menutup avenue yang menjanjikan secara semu.

## Rekomendasi Iter4

Dua arah investigasi:

1. **Karakterisasi split val.** Mengapa val 5,55pp lebih rendah? Distribusi kelas? Jumlah deteksi? Variasi varietas (DAMIMAS vs LONSUM)? Ini mengungkap apakah plateau 86,04% bersifat struktural atau distribusi.

2. **Cari koreksi yang generalisir lintas split.** Setiap kandidat baru harus minimal:
   - Tidak menurunkan akurasi train, val, atau test secara individual.
   - Naik di setidaknya dua split untuk dianggap perbaikan.
   - Dipertimbangkan hanya jika penurunan di split manapun ≤ 0,5pp (toleransi noise).

## Status Algoritma Produksi

`floor_clamped_hybrid` tetap kandidat terbaik. Acc±1 = 86,04% pada 953 pohon, MAE 0,4050. Stabil lintas split (train 87,01%, val 81,46%, test 87,43%).
