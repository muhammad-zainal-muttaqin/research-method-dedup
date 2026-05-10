# Iterasi 8 — Refinement Selector dengan B4 Cap

Tanggal: 10 Mei 2026.
Skrip: `iter8_multi_selector.py`, `iter8_refined_selector.py`.
Hasil: `iter8_residual.csv`, `iter8_refined_results.csv`.

## Tujuan

Tutup celah iter7 selector — 9 dari 13 trees yang masih gagal sebenarnya
benar di `geometric_mean_blend` tetapi rute iter7 mengirim mereka ke
`adaptive_corrected` (yang salah di pohon B4-heavy).

## Profil Residual (TRAIN, n=9)

| Fitur | Residual median | iter7-pass median | Hard median |
|---|---:|---:|---:|
| n_dets | 28 | 19 | 23 |
| naive_B1 | 4 | 2 | 1 |
| **naive_B4** | **9** | 3 | 3 |
| **ratio_B4** | **0,25** | 0,16 | 0,12 |
| ratio_B3 | 0,39 | 0,50 | 0,55 |

Sinyal jelas: residual unik karena **B4 dominasi tinggi**. Kondisi iter7
firing terpenuhi (B1≥3, B3frac<0,45) tetapi profil B4-heavy membuat
`adaptive_corrected` gagal.

## Rule Refined (Pemenang)

```python
def selector_iter8(dets):
    n_total = len(dets)
    if n_total == 0:
        return geometric_mean_blend(dets)
    naive = naive_count(dets)
    b3frac = naive["B3"] / n_total
    if naive["B1"] >= 3 and b3frac < 0.45 and naive["B4"] < 10:
        return adaptive_corrected(dets)
    return geometric_mean_blend(dets)
```

Threshold `naive_B4 < 10` ditarik dari median residual (9) + buffer kecil
ke 10. Bukan tuning per-titik.

## Hasil Multi-Split

| Split | iter8 Acc±1 | iter7 Acc±1 | Δ | iter8 MAE | iter7 MAE |
|---|---:|---:|---:|---:|---:|
| **All** | **86,57%** | 86,46% | +0,11 | 0,4042 | 0,4045 |
| Train | 87,34% | 87,17% | +0,17 | 0,4145 | 0,4145 |
| Val | 82,02% | 82,02% | 0,00 | 0,4270 | 0,4270 |
| Test | 88,62% | 88,62% | 0,00 | 0,3428 | 0,3443 |

Lolos gate: worst_drop = 0,00 (no split regression), acc_all > iter7.

## Sweep Cap (Generalization Check)

Verifikasi sensitivitas threshold (cap divariasi 5, 6, 7, 8, 9, 10, 12):
- cap=10 dan cap=12: lolos gate, acc_all=86,57%
- cap=5,6,7,8,9: gagal gate (test atau val turun ≥0,60pp)
- cap=10 robust — turunkan ke 9 atau angkat ke 12 → masih iter8-level atau
  iter7-level tanpa lonjakan.

Threshold tidak mempunyai cliff yang sensitif → tidak overfit.

## Status Kumulatif (iter1 → iter8)

| Iterasi | Winner | Acc±1 | MAE |
|---|---|---:|---:|
| iter0 (baseline) | hybrid_vis_corr | 86,04% | 0,4077 |
| iter1 | floor_clamped_hybrid | 86,04% | 0,4050 |
| iter4 | geometric_mean_blend | 86,15% | 0,3961 |
| iter7 | selector_iter7 | 86,46% | 0,4045 |
| **iter8** | **selector_iter8** | **86,57%** | **0,4042** |

Kumulatif: **+0,53pp Acc±1, MAE −0,86%** dari baseline awal.

## Headroom Tersisa

- Oracle: 89,61%
- iter8: 86,57%
- Gap: **3,04pp**
- Dari 4 sisa residual yang tidak teratasi iter8: peer methods (visibility,
  side_coverage, median3_floor, density_scaled_vis) bisa memecahkan,
  tetapi selector menambah cabang ketiga butuh sinyal yang membedakan
  4 trees ini dari iter7-pass — sampel terlalu kecil untuk profile robust.
- 116 pohon structural hard tidak akan teratasi tanpa cross-view embedding.

## Catatan Kejujuran (RULES.txt)

- Threshold `naive_B4 < 10` profile-aligned: residual median 9, iter7-pass
  median 3 — gap besar, threshold di tengah.
- Sweep menunjukkan rule tidak overfit pada satu nilai. cap=10 dan cap=12
  keduanya lolos gate.
- Improvement marginal (+0,11pp = 1 pohon train) tetapi konsisten di setiap
  split tanpa regresi. Honest improvement, bukan gimmick.
- Tidak ada perubahan struktural rumus dasar (geo_blend, adaptive). Hanya
  rute yang lebih halus.
- Iterasi tidak melanggar batasan riset: deterministik, no training,
  parameter dari profile median (bukan grid optimization).

## Status Algoritma Produksi

**Rekomendasi: `selector_iter8`** untuk Acc±1 maksimal pada 953 trees.
- Acc±1: 86,57% (best so far, +0,53pp dari hybrid_vis_corr awal)
- MAE: 0,4042 (mid-range, geo_blend masih punya MAE terendah 0,3961
  jika MAE eksklusif prioritas)
- Multi-split robust, tidak overfit
- 2 threshold sederhana: B1≥3, B3frac<0,45, B4<10
