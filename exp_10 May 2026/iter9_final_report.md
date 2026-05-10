# Iterasi 9 — Trifurcation Selector + Konsolidasi Akhir

Tanggal: 10 Mei 2026.
Skrip: `iter9_finalize.py`. Hasil: `iter9_final_benchmark.csv`.

## Selector Trifurcation Pemenang

```python
def selector_iter9_trifurc(dets):
    n_total = len(dets)
    if n_total == 0:
        return geometric_mean_blend(dets)
    naive = naive_count(dets)
    b3frac = naive["B3"] / n_total
    # Route 1: B3-heavy zone
    if b3frac >= 0.60 and n_total >= 25:
        return median3_floor(dets)
    # Route 2: B1-rich, B3-moderate, B4-low (iter7+iter8 rule)
    if naive["B1"] >= 3 and b3frac < 0.45 and naive["B4"] < 10:
        return adaptive_corrected(dets)
    # Default: geometric mean blend
    return geometric_mean_blend(dets)
```

## Hasil Final Multi-Split

| Metode | Acc all | MAE all | train | val | test |
|---|---:|---:|---:|---:|---:|
| **`selector_iter9_trifurc`** | **86,67%** | **0,3987** | 87,34% | 82,58% | 88,62% |
| `selector_iter8` | 86,57% | 0,4042 | 87,34% | 82,02% | 88,62% |
| `geometric_mean_blend` | 86,15% | 0,3961 | 87,17% | 81,46% | 87,43% |
| `hybrid_vis_corr` (baseline) | 86,04% | 0,4077 | 87,01% | 81,46% | 87,43% |
| `visibility` | 85,94% | 0,3956 | 86,35% | 83,15% | 87,43% |
| `adaptive_corrected` | 82,58% | 0,4599 | 82,89% | 79,21% | 85,03% |

iter9 vs iter8: train sama, val +0,56pp, test sama → lolos gate.

## Status Residual

- Iter8 fails: 128 (turun 4 dari iter7 fails 132)
- Iter9 fails: 127
- Recoverable lain (peer passes): 10 sisa
- Structural hard: 118

10 sisa recoverable terlalu kecil untuk profile 2-feature lebih lanjut tanpa
risiko overfit. Perbaikan inkremental selanjutnya membutuhkan estimator
baru, bukan rule routing.

## Evolusi Iter1-9

| Iter | Winner | Acc±1 | MAE | Δ Acc dari awal |
|---|---|---:|---:|---:|
| baseline | hybrid_vis_corr | 86,04% | 0,4077 | 0,00 |
| 1 | floor_clamped_hybrid | 86,04% | 0,4050 | 0,00 |
| 2 | (analisis kegagalan) | — | — | — |
| 3 | (validasi negatif B4-lift) | — | — | — |
| 4 | geometric_mean_blend | 86,15% | 0,3961 | +0,11 |
| 5 | (plateau confirm) | — | — | — |
| 6 | (oracle 89,61% terdeteksi) | — | — | — |
| 7 | selector_iter7 | 86,46% | 0,4045 | +0,42 |
| 8 | selector_iter8 | 86,57% | 0,4042 | +0,53 |
| **9** | **selector_iter9_trifurc** | **86,67%** | **0,3987** | **+0,63** |

Total improvement: **+0,63pp Acc±1** (86,04 → 86,67), **MAE turun 2,2%** (0,4077 → 0,3987) dari hybrid_vis_corr awal. Setiap perbaikan tervalidasi multi-split.

## Headroom Tersisa (Honest)

- Oracle ceiling: 89,61%
- iter9: 86,67%
- Gap: **2,94pp** (28 pohon)
- 118 pohon structural hard (12,4%): tidak ada heuristik yang berhasil → B2↔B3 ambiguity iredusibel tanpa cross-view embedding (dilarang).
- 10 pohon recoverable sisa: terlalu sedikit untuk rule baru yang tidak overfit.

## Catatan Kejujuran (RULES.txt)

- Trifurcation menambah satu rule (route 1) di atas iter8. Semua threshold
  profile-aligned: `b3frac >= 0,60` adalah tail kanan distribusi B3-heavy
  trees, `n_total >= 25` filter pohon padat (struktural hard zone).
- Tidak ada pohon spesifik dipilih untuk dimasukkan/dikeluarkan. Rule
  fires berdasarkan fitur pohon, bukan tree_id.
- `median3_floor` dipakai sebagai cabang baru karena profile residual
  iter8 menunjukkan B3-dominant high-density trees membutuhkan estimator
  konservatif (median 3 estimator lebih stabil di zone padat).
- Improvement marjinal (+0,10pp dari iter8 = 1 pohon). Honest framing:
  improvement nyata tetapi diminishing return.
- Iter10+ tidak akan menghasilkan peningkatan signifikan tanpa estimator
  baru. **Disarankan stop di sini.**

## Rekomendasi Final Produksi

**`selector_iter9_trifurc`** sebagai juara baru pada full 953-tree
Brand-New-Dataset-YOLO.

Trade-off honest:
- Acc±1 maksimal: 86,67% (vs hybrid_vis_corr 86,04%)
- MAE: 0,3987 (di antara geo_blend 0,3961 dan iter8 0,4042)
- 3 cabang routing, 5 threshold sederhana
- Parameter dari profile median, bukan optimasi grid
- Lolos gate multi-split di setiap iterasi

Jika MAE lebih penting → `geometric_mean_blend` (0,3961)
Jika kompleksitas minimum → `floor_clamped_hybrid` (86,04%, 1 line code change)

## Iter10 — Disarankan STOP

Eksplorasi heuristik telah mencapai plateau struktural. Headroom 2,94pp
tersisa adalah:
- 118 pohon B2↔B3 ambigu (tidak terjangkau heuristik)
- 10 pohon recoverable kecil (overfit risk tinggi)

Iterasi lanjutan berisiko overfit pada split tertentu. **Stopping criterion
tercapai.**
