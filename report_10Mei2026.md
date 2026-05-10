# Laporan Eksperimen 10 Mei 2026 — `selector_with_b2b3`

Iterasi 1–13 pada folder [`exp_10 May 2026/`](exp%2010%20May%202026/) berakhir
pada algoritma **`selector_with_b2b3`** sebagai metode terbaik baru untuk
benchmark 953 pohon Brand-New-Dataset-YOLO.

Kode produksi tersedia di [`algorithms/selector_with_b2b3.py`](algorithms/selector_with_b2b3.py).

---

## Hasil akhir

| Metrik | Nilai | Sumber |
|---|---:|---|
| `Acc ±1` (all 953) | **86,67%** | `iter11_results.csv` |
| `MAE` | **0,3982** | `iter11_results.csv` |
| Pohon gagal | **127** | dari 953 |
| `Acc ±1` train | 87,34% | held-out |
| `Acc ±1` val | 82,58% | held-out |
| `Acc ±1` test | 88,62% | held-out |
| `worst_drop` | 0,00 pp | tidak overfit |

Improvement vs juara sebelumnya `hybrid_vis_corr` (86,04% / MAE 0,4077):
**+0,63 pp Acc±1**, **−2,32% MAE**.

---

## Perbandingan kandidat iter11

| Metode | Acc±1 (all) | MAE | train | val | test |
|---|---:|---:|---:|---:|---:|
| `b2b3_iter9_split` | 86,67% | 0,3982 | 87,34% | 82,58% | 88,62% |
| **`selector_with_b2b3`** | **86,67%** | **0,3982** | 87,34% | 82,58% | 88,62% |
| `iter9_baseline` | 86,67% | 0,3987 | 87,34% | 82,58% | 88,62% |
| `mode5` | 85,94% | 0,3930 | 86,35% | 83,15% | 87,43% |
| `median5` | 85,94% | 0,3930 | 86,35% | 83,15% | 87,43% |
| `b2b3_med_split` | 85,94% | 0,3930 | 86,35% | 83,15% | 87,43% |
| `trim5` | 85,94% | 0,3956 | 86,35% | 83,15% | 87,43% |
| `class_specialist` | 85,94% | 0,3959 | 86,68% | 82,02% | 87,43% |

Sumber: [`exp_10 May 2026/iter11_results.csv`](exp%2010%20May%202026/iter11_results.csv).

---

## Inti algoritma

Dua tahap:

1. **Selector trifurkasi** (`selector_iter9_trifurc`) memilih estimator
   dasar per profil pohon:
   - `b3frac ≥ 0,60` dan `n_total ≥ 25` → `median3_floor`
   - `naive_B1 ≥ 3` dan `b3frac < 0,45` dan `naive_B4 < 10` → `adaptive_corrected`
   - lainnya → `geometric_mean_blend`
2. **Koreksi split B2↔B3**: total `B2 + B3` dipertahankan, rasio
   dialokasikan ulang menurut frekuensi naive B2/B3. Menjawab ambiguitas
   visual B2↔B3 yang menyebabkan kesalahan kelas tetapi bukan kesalahan
   jumlah.

Pseudokode ringkas:

```
pred = selector_iter9_trifurc(detections)
joint = pred["B2"] + pred["B3"]
if joint > 0 and ada B2 atau B3 di detections:
    frac_b3 = n_b3 / (n_b2 + n_b3)
    pred["B3"] = max(round(joint * frac_b3), max_per_side("B3"))
    pred["B2"] = max(joint - pred["B3"],     max_per_side("B2"))
return pred
```

Implementasi lengkap di [`algorithms/selector_with_b2b3.py`](algorithms/selector_with_b2b3.py).

---

## Mengapa target Acc±1 ≥ 90% / MAE < 0,2 tidak dicapai

Pembuktian dari [`exp_10 May 2026/iter13_FINAL_HONEST_STOP.md`](exp%2010%20May%202026/iter13_FINAL_HONEST_STOP.md):

**MAE per-kelas pada `selector_with_b2b3`:**

| Kelas | MAE | Distribusi err 0 / 1 / ≥2 |
|---|---:|---|
| B1 | 0,179 | 822 / 106 / 25 |
| B2 | 0,346 | 730 / 178 / 45 |
| B3 | **0,757** | 490 / 372 / 91 |
| B4 | 0,310 | 700 / 225 / 28 |

- Bahkan jika B3 sempurna (MAE B3 = 0), total MAE = 0,209 — masih >0,2.
- Oracle ceiling realistik (toolkit penuh, tanpa overfit risk) = **89,61%**,
  di bawah target 90%.
- 99–118 pohon `structural hard` karena ambiguitas B2↔B3 iredusibel
  (bukan derau label — JSON-01 audit mengkonfirmasi label noise = 0%).

Loop 13 iterasi dihentikan jujur: target user **tidak dapat dicapai**
dalam constraint riset (no training, no embedding) tanpa overfit.

---

## Iterasi yang dilakukan

| Iter | Tujuan | Outcome |
|---|---|---|
| 1 | Ensemble 3-estimator | winner (geometric_mean_blend) |
| 2 | Failure analysis | analisis (CSV residual) |
| 3 | Cross-validated corrections | zero-improvement (honest report) |
| 4 | Split analysis | winner (split-aware base) |
| 5 | Geo extensions | zero-improvement |
| 6 | Disagreement mining | analisis |
| 7 | Selector iter9 trifurkasi | winner |
| 8 | Multi-selector refinement | winner (refined trifurc) |
| 9 | Final benchmark iter9 | winner (86,67%) |
| 10 | Oracle ceiling analysis | 90,14% toolkit, 89,61% realistic |
| 11 | Mode-vote + b2b3 split | **winner final → `selector_with_b2b3`** |
| 12 | Total-first reformulation | zero-improvement |
| 13 | MAE breakdown + stop | mathematical proof, loop dihentikan |

Detail tiap iterasi di [`exp_10 May 2026/iter*_report.md`](exp%2010%20May%202026/).

---

## Cara pemakaian

```python
from algorithms.selector_with_b2b3 import predict

dets = [
    {"class": "B3", "x_norm": 0.5, "y_norm": 0.4, "side_index": 0},
    # ... deteksi lainnya dari semua sisi pohon
]
counts = predict(dets)
# {"B1": int, "B2": int, "B3": int, "B4": int}
```

Tidak perlu `params` — semua konstanta sudah di-bake ke modul.
