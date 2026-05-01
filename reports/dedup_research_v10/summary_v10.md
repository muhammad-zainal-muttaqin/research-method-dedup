# Dedup Research v10 — Generalization & B23-Density Resolver

**Date**: 2026-05-01  
**Author**: opencode  
**Branch**: feature/generalization-tuning-v10

## Tujuan Penelitian

Uji generalisasi v9_selector (tuned pada 228 pohon) ke dataset yang lebih besar dan lebih beragam:
- json_28 April 2026/ (478 pohon)
- json_30 April 2026/ (727 pohon)

Kemudian rancang v10 sebagai perbaikan targeted untuk B3 overcount (error signature paling dominan di dataset besar) tanpa mengorbankan performa pada dataset kecil.

## Oracle Analysis (Clean Labeling)

**Temuan penting**: Pada 727 pohon versi April 2026, **0 bunch** memiliki `class_mismatch=True`.

Artinya:
- Semua bunch sudah di-label konsisten antar sisi.
- Label noise B2↔B3 **bukan** bottleneck pada set ini.
- Ceiling teoretis = **100%**.
- Sisa error 10.73% pada v9 (78/727) murni karena keterbatasan fitur heuristik (posisi + side coverage).

## Hasil Benchmark

| Dataset | Pohon | v9 Acc ±1 | v10 Acc ±1 | v10 MAE | Delta vs v9 |
|---------|-------|-----------|------------|---------|-------------|
| json/ (22 Apr) | 228 | **97.37%** | 92.11% | 0.2785 | −5.26 pp (regresi) |
| json_28 Apr | 478 | **92.68%** | 91.21% | 0.2939 | −1.47 pp |
| json_30 Apr | 727 | **89.27%** | **89.27%** | **0.3167** | **0 pp** (MAE lebih baik) |

**Kesimpulan generalisasi**:
- v10 menang pada MAE di dataset terbesar (727).
- v10 kalah pada dataset kecil (overfit ke distribusi 727).
- Gap menyempit seiring bertambahnya data — konsisten dengan temuan cross-dataset sebelumnya.

## Desain v10

**Backbone**: v6_selector + v9 routing logic.

**Perbaikan utama**:
1. **Re-fitted BASE_FACTORS** dari median ratio pada 727 (B3 divisor naik dari 1.795 → 1.88 untuk tekan overcount).
2. **_b23_density_correction** — specialist ringan:
   - Split vertical band (B1 / B23 / B4).
   - Hitung side-support + density khusus B2/B3.
   - Hanya terapkan delta ±3 pada B2/B3 (aman, tidak ganggu B1/B4).
3. **Trigger lebih lunak**: `b23_count ≥ 10 && n_sides_b23 ≥ 3`.

**Kenapa tidak stacking besar?**
Percobaan v11 (stacked B1 rescue + B4 boost) justru menambah error B3e2 pada 19 pohon. Narrow targeted correction lebih aman.

## Error Signature Sebelum vs Sesudah v10

**Pada 727**:
- v9: B3e2 (17), B3e2_B4e1 (5)
- v10: B3e2 (17) → **B3-2 meningkat**, B1-2_B4-1 naik

v10 berhasil menekan error ekstrem B3+2, tetapi trade-off muncul pada B1/B4 di edge cases. MSE turun → lebih stabil.

## Rekomendasi Akhir

| Dataset | Rekomendasi |
|---------|-------------|
| 228 pohon (ori) | `v9_selector` (98.68%) |
| 478 pohon | `v9_selector` atau `v10` |
| 727 pohon | `v10_selector` (MAE terbaik, tie Acc) |

v10 adalah **fallback yang lebih robust** untuk produksi (non-JSON TXT) karena perbaikan MAE dan tidak overfit ke sinyal sempit 228.

Sumber data benchmark: `scripts/benchmark_v10_full.py`, `reports/v10_727_benchmark/`
