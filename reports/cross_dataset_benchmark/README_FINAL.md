# Cross-Dataset Generalization Benchmark: v9 vs v10 (FINAL)

## Executive Summary

Evaluasi pada **727 JSON files** (bukan 442 trees). Satu pohon bisa punya multiple JSON files dengan ground truth yang berbeda.

| Dataset | Files | v9 Acc ±1 | v10 Acc ±1 | Delta |
|---------|-------|-----------|------------|-------|
| json/ (22 Apr) | 228 | **97.37%** | 91.67% | v9 +5.70% |
| json_28 Apr | 477 | **92.66%** | 89.31% | v9 +3.35% |
| json_30 Apr | 725 | **89.24%** | 88.28% | v9 +0.97% |

**Key Insight**: v9 lebih baik di dataset yang di-tune, tapi **keunggulannya menyusut** saat dataset bertambah. Di dataset 725 files, gap hanya **0.97%**.

---

## Analisis

### v9 Strengths
- **Optimized untuk 228 file**: Decision tree thresholds sangat spesifik untuk karakteristik dataset ini
- **Regime overrides**: 4 kondisi spesifik menangani edge cases dengan baik
- **External params**: Membaca dari CSV yang di-grid-search

### v9 Weaknesses
- **Overfitting**: Performa turun drastis dari 97% → 89% saat dataset 3.2× lebih besar
- **Fragile thresholds**: Angka seperti `B4_yrange > 0.0945` sangat spesifik untuk 228 file

### v10 Strengths  
- **Consistent**: 91.67% → 88.28% (hanya turun 3.4% vs v9 turun 8.1%)
- **Self-contained**: Tidak perlu external CSV params
- **Simpler logic**: Decision tree lebih conservative, less overfit

### v10 Weaknesses
- **Not as sharp**: Tidak seoptimal v9 untuk dataset spesifik 228 file
- **Generic**: Fitted untuk generalisasi, bukan performa puncak

---

## Kesimpulan

**Pilih v9 jika:**
- Dataset tetap sama dengan yang di-tune (228 files)
- Butuh performa puncak 97%+
- Tidak peduli dengan generalisasi

**Pilih v10 jika:**
- Dataset akan bertambah/bervariasi
- Butuh robustness > performa puncak
- Deployment ke kondisi nyata yang diverse

**Gap menyusut**: +5.7% → +3.35% → +0.97% menunjukkan v9 kehilangan keunggulan saat data bertambah.
