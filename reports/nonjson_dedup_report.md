# Laporan Dedup Multi-View — Pohon Non-JSON (717 pohon)

**Tanggal:** 2026-04-23  
**Dataset:** DAMIMAS + LONSUM, 953 pohon total  
**Fokus:** 717 pohon tanpa JSON ground truth (hanya YOLO TXT labels)  
**Referensi validasi:** 228 pohon ber-JSON (ground truth tersedia)

---

## 1. Konteks: Mengapa Ada Pohon Non-JSON?

| Item | Jumlah |
|------|--------|
| Total pohon | 953 |
| Ber-JSON (dedup akurat, manual linking) | 228 |
| Non-JSON (hanya TXT labels) | **717** |
| Pohon 0 deteksi di TXT | 8 |
| Pohon non-JSON yang diproses | **717** |

File TXT di `dataset/labels/{train,val,test}/` berisi **prediksi model YOLO** (format: `class_id cx cy w h`), bukan anotasi manual seperti JSON. Ini penting karena:

- TXT memiliki **noise koordinat** dan **noise klasifikasi** (B2↔B3 sering tertukar)
- Tidak ada informasi `side_index` eksplisit — dideduksi dari nama file (`*_1.txt` = sisi_1, dst.)
- Tidak ada `_confirmedLinks` untuk belajar threshold

---

## 2. Validasi Akurasi pada 228 Pohon JSON (Benchmark)

Sebelum menerapkan ke non-JSON, setiap metode dievaluasi pada pohon ber-JSON:

| Method | Mean MAE | Acc ±1 | Mean Total Err | Score |
|--------|---------:|-------:|---------------:|------:|
| **visibility** | 0.2719 | **92.11%** | 1.09 | 89.39 |
| **corrected** | 0.2851 | 90.79% | 1.14 | 87.94 |
| hungarian_match | 1.0976 | 18.86% | 4.39 | 7.88 |
| cascade_match | 1.7730 | 4.39% | 7.09 | -13.34 |
| learned_graph | 1.8202 | 4.39% | 7.28 | -13.82 |
| feature_cluster | 1.8728 | 3.51% | 7.49 | -15.22 |
| naive | 2.1294 | 2.63% | 8.52 | -18.66 |

**Temuan kunci:**
- `visibility` dan `corrected` adalah **satu-satunya metode yang akurat** (>90%)
- Graph matching, cascade, clustering **gagal total** jika dipakai sendiri (<20%)
- Ceiling heuristic bbox ≈ 92% — untuk tembus lebih tinggi butuh image embedding

---

## 3. Hasil Dedup pada 717 Pohon Non-JSON

Setelah rerun `scripts/dedup_all_trees_final.py`, semua metode v5 ikut dihitung pada data TXT non-JSON.

### 3.1 Total Count per Metode

| Method | B1 | B2 | B3 | B4 | **Total** | Rasio vs Naive |
|--------|----|----|----|----|----------:|---------------:|
| naive | 1,618 | 2,974 | 6,417 | 2,656 | **13,665** | 100.0% |
| class_aware_vis | 917 | 1,841 | 3,974 | 1,478 | **8,210** | 60.4% |
| adaptive_corrected | 855 | 1,760 | 3,724 | 1,671 | **8,010** | 57.4% |
| adaptive_visibility | 956 | 1,756 | 3,728 | 1,559 | **7,999** | 57.5% |
| best_class_aware_grid | 919 | 1,744 | 3,718 | 1,481 | **7,862** | 58.0% |
| best_ensemble_grid | 880 | 1,732 | 3,653 | 1,565 | **7,830** | 56.6% |
| corrected | 917 | 1,691 | 3,573 | 1,598 | **7,779** | 57.4% |
| hybrid_vis_corr | 919 | 1,668 | 3,504 | 1,505 | **7,596** | 56.0% |
| side_coverage | 921 | 1,665 | 3,465 | 1,500 | **7,551** | 56.1% |
| density_scaled_vis | 919 | 1,657 | 3,486 | 1,484 | **7,546** | 55.8% |
| best_visibility_grid | 919 | 1,655 | 3,486 | 1,481 | **7,541** | 55.9% |
| visibility | 919 | 1,650 | 3,458 | 1,483 | **7,510** | 55.7% |
| ordinal_prior | 919 | 1,674 | 3,434 | 1,483 | **7,510** | 55.7% |
| naive_mean_ensemble | 917 | 1,646 | 3,394 | 1,483 | **7,440** | 55.3% |
| best_relaxed_grid | 620 | 818 | 879 | 748 | **3,065** | 26.0% |
| relaxed_match | 512 | 674 | 720 | 623 | **2,529** | 21.6% |

### 3.2 Rasio Dedup per Pohon (Dedup Total / Naive Total)

| Method | Mean Ratio | Median Ratio | Std Dev |
|--------|-----------:|-------------:|--------:|
| hybrid_vis_corr | 0.5601 | 0.5556 | 0.0510 |
| best_visibility_grid | 0.5587 | 0.5556 | 0.0501 |
| side_coverage | 0.5615 | 0.5556 | 0.0553 |
| density_scaled_vis | 0.5582 | 0.5500 | 0.0509 |
| visibility | 0.5570 | 0.5500 | 0.0509 |
| ordinal_prior | 0.5570 | 0.5500 | 0.0509 |
| best_ensemble_grid | 0.5660 | 0.5714 | 0.0514 |
| naive_mean_ensemble | 0.5525 | 0.5455 | 0.0521 |
| adaptive_corrected | 0.5736 | 0.5769 | 0.0574 |
| corrected | 0.5741 | 0.5714 | 0.0480 |
| adaptive_visibility | 0.5754 | 0.5833 | 0.0623 |
| best_class_aware_grid | 0.5799 | 0.5714 | 0.0492 |
| class_aware_vis | 0.6042 | 0.6000 | 0.0497 |
| best_relaxed_grid | 0.2604 | 0.2143 | 0.1578 |
| relaxed_match | 0.2156 | 0.1875 | 0.1208 |

### 3.3 Ranking Non-JSON

Karena non-JSON tidak punya ground truth, ranking paling berguna adalah kedekatan ke rasio dedup terverifikasi dari JSON, yaitu sekitar `0.56`.

| Rank | Method | Mean Ratio | Jarak ke 0.56 |
|-----:|--------|-----------:|--------------:|
| 1 | hybrid_vis_corr | 0.5601 | 0.0001 |
| 2 | best_visibility_grid | 0.5587 | 0.0013 |
| 3 | side_coverage | 0.5615 | 0.0015 |
| 4 | density_scaled_vis | 0.5582 | 0.0018 |
| 5 | visibility | 0.5570 | 0.0030 |
| 5 | ordinal_prior | 0.5570 | 0.0030 |
| 7 | best_ensemble_grid | 0.5660 | 0.0060 |
| 8 | naive_mean_ensemble | 0.5525 | 0.0075 |
| 9 | adaptive_corrected | 0.5736 | 0.0136 |
| 10 | corrected | 0.5741 | 0.0141 |
| 11 | adaptive_visibility | 0.5754 | 0.0154 |
| 12 | best_class_aware_grid | 0.5799 | 0.0199 |
| 13 | class_aware_vis | 0.6042 | 0.0442 |
| 14 | best_relaxed_grid | 0.2604 | 0.2996 |
| 15 | relaxed_match | 0.2156 | 0.3444 |

---

## 4. Analisis & Interpretasi

### 4.1 Verifikasi dengan JSON-05
Dari laporan `json_05/`, overcounting rate rata-rata = **78.8%**. Artinya:

```
unique_count ≈ naive_count / 1.788 ≈ naive_count × 0.559
```

Dengan kata lain, **rasio dedup yang benar ≈ 56%**.

| Metode | Rasio | Verdict |
|--------|------:|---------|
| corrected | 57.4% | **Sangat masuk akal** |
| visibility | 55.7% | **Sangat masuk akal** |
| hungarian | 42.6% | Undercount ringan |
| learned_graph | 23.9% | **Undercount parah** |
| cascade | 22.5% | **Undercount parah** |
| feature_cluster | 20.6% | **Undercount parah** |

### 4.2 Mengapa Graph/Cascade/Cluster Undercount Parah?

1. **Threshold v3 dipelajari dari `_confirmedLinks` di JSON** — bbox manual yang sangat presisi (tol_y ≈ 0.11, tol_area ≈ 0.09)
2. **TXT labels adalah prediksi model** — koordinat noisy, klasifikasi sering salah (terutama B2↔B3)
3. Matching dengan threshold ketat pada data noisy → menggabungkan terlalu banyak deteksi → undercount drastis
4. DBSCAN clustering juga over-merge karena tidak ada side-adjacency constraint

### 4.3 Mengapa Corrected & Visibility Berhasil?

- **corrected**: Hanya membagi naive count dengan faktor overcount per kelas (dari verifikasi JSON-05). Tidak bergantung pada matching quality.
- **visibility**: Menambahkan downweight berdasarkan posisi horizontal (`cx`), yang menangkap bias visibilitas antar-sisi. Cukup robust terhadap noise koordinat.

---

## 5. Perbandingan Per-Kelas (Non-JSON)

| Kelas | Naive | Corrected | Visibility | Expected (÷1.788) |
|-------|------:|----------:|-----------:|------------------:|
| B1 | 1,618 | 917 | 919 | 905 |
| B2 | 2,974 | 1,691 | 1,650 | 1,663 |
| B3 | 6,417 | 3,573 | 3,458 | 3,588 |
| B4 | 2,656 | 1,598 | 1,483 | 1,485 |

- **B1 & B4**: corrected & visibility sangat dekat dengan estimasi expected
- **B2 & B3**: visibility sedikit lebih agresif (lebih rendah) dibanding corrected — sesuai ekspektasi karena B2↔B3 paling ambigu

---

## 6. Rekomendasi

### Untuk Produksi / Inference Pipeline

| Skenario | Rekomendasi |
|----------|-------------|
| **Pohon non-JSON (717)** | Gunakan **`hybrid_vis_corr`**, `best_visibility_grid`, `side_coverage`, `density_scaled_vis`, atau `visibility`. Kelimanya paling dekat ke rasio target ~56% |
| **Pohon ber-JSON (228)** | Gunakan **`visibility`** (akurasi ±1 = 92.1%) atau `corrected` (90.8%) |
| **Hindari** | `best_relaxed_grid` dan `relaxed_match` untuk TXT labels — undercount parah |

### Untuk Research Selanjutnya (Algorithmic Only)

- **Tidak usah** tuning heuristic bbox lagi — ceiling ≈ 92%
- **Langsung ke** geometric algorithmic methods untuk tembus >95%:
  - Multi-camera geometry (calibration, epipolar constraints, 3D triangulation)
  - Topological matching dengan relaxed constraints
  - Statistical ensemble refinement
- **JANGAN** gunakan embedding-based matching — requires training, overfitting risk on 228 samples

---

## 7. File Output

| File | Lokasi | Isi |
|------|--------|-----|
| `all_trees_dedup_counts.csv` | `reports/dedup_all_trees_final/` | 953 baris — semua metode untuk semua pohon |
| `json_228_accuracy.csv` | `reports/dedup_all_trees_final/` | Validasi akurasi per metode pada 228 pohon JSON |
| `nonjson_725_counts.csv` | `reports/dedup_all_trees_final/` | 717 baris — per pohon non-JSON |
| `nonjson_725_summary.csv` | `reports/dedup_all_trees_final/` | Agregat total count per metode |
| `nonjson_725_ratios.csv` | `reports/dedup_all_trees_final/` | Rasio dedup per metode |
| `nonjson_dedup_report.md` | `reports/` | **Dokumen ini** |

---

*Report ini disintesis dari `scripts/dedup_all_trees_final.py` yang menjalankan metode v1/v2/v3/v5 pada seluruh dataset (228 JSON + 717 non-JSON) menggunakan data JSON untuk validasi dan TXT untuk prediksi pada pohon non-JSON.*
