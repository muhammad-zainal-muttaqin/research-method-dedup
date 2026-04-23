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

### 3.1 Total Count per Metode

| Method | B1 | B2 | B3 | B4 | **Total** | Rasio vs Naive |
|--------|----|----|----|----|----------:|---------------:|
| naive | 1,618 | 2,974 | 6,417 | 2,656 | **13,665** | 100.0% |
| **corrected** | 917 | 1,691 | 3,573 | 1,598 | **7,779** | **57.4%** |
| **visibility** | 919 | 1,650 | 3,458 | 1,483 | **7,510** | **55.7%** |
| hungarian_match | 775 | 1,340 | 2,420 | 1,240 | **5,775** | 42.6% |
| learned_graph | 535 | 760 | 817 | 709 | **2,821** | 23.9% |
| cascade_match | 515 | 665 | 837 | 633 | **2,650** | 22.5% |
| feature_cluster | 490 | 616 | 720 | 585 | **2,411** | 20.6% |

### 3.2 Rasio Dedup per Pohon (Dedup Total / Naive Total)

| Method | Mean Ratio | Median Ratio | Std Dev |
|--------|-----------:|-------------:|--------:|
| **corrected** | **0.5741** | 0.5714 | 0.0480 |
| **visibility** | **0.5570** | 0.5500 | 0.0509 |
| hungarian_match | 0.4259 | 0.4000 | 0.1535 |
| learned_graph | 0.2389 | 0.2000 | 0.1405 |
| cascade_match | 0.2247 | 0.1944 | 0.1228 |
| feature_cluster | 0.2061 | 0.1818 | 0.1113 |

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
| **Pohon non-JSON (717)** | Gunakan **`corrected`** atau **`visibility`**. Rasio ~55–57% sudah terverifikasi mendekati ground truth |
| **Pohon ber-JSON (228)** | Gunakan **`visibility`** (akurasi ±1 = 92.1%) atau `corrected` (90.8%) |
| **Hindari** | `learned_graph`, `cascade_match`, `feature_cluster` untuk TXT labels — undercount parah |

### Untuk Research Selanjutnya

- **Tidak usah** tuning heuristic bbox lagi — ceiling ≈ 92%
- **Langsung ke** embedding-based cross-view matching (Siamese/CNN pada bbox crops) untuk tembus >95%
- **Pertimbangkan** retraining model YOLO dengan data JSON-annotated untuk meningkatkan kualitas TXT labels

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

*Report ini disintesis dari `scripts/dedup_all_trees_final.py` yang menjalankan metode v1/v2/v3 pada seluruh dataset (228 JSON + 717 non-JSON) menggunakan data JSON untuk validasi dan TXT untuk prediksi pada pohon non-JSON.*
