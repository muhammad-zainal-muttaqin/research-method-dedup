# Laporan GT Bunch Counting — Semua Pohon
**Tanggal:** 2026-04-23
**Dataset:** DAMIMAS + LONSUM (seluruh data GT yang tersedia)

---

## 1. Ringkasan Dataset

| Item | Nilai |
|------|-------|
| Total pohon diproses | **953** |
| Domain DAMIMAS | 854 |
| Domain LONSUM | 99 |
| Pohon 4-sisi | 908 |
| Pohon 8-sisi | 45 |
| Pohon dengan JSON (dedup akurat) | **953** |
| Pohon tanpa JSON (naive sum) | **0** |

---

## 2. Jumlah Tandan per Kelas (Seluruh Pohon)

> Pohon ber-JSON: hitungan **unik/dedup** (akurat).
> Pohon non-JSON: hitungan **naif** (tanpa dedup — estimasi overcounting ~79%).

| Kelas | JSON-Dedup (953 pohon) | Naive-Sum (0 pohon) | Total |
|-------|---:|---:|---:|
| B1 | 954 | 0 | 954 |
| B2 | 1,791 | 0 | 1,791 |
| B3 | 5,067 | 0 | 5,067 |
| B4 | 2,011 | 0 | 2,011 |
| **TOTAL** | **9,823** | **0** | **9,823** |

### Estimasi True Count untuk Pohon Non-JSON
Berdasarkan hasil JSON-05 (overcounting rate 78.8%), estimasi tandan unik sesungguhnya
untuk 0 pohon non-JSON:

| Kelas | Naive Count | Est. Unique (÷1.788) |
|-------|---:|---:|
| B1 | 0 | 0 |
| B2 | 0 | 0 |
| B3 | 0 | 0 |
| B4 | 0 | 0 |
| **TOTAL** | **0** | **0** |

---

## 3. Breakdown per Domain

### DAMIMAS (854 pohon)

| Kelas | Count | % |
|-------|------:|---:|
| B1 | 946 | 10.3% |
| B2 | 1,709 | 18.5% |
| B3 | 4,653 | 50.5% |
| B4 | 1,908 | 20.7% |
| **Total** | **9,216** | 100% |

- Pohon ber-JSON: 854 | Non-JSON: 0

### LONSUM (99 pohon)

| Kelas | Count | % |
|-------|------:|---:|
| B1 | 8 | 1.3% |
| B2 | 82 | 13.5% |
| B3 | 414 | 68.2% |
| B4 | 103 | 17.0% |
| **Total** | **607** | 100% |

- Pohon ber-JSON: 99 | Non-JSON: 0

---

## 4. Breakdown per Split

### Split: TRAIN (763 pohon)

| Kelas | Count |
|-------|------:|
| B1 | 773 |
| B2 | 1,441 |
| B3 | 4,055 |
| B4 | 1,626 |
| **Total** | **7,895** |

- Pohon ber-JSON: 763 | Non-JSON: 0

### Split: VAL (95 pohon)

| Kelas | Count |
|-------|------:|
| B1 | 91 |
| B2 | 153 |
| B3 | 519 |
| B4 | 203 |
| **Total** | **966** |

- Pohon ber-JSON: 95 | Non-JSON: 0

### Split: TEST (95 pohon)

| Kelas | Count |
|-------|------:|
| B1 | 90 |
| B2 | 197 |
| B3 | 493 |
| B4 | 182 |
| **Total** | **962** |

- Pohon ber-JSON: 95 | Non-JSON: 0

---

## 5. Catatan Metodologi

- **Sumber data:** Ground truth label (bukan prediksi model) — sesuai arahan dosen
- **JSON dedup:** 228 pohon sudah di-link manual antar sisi → hitungan tandan unik akurat
- **TXT naive:** 725 pohon dihitung langsung dari file label YOLO → setiap penampakan dihitung 1×
- **Overcounting rate** (dari JSON-05): naive sum rata-rata **78.8% lebih tinggi** dari count unik
- **Pohon 8-sisi (45 pohon):** data baru dengan 8 sudut foto — dihitung naive sum (belum ada JSON)
- File detail per pohon tersimpan di: `reports/full_gt_count/count_all_trees.csv`

---

## 6. File Output

| File | Isi |
|------|-----|
| `count_all_trees.csv` | 953 baris — detail per pohon |
| `summary_by_domain.csv` | Agregat DAMIMAS vs LONSUM |
| `summary_by_split.csv` | Agregat train / val / test |
| `summary.md` | Dokumen ini |
