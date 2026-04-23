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
| Pohon dengan JSON (dedup akurat) | **228** |
| Pohon tanpa JSON (naive sum) | **725** |

---

## 2. Jumlah Tandan per Kelas (Seluruh Pohon)

> Pohon ber-JSON: hitungan **unik/dedup** (akurat).
> Pohon non-JSON: hitungan **naif** (tanpa dedup — estimasi overcounting ~79%).

| Kelas | JSON-Dedup (228 pohon) | Naive-Sum (725 pohon) | Total |
|-------|---:|---:|---:|
| B1 | 291 | 1,618 | 1,909 |
| B2 | 532 | 2,974 | 3,506 |
| B3 | 1,144 | 6,417 | 7,561 |
| B4 | 499 | 2,656 | 3,155 |
| **TOTAL** | **2,466** | **13,665** | **16,131** |

### Estimasi True Count untuk Pohon Non-JSON
Berdasarkan hasil JSON-05 (overcounting rate 78.8%), estimasi tandan unik sesungguhnya
untuk 725 pohon non-JSON:

| Kelas | Naive Count | Est. Unique (÷1.788) |
|-------|---:|---:|
| B1 | 1,618 | 904 |
| B2 | 2,974 | 1,663 |
| B3 | 6,417 | 3,588 |
| B4 | 2,656 | 1,485 |
| **TOTAL** | **13,665** | **7,642** |

---

## 3. Breakdown per Domain

### DAMIMAS (854 pohon)

| Kelas | Count | % |
|-------|------:|---:|
| B1 | 1,892 | 12.5% |
| B2 | 3,354 | 22.2% |
| B3 | 6,882 | 45.5% |
| B4 | 2,981 | 19.7% |
| **Total** | **15,109** | 100% |

- Pohon ber-JSON: 228 | Non-JSON: 626

### LONSUM (99 pohon)

| Kelas | Count | % |
|-------|------:|---:|
| B1 | 17 | 1.7% |
| B2 | 152 | 14.9% |
| B3 | 679 | 66.4% |
| B4 | 174 | 17.0% |
| **Total** | **1,022** | 100% |

- Pohon ber-JSON: 0 | Non-JSON: 99

---

## 4. Breakdown per Split

### Split: TRAIN (666 pohon)

| Kelas | Count |
|-------|------:|
| B1 | 1,358 |
| B2 | 2,476 |
| B3 | 5,333 |
| B4 | 2,149 |
| **Total** | **11,316** |

- Pohon ber-JSON: 161 | Non-JSON: 505

### Split: VAL (144 pohon)

| Kelas | Count |
|-------|------:|
| B1 | 285 |
| B2 | 566 |
| B3 | 1,131 |
| B4 | 482 |
| **Total** | **2,464** |

- Pohon ber-JSON: 32 | Non-JSON: 112

### Split: TEST (143 pohon)

| Kelas | Count |
|-------|------:|
| B1 | 266 |
| B2 | 464 |
| B3 | 1,097 |
| B4 | 524 |
| **Total** | **2,351** |

- Pohon ber-JSON: 35 | Non-JSON: 108

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
