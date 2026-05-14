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
| B1 | 937 | 0 | 937 |
| B2 | 1,780 | 0 | 1,780 |
| B3 | 5,013 | 0 | 5,013 |
| B4 | 2,009 | 0 | 2,009 |
| **TOTAL** | **9,739** | **0** | **9,739** |

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
| B1 | 929 | 10.2% |
| B2 | 1,701 | 18.6% |
| B3 | 4,608 | 50.4% |
| B4 | 1,906 | 20.8% |
| **Total** | **9,144** | 100% |

- Pohon ber-JSON: 854 | Non-JSON: 0

### LONSUM (99 pohon)

| Kelas | Count | % |
|-------|------:|---:|
| B1 | 8 | 1.3% |
| B2 | 79 | 13.3% |
| B3 | 405 | 68.1% |
| B4 | 103 | 17.3% |
| **Total** | **595** | 100% |

- Pohon ber-JSON: 99 | Non-JSON: 0

---

## 4. Breakdown per Split

### Split: TRAIN (607 pohon)

| Kelas | Count |
|-------|------:|
| B1 | 648 |
| B2 | 1,158 |
| B3 | 3,188 |
| B4 | 1,231 |
| **Total** | **6,225** |

- Pohon ber-JSON: 607 | Non-JSON: 0

### Split: VAL (179 pohon)

| Kelas | Count |
|-------|------:|
| B1 | 129 |
| B2 | 339 |
| B3 | 937 |
| B4 | 402 |
| **Total** | **1,807** |

- Pohon ber-JSON: 179 | Non-JSON: 0

### Split: TEST (167 pohon)

| Kelas | Count |
|-------|------:|
| B1 | 160 |
| B2 | 283 |
| B3 | 888 |
| B4 | 376 |
| **Total** | **1,707** |

- Pohon ber-JSON: 167 | Non-JSON: 0

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
