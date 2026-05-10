# Penghitungan Tandan Kelapa Sawit Multi-Tampilan

Pipeline deduplikasi multi-tampilan untuk menghitung jumlah tandan unik per pohon dari 4 hingga 8 sisi foto. Pekerjaan dilakukan **tanpa pelatihan model** — hanya heuristik dan rute (*routing*) algoritmik yang deterministik.

---

## Hasil akhir

Metode **`selector_with_b2b3`** mencapai akurasi `Acc ±1` sebesar **86,67%**
dengan `MAE` **0,3982** pada 953 pohon (`Brand-New-Dataset-YOLO/json/`),
sesuai eksperimen final tertanggal 10 Mei 2026 (iterasi 1–13). Detail di
[`report_10Mei2026.md`](report_10Mei2026.md). Kode produksi:
[`algorithms/selector_with_b2b3.py`](algorithms/selector_with_b2b3.py).

### Tujuh metode terbaik pada 953 pohon

| Peringkat | Metode | `Acc ±1` | `MAE` | Macro class-MAE | Exact profile | Total ±1 | Pohon gagal |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `selector_with_b2b3` | **86,67%** | **0,3982** | 0,3982 | 26,34% | 74,08% | 127 |
| 2 | `selector_iter9_trifurc` | 86,67% | 0,3987 | 0,3987 | 26,34% | 74,08% | 127 |
| 3 | `geometric_mean_blend` | 86,15% | 0,3961 | 0,3961 | 26,86% | 74,50% | 132 |
| 4 | `floor_clamped_hybrid` | 86,04% | 0,4050 | 0,4050 | 25,81% | 74,19% | 133 |
| 5 | `hybrid_vis_corr` | 86,04% | 0,4077 | 0,4077 | 25,29% | 73,98% | 133 |
| 6 | `visibility` | 85,94% | 0,3956 | 0,3956 | 25,29% | 73,56% | 134 |
| 7 | `side_coverage` | 85,94% | 0,3930 | 0,3930 | 25,81% | 73,77% | 134 |

### Rekomendasi pemakaian

| Kebutuhan | Pilihan |
|---|---|
| Akurasi tertinggi (produksi) | `selector_with_b2b3` (86,67%, MAE 0,3982) |
| Alternatif paling sederhana | `hybrid_vis_corr` (86,04%, satu baris bobot) |
| Tercepat dan tahan derau koordinat | `v1_corrected` (0,005 ms/pohon, akurasi 84,37%) |
| Hindari di produksi | `v9_selector` — terlalu cocok (*overfit*) pada dataset 228, turun 12,69 poin persen di 953 |

Sumber data: [`exp_10 May 2026/iter11_results.csv`](exp%2010%20May%202026/iter11_results.csv) dan [`reports/dedup_brand_new_953/accuracy_953.csv`](reports/dedup_brand_new_953/accuracy_953.csv).

### Metrik laporan lengkap

Setiap *run benchmark* wajib melaporkan enam metrik tambahan di luar `Acc ±1` dan `MAE` agregat. Hasil terbaru (10 Mei 2026, `reports/dedup_brand_new_953/accuracy_953.csv`) mencakup seluruh metrik berikut untuk 30 metode:

1. **MAE per kelas** (`MAE_B1` … `MAE_B4`) — rata-rata kesalahan absolut tiap kelas kematangan. B3 adalah bottleneck utama (MAE ~0,75–0,93 untuk metode top).
2. **Macro class-MAE** — rata-rata tidak berbobot dari keempat MAE per kelas. Untuk metode terbaik bernilai ~0,39–0,41.
3. **Akurasi *exact profile*** — persentase pohon dengan prediksi `[B1,B2,B3,B4]` tepat sama dengan *ground truth*. Hanya ~22–27% pohon yang seluruh profilnya tepat, menunjukkan off-by-1 pada satu/two kelas sangat umum.
4. **Total-count MAE** — MAE terhadap jumlah total tandan (`B1+B2+B3+B4`) per pohon. Lebih rendah dari `mean_total_err` karena kesalahan antar-kelas saling menutupi.
5. **Total ±1 accuracy** — persentase pohon yang total prediksinya berada dalam selisih ±1 dari total *ground truth*. ~65–74% untuk metode top.
6. **Per-class mean error (*bias*)** — rata-rata kesalahan bertanda per kelas. Semua metode top memiliki bias positif pada B1–B3 (overcount) dan negatif pada B4 (undercount), menunjukkan kecenderungan klasifikasi naive ke kelas tengah.

Lihat tabel lengkap di atas atau berkas CSV `reports/dedup_brand_new_953/accuracy_953.csv` untuk nilai numerik seluruh metode.

---

## Konteks masalah

Satu tandan kelapa sawit dapat tertangkap pada beberapa sisi foto sekaligus, terutama jika posisinya dekat tepi *frame*. Akibatnya, penjumlahan langsung deteksi dari semua sisi melebihi jumlah sebenarnya sekitar 83,4%. Tujuan riset ini adalah mengubah deteksi per sisi menjadi **jumlah tandan unik per kelas kematangan** B1 hingga B4.

Batasan riset bersifat ketat: 100% algoritmik, tanpa pelatihan, *embedding*, *backprop*, atau pencocok terlatih (*learned matcher*). Pencocokan ketat antar sisi (Hungarian, *graph matching*, *clustering*) terbukti gagal pada label TXT karena derau koordinat — pada *benchmark* 953, metode `naive` hanya mencapai 3,99% dan `relaxed_match` 5,98%.

Kelas kematangan bersifat ordinal B1 menjadi B4: B1 merah paling matang (posisi bawah), B2 transisi, B3 hitam, B4 kecil berduri (posisi atas). Ambiguitas inti adalah **B2↔B3** yang bersifat irreducible — bukan derau label, melainkan kemiripan visual antar dua tahap kematangan.

---

## Dataset

| Sumber | Jumlah pohon | Status |
|---|---:|---|
| `Brand-New-Dataset-YOLO/json/` | **953** | Kanonik, GT lengkap (10 Mei 2026) |
| `json_05 mei 2026/` | 882 | Legacy, *snapshot* 5 Mei |
| `json/` | 228 | Legacy, dataset pengembangan v9 |
| `archive/json_30 April 2026/` | 727 | Legacy, *snapshot* 30 April |

Total 953 pohon terdiri atas dua varietas: DAMIMAS (854 pohon) dan LONSUM (99 pohon). Mayoritas pohon difoto dari 4 sisi, sementara 45 pohon difoto dari 8 sisi. Resolusi gambar 960×1280 piksel.

Sumber *ground truth*: web app [`tools_sawit/`](tools_sawit/) dengan skema v2 — satu file JSON per `tree_name` yang berisi anotasi *bounding box*, daftar tandan unik, dan ringkasan jumlah per kelas.

---

## Tabel *benchmark* lengkap pada 953 pohon

Daftar lengkap 16 metode aktif diurutkan menurun berdasarkan `Acc ±1`. Hasil dihitung dengan `scripts/dedup_brand_new_953.py`.

| Metode | `Acc ±1` | `MAE` | Macro class-MAE | Exact profile | Total ±1 | `mean_total_err` | Pohon gagal |
|---|---:|---:|---:|---:|---:|---:|---:|
| `selector_with_b2b3` | **86,67%** | **0,3982** | 0,3982 | 26,34% | 74,08% | 1,593 | 127 |
| `selector_iter9_trifurc` | 86,67% | 0,3987 | 0,3987 | 26,34% | 74,08% | 1,595 | 127 |
| `geometric_mean_blend` | 86,15% | 0,3961 | 0,3961 | 26,86% | 74,50% | 1,584 | 132 |
| `hybrid_vis_corr` | 86,04% | 0,4077 | 0,4077 | 25,29% | 73,98% | 1,631 | 133 |
| `visibility` | 85,94% | 0,3956 | 0,3956 | 25,29% | 73,56% | 1,582 | 134 |
| `side_coverage` | 85,94% | 0,3930 | 0,3930 | 25,81% | 73,77% | 1,572 | 134 |
| `density_scaled_vis` | 85,94% | 0,4024 | 0,4024 | 25,39% | 73,56% | 1,610 | 134 |
| `v9_median_strong5` | 85,73% | 0,4006 | 0,4006 | 27,39% | 72,51% | 1,602 | 136 |
| `v8_entropy_modulated` | 84,78% | 0,4507 | 0,4507 | 23,92% | 66,32% | 1,803 | 145 |
| `v9_b2_median_v6` | 84,78% | 0,4294 | 0,4294 | 23,08% | 69,78% | 1,718 | 145 |
| `v8_entropy_stacking` | 84,78% | 0,4507 | 0,4507 | 23,92% | 66,32% | 1,803 | 145 |
| `v9_selector` | 84,68% | 0,4410 | 0,4410 | 22,35% | 68,21% | 1,764 | 146 |
| `v7_stacking_bracketed` | 84,58% | 0,4284 | 0,4284 | 25,39% | 68,52% | 1,714 | 147 |
| `v7_stacking_density` | 84,58% | 0,4347 | 0,4347 | 23,92% | 67,89% | 1,739 | 147 |
| `corrected` | 84,37% | 0,4158 | 0,4158 | 23,29% | 68,52% | 1,663 | 149 |
| `v8_b2_b4_boosted` | 84,37% | 0,4111 | 0,4111 | 26,86% | 71,98% | 1,644 | 149 |
| `v6_selector` | 84,26% | 0,4436 | 0,4436 | 21,93% | 67,89% | 1,774 | 150 |
| `adaptive_corrected` | 82,58% | 0,4599 | 0,4599 | 21,51% | 65,58% | 1,840 | 166 |
| `best_visibility_grid` | 80,80% | 0,4596 | 0,4596 | 19,73% | 65,90% | 1,838 | 183 |
| `class_aware_vis` | 70,93% | 0,5456 | 0,5456 | 12,38% | 58,45% | 2,183 | 277 |
| `relaxed_match` | 5,98% | 1,8114 | 1,8114 | 2,41% | 5,04% | 7,246 | 896 |
| `naive` | 3,99% | 2,2804 | 2,2804 | 1,89% | 2,83% | 9,122 | 915 |

Catatan: metode `naive`, `relaxed_match`, dan `v7_ordinal_b3` sengaja dipertahankan sebagai pembanding dan bukti bahwa pendekatan pencocokan langsung tidak dapat diandalkan di skala penuh.

---

## Regresi lintas dataset

Kenaikan ukuran dataset dari 228 menjadi 953 pohon memengaruhi setiap metode secara berbeda. Tabel di bawah menunjukkan akurasi `Acc ±1` pada lima ukuran dataset.

| Metode | 228 | 478 | 727 | 882 | **953** | Delta 228→953 |
|---|---:|---:|---:|---:|---:|---:|
| `selector_with_b2b3` | — | — | — | — | **86,67%** | — (puncak baru, 10 Mei 2026) |
| `selector_iter9_trifurc` | — | — | — | — | 86,67% | — (iter11, belum dievaluasi pada snapshot lama) |
| `geometric_mean_blend` | — | — | — | — | 86,15% | — (iter11, belum dievaluasi pada snapshot lama) |
| `floor_clamped_hybrid` | — | — | — | — | 86,04% | — (iter11, belum dievaluasi pada snapshot lama) |
| `hybrid_vis_corr` | — | — | — | — | 86,04% | — (juara sebelumnya) |
| `visibility` / `v2_visibility` | 92,54% | 90,38% | 89,41% | 89,34% | 85,94% | −6,60 pp |
| `v5_best_visibility` | 92,54% | 90,38% | 89,41% | 89,34% | 85,94% | −6,60 pp |
| `v9_b2_median_v6` | 96,05% | 92,68% | 89,00% | 88,78% | 84,78% | −11,27 pp |
| `v8_entropy_modulated` | 94,30% | 91,63% | 88,86% | 88,78% | 84,78% | −9,52 pp |
| `v9_selector` | **97,37%** | 92,68% | 89,27% | 88,78% | 84,68% | −12,69 pp |
| `v7_stacking_bracketed` | 94,30% | 91,84% | 88,45% | 88,44% | 84,58% | −9,72 pp |
| `v7_stacking_density` | 94,30% | 91,84% | 88,45% | 88,44% | 84,58% | −9,72 pp |
| `corrected` / `v1_corrected` | 90,79% | 89,12% | 87,90% | 88,21% | 84,37% | **−6,42 pp** |
| `v8_b2_b4_boosted` | 92,54% | 91,00% | 87,62% | 88,21% | 84,37% | −8,17 pp |
| `v6_selector` | 96,05% | 91,84% | 88,86% | 88,55% | 84,26% | −11,79 pp |
| `v5_adaptive_corrected` | 93,86% | 89,96% | 86,11% | 86,28% | 82,58% | −11,28 pp |

Lima poin penting dari regresi ini:

1. `selector_with_b2b3` adalah puncak baru pada 953 (86,67%, MAE 0,3982). Trifurkasi selector + koreksi split B2↔B3 mengalahkan `hybrid_vis_corr` 0,63 poin persen sambil memangkas MAE 2,32%. Detail: [`report_10Mei2026.md`](report_10Mei2026.md).
2. `hybrid_vis_corr` (juara sebelumnya). Komposisi sederhana mengalahkan seluruh *selector* berbasis aturan kompleks.
3. Metode sederhana yaitu `v1_corrected` dan `v2_visibility` paling stabil dengan delta hanya 6,4 hingga 6,6 poin persen — generalisasi terbaik.
4. `v9_selector` turun 12,69 poin persen — bukti kuat bahwa *narrow overrides* yang dirancang pada dataset 228 terlalu cocok pada pola lokal dan tidak generalisasi.
5. Pencocokan ketat (`naive`, `relaxed_match`) gagal total pada skala penuh, mengonfirmasi bahwa derau koordinat label TXT tidak dapat diatasi tanpa *embedding* lintas tampilan.

Sumber: `reports/benchmark_228/`, `reports/benchmark_478/`, `reports/benchmark_727/`, `reports/benchmark_882/`, dan `reports/dedup_brand_new_953/`.

---

## Ringkasan algoritma per generasi

Setiap generasi metode dijelaskan ringkas di bawah ini. Detail rumus, derivasi parameter, dan analisis oracle terdapat di [`RESEARCH.md`](RESEARCH.md).

**v1 — `corrected`**: Pembagi global per kelas yang diturunkan dari rasio jumlah naive terhadap GT pada dataset 727. Setiap deteksi dibagi dengan faktor tetap (B1=2,060, B2=1,842, B3=1,861, B4=1,654). Sederhana, langsung memangkas hitungan berlebih, dan paling tahan derau koordinat karena tidak menggunakan posisi *bbox*.

**v2 — `visibility`**: Pembobotan berbasis fungsi Gauss menurut jarak deteksi dari pusat *frame*. Tandan di tengah dianggap pasti unik (bobot mendekati 1), sementara tandan di tepi diturunkan bobotnya karena berpotensi terlihat dari sisi sebelah. Generalisasi paling stabil di seluruh ukuran dataset.

**v5 — `adaptive_corrected`**: Pembagi adaptif yang menyesuaikan diri terhadap total deteksi per pohon. Pohon padat memerlukan koreksi lebih kuat karena kemungkinan duplikasi lebih tinggi. Skor cukup baik pada dataset kecil, namun paling lemah saat skala dataset bertambah.

**v6 — `v6_selector`**: Titik balik konseptual — *routing per regime*. Tidak ada satu rumus tunggal yang optimal untuk semua jenis pohon. v6 membaca fitur permukaan (jumlah deteksi, sisi aktif, rasio per kelas) lalu memilih metode yang paling sesuai. Mencapai 96,49% pada dataset 228.

**v7 — `stacking_bracketed`**: Menambahkan koreksi densitas vertikal (deteksi yang bertumpuk pada rentang `y` sempit kemungkinan duplikat) dan *bracket constraint* yaitu lantai dan plafon fisik (estimasi tidak boleh kurang dari deteksi maksimum per sisi, dan tidak boleh lebih dari `naive ÷ 1,10`).

**v8 — `b2_b4_boosted`, `entropy_modulated`, dan varian lain**: Kumpulan *specialist tools* per kelas. Kelas B2 dan B4 yang paling rentan *overcount* mendapat pengali pembagi tambahan (B2 ×1,10, B4 ×1,08). Ada pula varian berbasis entropi distribusi sisi dan *floor anchor* untuk pohon kecil.

**v9 — `v9_selector`**: *Narrow overrides* di atas v6, mencapai 97,37% pada dataset 228 dengan menambah aturan khusus untuk empat *regime* sempit. Skor turun drastis menjadi 84,68% pada 953 — bukti klasik *overfit*.

**`hybrid_vis_corr`**: Rata-rata terbobot dari `visibility` dan `adaptive_corrected` dengan komposisi 60% visibility dan 40% *adaptive*. Juara 953 sebelum eksperimen 10 Mei 2026.

**`selector_with_b2b3` (juara 953 saat ini)**: Selector trifurkasi + koreksi split B2↔B3. Tahap 1 memilih estimator dasar per profil pohon (B3-dominan padat → median3_floor, B1 cukup + B3 sedikit + B4 sedikit → adaptive_corrected, lainnya → geometric_mean_blend). Tahap 2 mempertahankan total `B2 + B3` namun mengalokasikan ulang rasio menggunakan frekuensi naive — menjawab ambiguitas visual B2↔B3. Hasil 86,67% / MAE 0,3982 pada 953 pohon, validated train/val/test held-out tanpa overfit (worst_drop 0,00 pp).

Pelajaran utama: pencocokan *bbox* individual gagal pada label TXT bernoise, koreksi statistik agregat jauh lebih efektif, dan ambiguitas B2↔B3 menjadi *ceiling* irreducible yang membatasi seluruh metode.

---

## Cara menjalankan

```bash
pip install -r requirements.txt

# Benchmark utama (953 pohon, kanonik) — output: reports/dedup_brand_new_953/
python scripts/dedup_brand_new_953.py

# Audit GT
python scripts/count_all_trees.py
python scripts/count_gt_vs_naive.py
```

Untuk reproduksi *benchmark* legacy pada *snapshot* 228, 478, 727, dan 882 pohon, gunakan `scripts/benchmark_multidim.py` dengan variabel lingkungan `JSON_DIR` dan `OUT_DIR`. Hasil tersimpan di `reports/benchmark_*/`.

---

## Struktur repositori

```
Brand-New-Dataset-YOLO/    Dataset kanonik 953 pohon (gambar, label YOLO, JSON GT)
  data.yaml                Konfigurasi YOLO 4 kelas
  images/{train,val,test}/ Gambar sumber
  labels/{train,val,test}/ Label YOLO format TXT
  json/                    953 berkas JSON GT (satu per tree_name)

algorithms/                Modul algoritma (satu berkas per metode)
scripts/                   Skrip audit, perhitungan, dan benchmark
  dedup_brand_new_953.py   Benchmark utama 953 pohon
  benchmark_multidim.py    Benchmark legacy multi-snapshot
reports/                   Output skrip
  dedup_brand_new_953/     Hasil final 953 pohon
  benchmark_{228,478,727,882}/  Snapshot legacy

dataset/                   Konfigurasi YOLO untuk pelatihan model deteksi
json_05 mei 2026/          Snapshot legacy 882 pohon
json/                      Snapshot legacy 228 pohon
archive/                   Snapshot legacy lainnya (read-only)
tools_sawit/               Web app annotator (vanilla JS, skema v2)

RESEARCH.md                Dokumen riset utama (baca Bagian 0 lebih dulu)
report_05Mei2026.md        Laporan rinci pada 882 pohon
CLAUDE.md / AGENTS.md      Panduan operasional untuk asisten otomatis
```

---

## Skema JSON GT

```json
{
  "tree_id": "DAMIMAS_A21B_0001",
  "tree_name": "DAMIMAS_A21B_0001",
  "split": "train",
  "images": {
    "sisi_1": {
      "annotations": [
        {"class_name": "B3", "bbox_yolo": [0.66, 0.41, 0.06, 0.04], "box_index": 0}
      ]
    }
  },
  "bunches": [{"bunch_id": 1, "class": "B3", "appearance_count": 2}],
  "summary": {"by_class": {"B1": 1, "B2": 2, "B3": 5, "B4": 0}}
}
```

Nilai `summary.by_class` menjadi *ground truth* jumlah unik per kelas yang menjadi acuan evaluasi.

---

## Batasan riset

Pekerjaan ini bersifat 100% algoritmik dan tidak boleh memakai: jaringan Siamese, *CNN embedding*, MLP atas fitur *bbox*, *learned threshold* via *backprop*, ataupun pencocokan ketat (Hungarian, *graph*, *clustering*) langsung pada label TXT karena rentan terhadap derau koordinat.

---

## Dokumen pelengkap

- [`RESEARCH.md`](RESEARCH.md) — dokumen riset utama, mulai dari Bagian 0.
- [`report_10Mei2026.md`](report_10Mei2026.md) — laporan eksperimen final 10 Mei 2026 (`selector_with_b2b3`).
- [`report_05Mei2026.md`](report_05Mei2026.md) — laporan rinci hasil pada 882 pohon.
- [`reports/benchmark_multidim/REPORT.md`](reports/benchmark_multidim/REPORT.md) — laporan multi-dimensi (akurasi, kecepatan, *robustness*, *domain*).
- [`CLAUDE.md`](CLAUDE.md) dan [`AGENTS.md`](AGENTS.md) — panduan operasional untuk asisten otomatis.
