# Penghitungan Tandan Kelapa Sawit Multi-Tampilan

Pipeline deduplikasi multi-tampilan untuk menghitung jumlah tandan unik per pohon dari 4 hingga 8 sisi foto. Pekerjaan dilakukan **tanpa pelatihan model** — hanya heuristik dan rute (*routing*) algoritmik yang deterministik.

---

## Hasil akhir

Metode **`M01_selector_b2b3`** mencapai akurasi `Acc ±1` sebesar **86,67%**
dengan **Macro class-MAE 0,3982** (Total-count MAE 1,4145, Total ±1 74,08%, Exact profile 26,34%) pada 953 pohon (`Brand-New-Dataset-YOLO/json/`),
sesuai eksperimen final tertanggal 10 Mei 2026 (iterasi 1–13). Detail di
[`report_10Mei2026.md`](report_10Mei2026.md). Kode produksi:
[`algorithms/M01_selector_b2b3.py`](algorithms/M01_selector_b2b3.py).

### Tujuh metode terbaik pada 953 pohon

Catatan: pada dataset 4 kelas berimbang (n sama tiap kelas), `Macro class-MAE` ≡ MAE rata-rata flat. Kolom MAE redundan sengaja dihilangkan.

| Peringkat | Metode | `Acc ±1` | Macro class-MAE | `MAE_B1` | `MAE_B2` | `MAE_B3` | `MAE_B4` | Exact profile | Total-count MAE | Total ±1 | bias B1 | bias B2 | bias B3 | bias B4 | Pohon gagal |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `M01_selector_b2b3` | **86,67%** | **0,3982** | 0,1805 | 0,3463 | 0,7566 | 0,3095 | 26,34% | 1,4145 | 74,08% | +0,1448 | +0,1763 | +0,1689 | −0,1039 | 127 |
| 2 | `M02_selector_trifurc` | 86,67% | 0,3987 | 0,1805 | 0,3484 | 0,7566 | 0,3095 | 26,34% | 1,4145 | 74,08% | +0,1448 | +0,1763 | +0,1689 | −0,1039 | 127 |
| 3 | `M03_blend_geometric` | 86,15% | 0,3961 | 0,1752 | 0,3379 | 0,7671 | 0,3043 | 26,86% | 1,4061 | 74,50% | +0,1417 | +0,1322 | +0,1522 | −0,1700 | 132 |
| 4 | `M04_blend_floor_clamped` | 86,04% | 0,4050 | 0,2078 | 0,3400 | 0,7681 | 0,3043 | 25,81% | 1,4103 | 74,19% | +0,1910 | +0,1343 | +0,1616 | −0,1700 | 133 |
| 5 | `M05_blend_vis_divide` | 86,04% | 0,4077 | 0,2078 | 0,3400 | 0,7692 | 0,3137 | 25,29% | 1,4145 | 73,98% | +0,1910 | +0,1343 | +0,1605 | −0,1794 | 133 |
| 6 | `M06_weight_visibility` | 85,94% | 0,3956 | 0,2078 | 0,3326 | 0,7314 | 0,3106 | 25,29% | 1,3641 | 73,56% | +0,1910 | +0,1228 | +0,0976 | −0,1973 | 134 |
| 7 | `M07_weight_coverage` | 85,94% | 0,3930 | 0,2078 | 0,3326 | 0,7303 | 0,3012 | 25,81% | 1,3599 | 73,77% | +0,1910 | +0,1228 | +0,0986 | −0,1878 | 134 |

### Rekomendasi pemakaian

| Kebutuhan | Pilihan |
|---|---|
| Akurasi tertinggi (produksi) | `M01_selector_b2b3` (86,67%, Macro class-MAE 0,3982) |
| Alternatif paling sederhana | `M05_blend_vis_divide` (86,04%, satu baris bobot) |
| Tercepat dan tahan derau koordinat | `M15_divide_global` (0,005 ms/pohon, akurasi 84,37%) |
| Hindari di produksi | `M12_selector_overrides` — terlalu cocok (*overfit*) pada dataset 228, turun 12,69 poin persen di 953 |

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

Batasan riset bersifat ketat: 100% algoritmik, tanpa pelatihan, *embedding*, *backprop*, atau pencocok terlatih (*learned matcher*). Pencocokan ketat antar sisi (Hungarian, *graph matching*, *clustering*) terbukti gagal pada label TXT karena derau koordinat — pada *benchmark* 953, metode `M29_baseline_naive_sum` hanya mencapai 3,99% dan `M28_baseline_match_strict` 5,98%.

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

Daftar lengkap 29 metode diurutkan menurun berdasarkan `Acc ±1`. Hasil dihitung dengan `scripts/dedup_brand_new_953.py`. Kolom mengikuti enam metrik wajib di `CLAUDE.md` (Macro class-MAE, MAE per kelas, Exact profile, Total-count MAE, Total ±1, dan bias per kelas).

| Metode | `Acc ±1` | Macro class-MAE | `MAE_B1` | `MAE_B2` | `MAE_B3` | `MAE_B4` | Exact profile | Total-count MAE | Total ±1 | bias B1 | bias B2 | bias B3 | bias B4 | Pohon gagal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `M01_selector_b2b3` | **86,67%** | **0,3982** | 0,1805 | 0,3463 | 0,7566 | 0,3095 | 26,34% | 1,4145 | 74,08% | +0,1448 | +0,1763 | +0,1689 | −0,1039 | 127 |
| `M02_selector_trifurc` | 86,67% | 0,3987 | 0,1805 | 0,3484 | 0,7566 | 0,3095 | 26,34% | 1,4145 | 74,08% | +0,1448 | +0,1763 | +0,1689 | −0,1039 | 127 |
| `M03_blend_geometric` | 86,15% | 0,3961 | 0,1752 | 0,3379 | 0,7671 | 0,3043 | 26,86% | 1,4061 | 74,50% | +0,1417 | +0,1322 | +0,1522 | −0,1700 | 132 |
| `M04_blend_floor_clamped` | 86,04% | 0,4050 | 0,2078 | 0,3400 | 0,7681 | 0,3043 | 25,81% | 1,4103 | 74,19% | +0,1910 | +0,1343 | +0,1616 | −0,1700 | 133 |
| `M05_blend_vis_divide` | 86,04% | 0,4077 | 0,2078 | 0,3400 | 0,7692 | 0,3137 | 25,29% | 1,4145 | 73,98% | +0,1910 | +0,1343 | +0,1605 | −0,1794 | 133 |
| `M06_weight_visibility` | 85,94% | 0,3956 | 0,2078 | 0,3326 | 0,7314 | 0,3106 | 25,29% | 1,3641 | 73,56% | +0,1910 | +0,1228 | +0,0976 | −0,1973 | 134 |
| `M07_weight_coverage` | 85,94% | 0,3930 | 0,2078 | 0,3326 | 0,7303 | 0,3012 | 25,81% | 1,3599 | 73,77% | +0,1910 | +0,1228 | +0,0986 | −0,1878 | 134 |
| `M08_divide_density_vis` | 85,94% | 0,4024 | 0,2078 | 0,3389 | 0,7513 | 0,3116 | 25,39% | 1,3914 | 73,56% | +0,1910 | +0,1312 | +0,1259 | −0,1962 | 134 |
| `M09_median_strong5` | 85,73% | 0,4006 | 0,1637 | 0,3295 | 0,8132 | 0,2959 | 27,39% | 1,4638 | 72,51% | +0,1259 | +0,1049 | +0,3389 | −0,1112 | 136 |
| `M10_entropy_divide` | 84,78% | 0,4507 | 0,1857 | 0,3987 | 0,8741 | 0,3442 | 23,92% | 1,6348 | 66,32% | +0,1626 | +0,2497 | +0,4460 | +0,1007 | 145 |
| `M11_median_b2` | 84,78% | 0,4294 | 0,1931 | 0,3295 | 0,8646 | 0,3305 | 23,08% | 1,5603 | 69,78% | +0,1385 | +0,1049 | +0,4407 | +0,0220 | 145 |
| `M18_entropy_stack` | 84,78% | 0,4507 | 0,1857 | 0,3987 | 0,8741 | 0,3442 | 23,92% | 1,6348 | 66,32% | +0,1626 | +0,2497 | +0,4460 | +0,1007 | 145 |
| `M12_selector_overrides` | 84,68% | 0,4410 | 0,1931 | 0,3799 | 0,8615 | 0,3295 | 22,35% | 1,6044 | 68,21% | +0,1385 | +0,2371 | +0,4313 | +0,0210 | 146 |
| `M13_stack_bracket` | 84,58% | 0,4284 | 0,1637 | 0,3683 | 0,8573 | 0,3242 | 25,39% | 1,5729 | 68,52% | +0,1259 | +0,2088 | +0,4082 | +0,0304 | 147 |
| `M14_stack_density` | 84,58% | 0,4347 | 0,1773 | 0,3683 | 0,8615 | 0,3316 | 23,92% | 1,5939 | 67,89% | +0,1123 | +0,2088 | +0,4040 | +0,0231 | 147 |
| `M15_divide_global` | 84,37% | 0,4158 | 0,2015 | 0,3316 | 0,8164 | 0,3137 | 23,29% | 1,4596 | 68,52% | +0,1847 | +0,1364 | +0,3232 | −0,0325 | 149 |
| `M16_boost_b2b4` | 84,37% | 0,4111 | 0,1637 | 0,3253 | 0,8573 | 0,2980 | 26,86% | 1,4911 | 71,98% | +0,1259 | +0,0504 | +0,4082 | −0,1238 | 149 |
| `M17_selector_regime` | 84,26% | 0,4436 | 0,1931 | 0,3861 | 0,8646 | 0,3305 | 21,93% | 1,6149 | 67,89% | +0,1385 | +0,2497 | +0,4407 | +0,0220 | 150 |
| `M19_divide_adaptive` | 82,58% | 0,4599 | 0,1952 | 0,3746 | 0,9307 | 0,3389 | 21,51% | 1,6905 | 65,58% | +0,1364 | +0,2340 | +0,5194 | +0,0661 | 166 |
| `M20_weight_visibility_grid` | 80,80% | 0,4596 | 0,2424 | 0,3767 | 0,9119 | 0,3075 | 19,73% | 1,5656 | 65,90% | +0,2298 | +0,2256 | +0,5446 | −0,1207 | 183 |
| `M23_agree_side` | 80,80% | 0,4273 | 0,1102 | 0,3463 | 0,8888 | 0,3641 | 22,35% | 1,5603 | 65,37% | −0,0010 | −0,0504 | −0,1647 | −0,2298 | 183 |
| `M27_weight_visibility_adaptive` | 80,27% | 0,4790 | 0,2602 | 0,3882 | 0,9412 | 0,3263 | 18,57% | 1,6474 | 64,01% | +0,2476 | +0,2539 | +0,5572 | −0,0850 | 188 |
| `M24_weight_class_aware` | 70,93% | 0,5456 | 0,2015 | 0,4858 | 1,1878 | 0,3075 | 12,38% | 1,8111 | 58,45% | +0,1847 | +0,3872 | +0,9696 | −0,2046 | 277 |
| `M22_anchor_floor50` | 69,99% | 0,4525 | 0,1312 | 0,3158 | 1,0252 | 0,3379 | 16,89% | 1,5540 | 60,55% | +0,0703 | −0,0388 | −0,7880 | −0,2392 | 286 |
| `M25_consensus_multi` | 25,29% | 0,9121 | 0,1826 | 0,5383 | 2,1731 | 0,7545 | 5,46% | 3,6401 | 16,79% | −0,1763 | −0,5383 | −2,1689 | −0,7545 | 712 |
| `M26_median_per_side` | 25,29% | 0,9121 | 0,1826 | 0,5383 | 2,1731 | 0,7545 | 5,46% | 3,6401 | 16,79% | −0,1763 | −0,5383 | −2,1689 | −0,7545 | 712 |
| `M28_baseline_match_strict` | 5,98% | 1,8114 | 0,4680 | 1,1196 | 4,3221 | 1,3358 | 2,41% | 7,0147 | 5,04% | −0,3316 | −1,0357 | −4,3200 | −1,3211 | 896 |
| `M29_baseline_naive_sum` | 3,99% | 2,2804 | 1,1511 | 1,7775 | 4,8342 | 1,3589 | 1,89% | 9,1217 | 2,83% | +1,1511 | +1,7775 | +4,8342 | +1,3589 | 915 |
| `M21_ordinal_b3` | 0,73% | 3,5842 | 1,9895 | 2,8993 | 6,3242 | 3,1238 | 0,00% | 14,3368 | 0,00% | −1,9895 | −2,8993 | −6,3242 | −3,1238 | 946 |

Catatan: metode `M29_baseline_naive_sum`, `M28_baseline_match_strict`, dan `M21_ordinal_b3` sengaja dipertahankan sebagai pembanding dan bukti bahwa pendekatan pencocokan langsung tidak dapat diandalkan di skala penuh.

---

## Regresi lintas dataset

Kenaikan ukuran dataset dari 228 menjadi 953 pohon memengaruhi setiap metode secara berbeda. Tabel di bawah menunjukkan akurasi `Acc ±1` pada lima ukuran dataset.

| Metode | 228 | 478 | 727 | 882 | **953** | Delta 228→953 |
|---|---:|---:|---:|---:|---:|---:|
| `M01_selector_b2b3` | — | — | — | — | **86,67%** | — (puncak baru, 10 Mei 2026) |
| `M02_selector_trifurc` | — | — | — | — | 86,67% | — (iter11, belum dievaluasi pada snapshot lama) |
| `M03_blend_geometric` | — | — | — | — | 86,15% | — (iter11, belum dievaluasi pada snapshot lama) |
| `M04_blend_floor_clamped` | — | — | — | — | 86,04% | — (iter11, belum dievaluasi pada snapshot lama) |
| `M05_blend_vis_divide` | — | — | — | — | 86,04% | — (juara sebelumnya) |
| `visibility` / `M06_weight_visibility` | 92,54% | 90,38% | 89,41% | 89,34% | 85,94% | −6,60 pp |
| `M20_weight_visibility_grid` | 92,54% | 90,38% | 89,41% | 89,34% | 85,94% | −6,60 pp |
| `M11_median_b2` | 96,05% | 92,68% | 89,00% | 88,78% | 84,78% | −11,27 pp |
| `M10_entropy_divide` | 94,30% | 91,63% | 88,86% | 88,78% | 84,78% | −9,52 pp |
| `M12_selector_overrides` | **97,37%** | 92,68% | 89,27% | 88,78% | 84,68% | −12,69 pp |
| `M13_stack_bracket` | 94,30% | 91,84% | 88,45% | 88,44% | 84,58% | −9,72 pp |
| `M14_stack_density` | 94,30% | 91,84% | 88,45% | 88,44% | 84,58% | −9,72 pp |
| `corrected` / `M15_divide_global` | 90,79% | 89,12% | 87,90% | 88,21% | 84,37% | **−6,42 pp** |
| `M16_boost_b2b4` | 92,54% | 91,00% | 87,62% | 88,21% | 84,37% | −8,17 pp |
| `M17_selector_regime` | 96,05% | 91,84% | 88,86% | 88,55% | 84,26% | −11,79 pp |
| `M19_divide_adaptive` | 93,86% | 89,96% | 86,11% | 86,28% | 82,58% | −11,28 pp |

Lima poin penting dari regresi ini:

1. `M01_selector_b2b3` adalah puncak baru pada 953 (86,67%, Macro class-MAE 0,3982). Trifurkasi selector + koreksi split B2↔B3 mengalahkan `M05_blend_vis_divide` 0,63 poin persen sambil memangkas Macro class-MAE 2,32%. Detail: [`report_10Mei2026.md`](report_10Mei2026.md).
2. `M05_blend_vis_divide` (juara sebelumnya). Komposisi sederhana mengalahkan seluruh *selector* berbasis aturan kompleks.
3. Metode sederhana yaitu `M15_divide_global` dan `M06_weight_visibility` paling stabil dengan delta hanya 6,4 hingga 6,6 poin persen — generalisasi terbaik.
4. `M12_selector_overrides` turun 12,69 poin persen — bukti kuat bahwa *narrow overrides* yang dirancang pada dataset 228 terlalu cocok pada pola lokal dan tidak generalisasi.
5. Pencocokan ketat (`M29_baseline_naive_sum`, `M28_baseline_match_strict`) gagal total pada skala penuh, mengonfirmasi bahwa derau koordinat label TXT tidak dapat diatasi tanpa *embedding* lintas tampilan.

Sumber: `reports/benchmark_228/`, `reports/benchmark_478/`, `reports/benchmark_727/`, `reports/benchmark_882/`, dan `reports/dedup_brand_new_953/`.

---

## Ringkasan algoritma per generasi

Setiap generasi metode dijelaskan ringkas di bawah ini. Detail rumus, derivasi parameter, dan analisis oracle terdapat di [`RESEARCH.md`](RESEARCH.md).

**v1 — `corrected`**: Pembagi global per kelas yang diturunkan dari rasio jumlah naive terhadap GT pada dataset 727. Setiap deteksi dibagi dengan faktor tetap (B1=2,060, B2=1,842, B3=1,861, B4=1,654). Sederhana, langsung memangkas hitungan berlebih, dan paling tahan derau koordinat karena tidak menggunakan posisi *bbox*.

**v2 — `visibility`**: Pembobotan berbasis fungsi Gauss menurut jarak deteksi dari pusat *frame*. Tandan di tengah dianggap pasti unik (bobot mendekati 1), sementara tandan di tepi diturunkan bobotnya karena berpotensi terlihat dari sisi sebelah. Generalisasi paling stabil di seluruh ukuran dataset.

**v5 — `M19_divide_adaptive`**: Pembagi adaptif yang menyesuaikan diri terhadap total deteksi per pohon. Pohon padat memerlukan koreksi lebih kuat karena kemungkinan duplikasi lebih tinggi. Skor cukup baik pada dataset kecil, namun paling lemah saat skala dataset bertambah.

**v6 — `M17_selector_regime`**: Titik balik konseptual — *routing per regime*. Tidak ada satu rumus tunggal yang optimal untuk semua jenis pohon. v6 membaca fitur permukaan (jumlah deteksi, sisi aktif, rasio per kelas) lalu memilih metode yang paling sesuai. Mencapai 96,49% pada dataset 228.

**v7 — `M13_stack_bracket`**: Menambahkan koreksi densitas vertikal (deteksi yang bertumpuk pada rentang `y` sempit kemungkinan duplikat) dan *bracket constraint* yaitu lantai dan plafon fisik (estimasi tidak boleh kurang dari deteksi maksimum per sisi, dan tidak boleh lebih dari `naive ÷ 1,10`).

**v8 — `M16_boost_b2b4`, `M10_entropy_divide`, dan varian lain**: Kumpulan *specialist tools* per kelas. Kelas B2 dan B4 yang paling rentan *overcount* mendapat pengali pembagi tambahan (B2 ×1,10, B4 ×1,08). Ada pula varian berbasis entropi distribusi sisi dan *floor anchor* untuk pohon kecil.

**v9 — `M12_selector_overrides`**: *Narrow overrides* di atas v6, mencapai 97,37% pada dataset 228 dengan menambah aturan khusus untuk empat *regime* sempit. Skor turun drastis menjadi 84,68% pada 953 — bukti klasik *overfit*.

**`M05_blend_vis_divide`**: Rata-rata terbobot dari `visibility` dan `M19_divide_adaptive` dengan komposisi 60% visibility dan 40% *adaptive*. Juara 953 sebelum eksperimen 10 Mei 2026.

**`M01_selector_b2b3` (juara 953 saat ini)**: Selector trifurkasi + koreksi split B2↔B3. Tahap 1 memilih estimator dasar per profil pohon (B3-dominan padat → median3_floor, B1 cukup + B3 sedikit + B4 sedikit → M19_divide_adaptive, lainnya → M03_blend_geometric). Tahap 2 mempertahankan total `B2 + B3` namun mengalokasikan ulang rasio menggunakan frekuensi naive — menjawab ambiguitas visual B2↔B3. Hasil 86,67% / Macro class-MAE 0,3982 pada 953 pohon, validated train/val/test held-out tanpa overfit (worst_drop 0,00 pp).

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
- [`report_10Mei2026.md`](report_10Mei2026.md) — laporan eksperimen final 10 Mei 2026 (`M01_selector_b2b3`).
- [`report_05Mei2026.md`](report_05Mei2026.md) — laporan rinci hasil pada 882 pohon.
- [`reports/benchmark_multidim/REPORT.md`](reports/benchmark_multidim/REPORT.md) — laporan multi-dimensi (akurasi, kecepatan, *robustness*, *domain*).
- [`CLAUDE.md`](CLAUDE.md) dan [`AGENTS.md`](AGENTS.md) — panduan operasional untuk asisten otomatis.
