# Multi-View Oil Palm Bunch Counting

Repository ini berisi riset deduplikasi multi-view untuk menghitung jumlah tandan sawit unik per pohon dari 4-8 sisi foto. Fokus proyek saat ini adalah **counting berbasis label ground truth JSON** dan **dedup algorithmic-only**, bukan training model baru.

## Inti Masalah

Satu tandan yang sama bisa muncul di beberapa sisi foto. Kalau semua bbox langsung dijumlahkan, hasilnya akan terlalu besar karena tandan yang sama terhitung berulang.

Contoh sederhana:

- pohon yang sama difoto dari 4 sisi
- tandan `B3` yang sama terlihat di sisi 1, sisi 2, dan sisi 3
- tanpa dedup, tandan itu dihitung `3x`
- dengan dedup, tandan itu harus dihitung `1x`

Jadi problem utama repo ini bukan sekadar "menghitung bbox", tetapi **mengubah multi-view detection menjadi unique bunch count per kelas**.

## Scope

- Input utama:
  - `json/` untuk 228 pohon yang sudah punya link antar-view
  - `dataset/labels/` untuk pohon non-JSON yang hanya punya label YOLO TXT
- Output utama:
  - evaluasi akurasi dedup pada 228 pohon JSON
  - estimasi count dedup pada seluruh 953 pohon
- Batasan riset:
  - **100% algorithmic / heuristic**
  - tidak boleh ada training, embedding, backprop, atau learned matcher

## Dataset Ringkas

| Item | Nilai |
|---|---:|
| Total pohon | 953 |
| DAMIMAS | 854 |
| LONSUM | 99 |
| Pohon dengan JSON | 228 |
| Pohon non-JSON | 725 |
| Total image | ~3,992+ |
| Sisi per pohon | mayoritas 4, sebagian 8 |

### Kelas kematangan

| Class | Deskripsi singkat |
|---|---|
| `B1` | merah, paling matang, posisi paling bawah |
| `B2` | transisi merah-hitam |
| `B3` | hitam penuh, di atas B2 |
| `B4` | paling kecil, berduri, paling atas |

Catatan penting: `B1 -> B4` bersifat ordinal. Ambiguitas utama tetap pada `B2` vs `B3`.

## Mengapa Dedup Penting

Hasil audit awal pada 228 pohon JSON menunjukkan:

- **naive sum overcount sekitar 78.8%**
- jadi dedup bukan kosmetik, tetapi komponen inti pipeline counting

Secara praktis:

- `naive` hampir selalu menghitung terlalu banyak
- metode matching yang terlalu rigid sering malah undercount
- metode terbaik harus cukup agresif mengurangi duplikasi, tetapi tidak sampai membuang tandan yang sebenarnya berbeda

## Status Terkini

Status terbaru yang harus dipakai adalah override **2026-04-24**. Jika ada file lama yang masih menyebut plateau `92%`, `93.86%`, `94.30%`, atau `v6_selector` sebagai best current method, anggap itu **outdated**.

### Benchmark Utama: JSON 228 Tree

Metric utama:

- **Acc +/-1 per class per tree**
- metrik sekunder: **MAE** dan **Mean Total Error**

| Rank | Method | Acc +/-1 | MAE |
|---:|---|---:|---:|
| 1 | `v9_selector` | **98.68%** | **0.2533** |
| 2 | `v9_b2_median_v6` | 96.49% | 0.2588 |
| 3 | `v6_selector` | 96.49% | 0.2632 |
| 4 | `v9_median_strong5` | 95.18% | 0.2390 |
| 5 | `stacking_bracketed_v7` | 94.30% | 0.2643 |

### Rekomendasi Pakai Metode

- JSON dengan GT:
  - pakai **`v9_selector`**
- Non-JSON tanpa GT:
  - prioritaskan `hybrid_vis_corr`, `side_coverage`, `stacking_density_v7`, `best_visibility_grid`, atau `visibility`
- Jangan asumsikan `v9_selector` otomatis terbaik untuk non-JSON, karena benchmark utamanya masih pada data JSON 228 tree

## Transparansi Perjalanan Metode

Bagian ini sengaja dibuat lebih lengkap supaya pembaca baru bisa melihat evolusi riset dari baseline awal sampai metode terbaik saat ini.

### Ringkasan Kronologis

| Tahap | Fokus | Hasil utama | Makna |
|---|---|---:|---|
| `naive` | jumlah semua bbox lintas sisi | overcount ~78.8%; pada benchmark ketat sangat buruk | baseline awal, tidak layak dipakai |
| JSON-01 | audit kualitas label | 0% mismatch kelas | ceiling bukan dari label noise |
| v1 `dedup_research.py` | grid search heuristik dasar | best `corrected` = **90.79%**, MAE 0.2851 | dedup sederhana sudah jauh lebih baik dari naive |
| v2 `dedup_research_v2.py` | visibility weighting + adaptive ridge + stack | best `visibility` = **92.11%**, MAE 0.2719 | geometri sederhana membantu |
| v3 `dedup_research_v3.py` | threshold dari `_confirmedLinks` + per-class ridge | best `per_class_ridge` = **90.79%**, MAE 0.2741 | threshold learned-from-links tidak memberi breakthrough |
| v4 `dedup_research_v4.py` | pixel-aware geometry + Hungarian | best `visibility` = **92.11%**, MAE 0.2719 | tambah sinyal visual belum menembus v2 |
| v5 `dedup_research_v5.py` | adaptive divisor, relaxed matching, class-aware heuristics | best `adaptive_corrected` = **93.86%**, MAE 0.2774 | perbaikan stabil pertama di atas 93% |
| v6 `dedup_research_v6.py` | selector antar-metode berbasis regime | best `v6_selector` = **96.49%**, MAE 0.2632 | lompatan besar; tidak ada satu rumus global yang cukup |
| v7 `dedup_research_v7.py` | generalization-first stacking + ordinal density | best `stacking_bracketed` = **94.30%**, MAE 0.2643 | family global-stacking membaik, tapi kalah dari selector |
| v8 `dedup_research_v8.py` | entropy / side-distribution / consensus | best `stacking_bracketed_v7` = **94.30%**, MAE 0.2643 | eksplorasi lanjutan tidak melewati v7 |
| v9 `dedup_research_v9.py` | regime-aware selector sempit di atas v6/v7/v8 | best `v9_selector` = **98.68%**, MAE 0.2533 | current best; target 97-98% tercapai |

### Penjelasan Sederhana per Generasi

#### 1. `naive`

Cara paling awal: semua bbox di semua sisi langsung dijumlahkan.

- kelebihan: sangat sederhana
- kekurangan: tandan yang terlihat di banyak sisi dihitung berulang
- verdict: **tidak boleh dipakai sebagai output final**

#### 2. v1: koreksi statistik dasar

Generasi awal mencoba membagi jumlah deteksi dengan faktor koreksi sederhana.

- inti idenya: overcount punya pola, jadi bisa dikoreksi secara global
- hasil: jauh lebih baik dari naive
- batasnya: satu faktor global belum cukup untuk semua pola pohon

#### 3. v2: visibility / geometric downweighting

Ide utamanya: objek yang berada di posisi tertentu di frame punya peluang lebih besar menjadi duplikat antar-sisi.

- visibility memberi bobot berbeda pada bbox berdasarkan geometri sederhana
- hasil naik ke `92.11%`
- ini membuktikan bahwa informasi posisi bbox memang relevan

#### 4. v3: threshold dari data link JSON

Generasi ini mencoba mengambil threshold dari `_confirmedLinks`, lalu memprediksi tanpa memakai link saat inferensi.

- secara metodologis masih closed-form dan deterministic
- hasilnya tidak lebih baik dari v2
- pelajarannya: threshold matching saja tidak cukup kuat untuk menembus ceiling saat itu

#### 5. v4: pixel-aware empirical geometry

Di sini ditambahkan sinyal visual ringan, misalnya HSV crop summary, lalu digabung ke matching geometry.

- ekspektasi awal: warna/tekstur mungkin membantu
- hasil aktual: **tidak menembus v2**
- pelajarannya: sinyal pixel ringan tanpa model yang benar-benar robust belum cukup

#### 6. v5: adaptive divisor dan family heuristik yang lebih kaya

v5 mulai terlihat lebih matang karena tidak hanya mengandalkan satu pendekatan.

Yang dicoba antara lain:

- `adaptive_corrected`
- `best_visibility_grid`
- `hybrid_vis_corr`
- `side_coverage`
- `class_aware_vis`
- `relaxed_match`

Hasil terbaik:

- `adaptive_corrected` = `93.86%`

Ini adalah tahap ketika repo mulai jelas bahwa:

- matching ketat sering gagal pada data noisy
- koreksi statistik yang adaptif justru lebih stabil

#### 7. v6: selector antar-regime

Ini titik balik penting.

v6 tidak memaksa satu metode menang untuk semua pohon. Sebaliknya, v6:

- memakai `adaptive_corrected` sebagai default
- me-routing pohon tertentu ke metode lain jika pola bbox-nya cocok

Hasil:

- `v6_selector` = `96.49%`

Makna teknisnya:

- problem ini **bukan** punya satu rumus global terbaik
- performa naik saat kita memilih metode berbeda untuk regime pohon yang berbeda

#### 8. v7: stacking dan density family

v7 mengejar ide lain: mungkin family global stacking yang lebih halus bisa mengungguli v6.

Metode yang muncul:

- `stacking_bracketed`
- `stacking_density`
- `v7_combined`
- `adaptive_bracketed`
- `b3_quadratic_bracketed`

Hasil terbaik:

- `stacking_bracketed` = `94.30%`

Catatan penting:

- sebelumnya laporan v7/v8 sempat tie-break kurang tepat
- sekarang sudah dikoreksi: jika accuracy sama, ranking melihat `MAE`
- setelah koreksi, best v7 tetap `stacking_bracketed`

Kesimpulan:

- family ini kuat
- tapi tetap belum mengalahkan `v6_selector`

#### 9. v8: entropy, per-side distribution, consensus

v8 mencoba memperbaiki error pattern v7 dengan sinyal distribusi sisi.

Metode yang diuji antara lain:

- `entropy_modulated`
- `v8_entropy_stacking`
- `b2_b4_boosted`
- `side_variance`
- `floor_anchor_50`
- `per_side_median`

Hasil terbaik:

- `stacking_bracketed_v7` = `94.30%`

Kesimpulan:

- v8 tidak memberi breakthrough
- artinya bottleneck bukan sekadar kurang fitur agregasi global

#### 10. v9: selector final yang sempit dan deterministik

v9 tidak mulai dari nol. Ia memakai pelajaran dari v6, v7, dan v8:

- `v6_selector` tetap jadi default
- override hanya dilakukan pada regime yang benar-benar sempit dan high-confidence
- override mengambil metode tertentu dari family v7/v8

Contoh logika v9:

1. default `v6_selector`
2. `b4_only_overlap` -> `v7_stacking_bracketed`
3. `classaware_compact_lowb4` -> `v8_b2_b4_boosted`
4. `b3b4_only_lowtotal` -> `v8_floor_anchor_50`
5. `dense_allside_moderatedup` -> `v8_b2_b4_boosted`

Hasil:

- `v9_selector` = **98.68%**
- MAE = **0.2533**
- hanya **3 / 228** tree yang masih gagal

Ini adalah status terbaik repo saat ini.

## Metode yang Pernah Dicoba

Supaya tidak terlihat seolah repo hanya punya "top 5", berikut family metode yang memang pernah diuji.

### Family statistik / divisor

- `naive`
- `corrected`
- `adaptive_corrected`
- `density_scaled_vis`
- `ordinal_prior`
- `side_coverage`
- `hybrid_vis_corr`

### Family visibility / geometry ringan

- `visibility`
- `best_visibility_grid`
- `adaptive_visibility`
- `class_aware_vis`

### Family matching / graph / cluster

- `hungarian_match`
- `relaxed_match`
- `best_relaxed_grid`
- `learned_graph`
- `cascade_match`
- `feature_cluster`

### Family regression / ensemble closed-form

- `adaptive_ridge`
- `per_class_ridge`
- `stack_3`
- `stack_5`
- `best_ensemble_grid`

### Family v7-v8 stacking / distribution-aware

- `stacking_density`
- `stacking_bracketed`
- `entropy_modulated`
- `v8_entropy_stacking`
- `b2_b4_boosted`
- `floor_anchor_50`
- `per_side_median`

### Family selector akhir

- `v6_selector`
- `v9_b2_median_v6`
- `v9_median_strong5`
- `v9_selector`

## Apa yang Gagal, dan Kenapa

README ini tidak hanya menampilkan metode yang menang. Beberapa arah juga penting dicatat karena menjelaskan batas teknis repo.

### 1. `naive`

- gagal karena overcount ekstrem
- berguna hanya sebagai baseline mentah

### 2. matching yang terlalu ketat

Contoh:

- `hungarian_match`
- `cascade_match`
- `learned_graph`
- `feature_cluster`

Masalah utamanya:

- koordinat bbox antar-sisi tidak stabil
- label TXT non-JSON mengandung noise kelas dan noise koordinat
- threshold matching yang terlalu rigid membuat banyak tandan unik malah dibuang

### 3. satu rumus global untuk semua pohon

Ini pelajaran dari v5-v8.

- satu divisor global bisa bekerja cukup baik
- tetapi tidak optimal untuk semua pola pohon
- beberapa pohon butuh heuristik berbeda

Inilah alasan kenapa `v6_selector` dan `v9_selector` menang.

## Kenapa `v9_selector` Menang

Secara sederhana:

- v1-v5 mencari satu metode terbaik
- v6 menunjukkan bahwa beberapa jenis pohon butuh metode berbeda
- v7-v8 memberi kandidat metode spesialis untuk pola tertentu
- v9 menggabungkan semuanya dalam selector yang sempit, deterministic, dan tidak over-broad

Jadi breakthrough ke `98.68%` datang **bukan** dari satu formula ajaib, tetapi dari:

- default method yang stabil
- override yang sangat terbatas
- pemilihan regime yang tepat

## Total Tandan Seluruh Dataset (945 Pohon, Semua Metode)

Hasil `scripts/dedup_all_953.py` — menjalankan semua metode ke 945 pohon unik (228 JSON + 717 TXT-only). `v7_ordinal_b3` broken (output negatif), dikecualikan dari rekomendasi.

### Jumlah Tandan Total per Metode

Diurutkan dari paling dekat ke target rasio empiris **0,5594** (dihitung dari 228 pohon ber-GT). `v7_ordinal_b3` dikecualikan karena broken (output negatif).

| Rank | Metode | B1 | B2 | B3 | B4 | Total | Rasio | Jarak ke 0,56 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `v8_b2_b4_boosted` | 1.165 | 2.109 | 4.817 | 2.038 | **10.129** | 0,5604 | 0,0010 |
| 2 | `v9_median_strong5` | 1.165 | 2.153 | 4.767 | 2.045 | **10.130** | 0,5605 | 0,0011 |
| 3 | `hybrid_vis_corr` | 1.232 | 2.190 | 4.610 | 1.956 | **9.988** | 0,5526 | 0,0068 |
| 4 | `corrected` | 1.229 | 2.212 | 4.708 | 2.102 | **10.251** | 0,5672 | 0,0078 |
| 5 | `side_coverage` | 1.234 | 2.187 | 4.570 | 1.952 | **9.943** | 0,5502 | 0,0092 |
| 6 | `density_scaled_vis` | 1.232 | 2.179 | 4.591 | 1.934 | **9.936** | 0,5498 | 0,0096 |
| 7 | `v9_b2_median_v6` | 1.158 | 2.153 | 4.836 | 2.156 | **10.303** | 0,5701 | 0,0107 |
| 8 | `visibility` | 1.232 | 2.172 | 4.563 | 1.933 | **9.900** | 0,5478 | 0,0116 |
| 9 | `v7_stacking_density` | 1.128 | 2.260 | 4.807 | 2.144 | **10.339** | 0,5721 | 0,0127 |
| 10 | `v7_stacking_bracketed` | 1.165 | 2.277 | 4.817 | 2.162 | **10.421** | 0,5766 | 0,0172 |
| 11 | `v9_selector` | 1.158 | 2.303 | 4.833 | 2.155 | **10.449** | 0,5782 | 0,0188 |
| 12 | `v6_selector` | 1.158 | 2.317 | 4.836 | 2.156 | **10.467** | 0,5792 | 0,0198 |
| 13 | `best_visibility_grid` | 1.268 | 2.298 | 4.898 | 2.027 | **10.491** | 0,5805 | 0,0211 |
| 14 | `adaptive_corrected` | 1.155 | 2.306 | 4.881 | 2.185 | **10.527** | 0,5825 | 0,0231 |
| 15 | `adaptive_visibility` | 1.289 | 2.318 | 4.922 | 2.051 | **10.580** | 0,5854 | 0,0260 |
| 16 | `v8_entropy_modulated` | 1.205 | 2.309 | 4.857 | 2.236 | **10.607** | 0,5869 | 0,0275 |
| 17 | `v8_entropy_stacking` | 1.205 | 2.309 | 4.857 | 2.236 | **10.607** | 0,5869 | 0,0275 |
| 18 | `class_aware_vis` | 1.229 | 2.433 | 5.253 | 1.928 | **10.843** | 0,6000 | 0,0406 |
| 19 | `v8_side_agreement` | 1.036 | 2.008 | 4.269 | 1.880 | **9.193** | 0,5087 | 0,0507 |
| 20 | `v8_floor_anchor_50` | 1.118 | 2.006 | 3.920 | 1.905 | **8.949** | 0,4952 | 0,0642 |
| — | `v8_per_side_median` | 869 | 1.511 | 2.777 | 1.421 | **6.578** | 0,3640 | 0,1954 ← undercount |
| — | `v8_multi_consensus` | 869 | 1.511 | 2.777 | 1.421 | **6.578** | 0,3640 | 0,1954 ← undercount |
| — | `relaxed_match` | 691 | 893 | 950 | 811 | **3.345** | 0,1851 | 0,3743 ← undercount |
| — | `naive` | 2.196 | 3.924 | 8.471 | 3.482 | **18.073** | 1,0000 | 0,4406 ← overcount |

### Mengapa Target Rasio ~0,56?

Dari 228 pohon yang punya ground truth JSON, kita bisa hitung langsung:

| | Naive sum | Unik (GT) | Rasio unik/naive |
|---|---:|---:|---:|
| B1 | 578 | 291 | 0,504 |
| B2 | 950 | 532 | 0,560 |
| B3 | 2.054 | 1.144 | 0,557 |
| B4 | 826 | 499 | 0,604 |
| **Total** | **4.408** | **2.466** | **0,5594** |

Artinya: dari semua bbox yang terdeteksi pada 228 pohon ber-GT, hanya **55,94%** yang benar-benar tandan unik — sisanya adalah tandan yang sama terlihat dari sisi berbeda. Angka inilah yang jadi patokan: metode dedup yang baik harus menghasilkan rasio mendekati **~0,56** pada keseluruhan dataset.

Metode yang jauh di bawah (misalnya `relaxed_match` = 0,185) artinya undercount parah. Metode yang mendekati 1,0 (naive) artinya tidak melakukan dedup sama sekali.

### Catatan Interpretasi

- **Metode rekomendasi** (`v9_selector`, `v6_selector`, `corrected`, `visibility`, `hybrid_vis_corr`) → total sekitar **9.900–10.527**, rasio dedup ~0,55–0,58 — selaras dengan rasio GT empiris 0,5594.
- **Naive** 18.073 → overcount ~78% dibanding metode dedup terbaik, sesuai temuan audit JSON-05.
- **`relaxed_match`, `v8_per_side_median`, `v8_multi_consensus`** → undercount ekstrem, tidak layak pakai di luar JSON.
- **`v7_ordinal_b3`** → broken (output negatif), dikecualikan.
- Detail per pohon: `reports/dedup_all_953/all_953_per_tree.csv`

## Validasi dan Output Penting

| Lokasi | Isi |
|---|---|
| [`reports/json_05/count_mae_gt.csv`](reports/json_05/count_mae_gt.csv) | baseline GT vs naive pada 228 pohon JSON |
| [`reports/full_gt_count/summary.md`](reports/full_gt_count/summary.md) | ringkasan count semua 953 pohon |
| [`reports/dedup_research/summary.md`](reports/dedup_research/summary.md) | hasil v1 |
| [`reports/dedup_research_v2/summary_v2.md`](reports/dedup_research_v2/summary_v2.md) | hasil v2 |
| [`reports/dedup_research_v3/summary_v3.md`](reports/dedup_research_v3/summary_v3.md) | hasil v3 |
| [`reports/dedup_research_v4/summary_v4.md`](reports/dedup_research_v4/summary_v4.md) | hasil v4 |
| [`reports/dedup_research_v5/summary_v5.md`](reports/dedup_research_v5/summary_v5.md) | hasil v5 |
| [`reports/dedup_research_v6/summary_v6.md`](reports/dedup_research_v6/summary_v6.md) | hasil v6 |
| [`reports/dedup_research_v7/summary_v7.md`](reports/dedup_research_v7/summary_v7.md) | hasil v7 |
| [`reports/dedup_research_v8/summary_v8.md`](reports/dedup_research_v8/summary_v8.md) | hasil v8 |
| [`reports/dedup_research_v9/summary_v9.md`](reports/dedup_research_v9/summary_v9.md) | benchmark terbaik saat ini untuk JSON |
| [`reports/nonjson_dedup_compare/json_accuracy_validation.csv`](reports/nonjson_dedup_compare/json_accuracy_validation.csv) | evaluasi metode pada non-JSON |
| [`reports/nonjson_dedup_report.md`](reports/nonjson_dedup_report.md) | laporan ringkas non-JSON |
| [`reports/dedup_all_953/all_953_per_tree.csv`](reports/dedup_all_953/all_953_per_tree.csv) | semua metode pada 945 pohon — total count + accuracy |

## Menjalankan Script

Semua script dijalankan dari root workspace dan menulis output ke `reports/`.

```bash
# GT counting semua 953 pohon
python scripts/count_all_trees.py

# JSON-05 + JSON-01 audit
python scripts/count_gt_vs_naive.py

# Dedup research v1-v9
python scripts/dedup_research.py
python scripts/dedup_research_v2.py
python scripts/dedup_research_v3.py
python scripts/dedup_research_v4.py
python scripts/dedup_research_v5.py
python scripts/dedup_research_v6.py
python scripts/dedup_research_v7.py
python scripts/dedup_research_v8.py
python scripts/dedup_research_v9.py

# Semua metode (25+) pada 945 pohon — total count + accuracy
python scripts/dedup_all_953.py

# Final dedup semua pohon
python scripts/dedup_all_trees_final.py

# Perbandingan non-JSON + report
python scripts/dedup_nonjson_compare.py
```

## Struktur Repo

```text
json/                    228 file JSON dengan bunch-link antar-view
dataset/                 image dan label YOLO
scripts/                 seluruh script audit, counting, dan dedup research
reports/                 output CSV/MD dari setiap eksperimen
assets/                  visual summary untuk README / laporan
RESEARCH.md              dokumen riset panjang, baca Section 0 dulu
AGENTS.md                instruksi operasional repo
```

## Schema JSON Ringkas

Setiap file JSON per pohon menyimpan:

- metadata pohon
- anotasi per sisi di `images`
- link tandan lintas sisi di `bunches`
- ground-truth count unik di `summary.by_class`

Contoh minimal:

```json
{
  "tree_id": "20260422-DAMIMAS-001",
  "images": {
    "sisi_1": {
      "annotations": [
        {"class_name": "B3", "bbox_yolo": [0.5, 0.5, 0.1, 0.2], "box_index": 0}
      ]
    }
  },
  "bunches": [
    {
      "bunch_id": 1,
      "class": "B3",
      "appearance_count": 2
    }
  ],
  "summary": {
    "by_class": {"B1": 1, "B2": 2, "B3": 5, "B4": 0}
  }
}
```

`summary.by_class` adalah ground truth count unik per kelas. Naive sum adalah jumlah seluruh bbox lintas sisi tanpa dedup.

## Keputusan Riset

- Dedup adalah masalah utama; naive sum overcount sekitar `78.8%`.
- Label JSON konsisten; bottleneck utama bukan label noise, tetapi ambiguitas visual `B2/B3`.
- Pendekatan matching yang terlalu rigid cenderung gagal pada TXT noisy.
- Pendekatan terbaik saat ini adalah **selector antar-regime**, bukan satu formula global.
- Arah lanjutan tetap algorithmic:
  - multi-camera geometry
  - epipolar constraint
  - triangulation
  - topological / combinatorial matching
  - statistical ensemble tanpa learned weights

## Yang Tidak Dilakukan

Pendekatan berikut dianggap di luar scope:

- Siamese / CNN embedding
- learned thresholds via backprop
- MLP pada fitur bbox
- penggunaan learned feature matcher sebagai solusi dedup utama

## Cara Membaca Repo Ini

Untuk pembaca baru, urutan paling masuk akal:

1. baca [README.md](README.md)
2. lanjut ke [AGENTS.md](AGENTS.md) untuk status operasional repo
3. baca [RESEARCH.md](RESEARCH.md), terutama Section 0
4. kalau ingin hasil terbaru, buka [reports/dedup_research_v9/summary_v9.md](reports/dedup_research_v9/summary_v9.md)
5. kalau ingin lihat sejarah lengkap, telusuri folder [reports/](reports/)

## Ringkasan Satu Kalimat

Repositori ini tidak berhenti di "top 5"; ia merekam perjalanan dari baseline `naive`, berbagai family heuristik dan matching, sampai **`v9_selector` sebagai benchmark JSON terbaik saat ini (`98.68%`)**, dengan alasan teknis yang cukup jelas kenapa metode tertentu dipakai dan kenapa metode lain ditinggalkan.
