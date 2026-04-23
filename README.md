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

## Validasi dan Output Penting

| Lokasi | Isi |
|---|---|
| `reports/json_05/` | baseline GT vs naive pada 228 pohon JSON |
| `reports/full_gt_count/` | ringkasan count semua 953 pohon |
| `reports/dedup_research/` | hasil v1 |
| `reports/dedup_research_v2/` | hasil v2 |
| `reports/dedup_research_v3/` | hasil v3 |
| `reports/dedup_research_v4/` | hasil v4 |
| `reports/dedup_research_v5/` | hasil v5 |
| `reports/dedup_research_v6/` | hasil v6 |
| `reports/dedup_research_v7/` | hasil v7 |
| `reports/dedup_research_v8/` | hasil v8 |
| `reports/dedup_research_v9/` | benchmark terbaik saat ini untuk JSON |
| `reports/nonjson_dedup_compare/` | evaluasi metode pada non-JSON |
| `reports/nonjson_dedup_report.md` | laporan ringkas non-JSON |

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

1. baca [README.md](/D:/Work/Assisten%20Dosen/research-method-dedup/README.md)
2. lanjut ke [AGENTS.md](/D:/Work/Assisten%20Dosen/research-method-dedup/AGENTS.md) untuk status operasional repo
3. baca [RESEARCH.md](/D:/Work/Assisten%20Dosen/research-method-dedup/RESEARCH.md), terutama Section 0
4. kalau ingin hasil terbaru, buka [reports/dedup_research_v9/summary_v9.md](/D:/Work/Assisten%20Dosen/research-method-dedup/reports/dedup_research_v9/summary_v9.md)
5. kalau ingin lihat sejarah lengkap, telusuri folder `reports/dedup_research*`

## Ringkasan Satu Kalimat

Repositori ini tidak berhenti di "top 5"; ia merekam perjalanan dari baseline `naive`, berbagai family heuristik dan matching, sampai **`v9_selector` sebagai benchmark JSON terbaik saat ini (`98.68%`)**, dengan alasan teknis yang cukup jelas kenapa metode tertentu dipakai dan kenapa metode lain ditinggalkan.
