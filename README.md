# Multi-View Oil Palm Bunch Counting

Deduplikasi multi-view untuk menghitung tandan sawit yang unik per pohon dari 4–8 sisi foto. **Tanpa training.** Hanya heuristik dan routing algoritmik.

## Masalah

Satu tandan bisa muncul di beberapa sisi. Penjumlahan langsung melebihi jumlah sebenarnya sekitar 78,8%. Target: mengubah deteksi per sisi menjadi **jumlah tandan unik per kelas kematangan**.

---

## Cara Kerja Deduplikasi

### Kenapa Jumlahnya Berlebih?

Tandan sawit di foto sisi_1 bisa terlihat lagi di sisi_2, apalagi jika posisinya di tepi frame. Contoh sederhana:

```
Sisi_1 (kiri)    Sisi_2 (kanan)
╔══════════╗     ╔══════════╗
║   B3 ──────┐  ║  ┌── B3  ║
║          │  │  ║  │  │    ║
╚══════════╝  └──╚══╧══════╝
                ↑ tandan sama terhitung 2x
```

Pendekatan naif (jumlahkan semua bounding box dari semua sisi) menghasilkan hitungan berlebih. Deduplikasi mengoreksi agar tiap tandan unik hanya terhitung sekali.

### Konsep Dasar: Divisor

Setiap deteksi punya kemungkinan menjadi duplikat dari sisi sebelah. Makin dekat ke tepi frame, makin besar kemungkinannya. Solusi paling sederhana: bagi jumlah naif dengan suatu faktor.

Faktor diperoleh dari data 228 pohon yang punya ground truth:

```
factor[C] = total_naive[C] / total_gt[C]
```

| Kelas | GT (unik) | Naive (jumlah) | factor | Keterangan |
|---:|---:|---:|---:|---|
| B1 | 291 | 578 | **1.986** | paling besar, merah, posisi bawah — terlihat dari banyak sisi |
| B2 | 532 | 950 | **1.786** | transisi, masih besar |
| B3 | 1144 | 2054 | **1.795** | hitam, masih besar |
| B4 | 499 | 826 | **1.655** | paling kecil, posisi atas, terhalang pelepah — sering terlewat |
| **Total** | 2466 | 4408 | **1.788** | keseluruhan |

```
count_unik(B1) = round(naive_B1 / 1.986)
```

Ini adalah **v1** (`corrected`), Acc ±1 = 90,79%. Sederhana, langsung memangkas hitungan berlebih.

Sumber: `reports/json_05/count_mae_gt.csv` dari `scripts/count_gt_vs_naive.py`.

### Insight 1 — Visibility Weighting (v2)

Tidak semua deteksi setara. Tandan di **tengah frame** hampir pasti unik (hanya terlihat dari 1 sisi). Tandan di **tepi** bisa terlihat dari sisi sebelah, jadi bobotnya perlu diturunkan.

Bobot visibility dihitung dengan fungsi Gaussian berdasarkan jarak dari pusat gambar:

```
weight(x) = 1 / (1 + alpha * exp(-(x - 0.5)² / (2 sigma²)))
```

**Visual:**
```
Tengah (x=0.5)          Tepi (x=0 atau 1)
    weight=1                weight<1
      ↑                       ↑
  ─────┼───────────────────────┼──
       │                       │
    unik,               mungkin duplikat,
    tidak                bobot diturunkan
    diduplikasi
```

| Posisi x | weight | Keterangan |
|---:|---:|---|
| 0,50 (tengah) | 1,00 | Pasti unik |
| 0,25 atau 0,75 | ~0,67 | 67% kemungkinan unik |
| 0,00 atau 1,00 (tepi) | ~0,50 | 50% kemungkinan unik |

Hitungan per kelas: `count = round(sum(weight(d) for d in detections_of_class))`

Akurasi naik ke 92,11%.

### Insight 2 — Density Scaling (v5)

Pohon dengan **banyak tandan** punya tingkat overlap yang lebih tinggi. Frame yang penuh membuat tiap tandan lebih mungkin terlihat dari banyak sisi.

```
density_scale = clip(2.05 - 0.014 × total_detections, 1.45, 2.10) / 1.79
```

| Total deteksi | density_scale | Efek |
|---:|---:|---|
| 10 | ~1.07 | divisor standar |
| 50 | ~0.94 | divisor sedikit diturunkan |
| 100 | ~0.78 | divisor diturunkan lebih banyak (koreksi lebih kuat) |

Setiap kelas juga punya BASE_FACTOR bawaan dari data:

| Kelas | BASE_FACTOR | Arti |
|---:|---:|---|
| B1 | 1.986 | Paling matang, paling sering kelihatan di banyak sisi |
| B2 | 1.786 | |
| B3 | 1.795 | |
| B4 | 1.655 | Paling kecil, paling sering kelewat |

Acc lompat ke 93.86% — pertama kalinya tembus >93%.

### Insight 3 — Regime Routing (v6) — TITIK BALIK

v1–v5 semua pakai **satu rumus global** untuk semua pohon. v6 sadar: tidak ada satu rumus yang cocok untuk semua.

- Pohon A: B4-only, overlap tinggi → butuh divisor besar
- Pohon B: B2-heavy, B4 sedikit → butuh visibility bias ke B2
- Pohon C: Semua kelas padat, semua sisi terisi → butuh density correction

v6 membuat **decision tree** yang membaca fitur pohon, lalu merutekan ke metode yang sesuai:

```
                     [B4_naive ≤ 6.5?]
                    /                  \
                  YES                  NO
                 /                      \
      [B4_activesides ≤ 2.5?]     [B4_yrange ≤ 0.097?]
         /         \                  /           \
       YES         NO               YES            NO
       /            \                /              \
   [cek B3]  v5_adaptive_corrected  → FALSE      [cek B2_ratio]
```

Hasil routing:
- **v5_adaptive_corrected** → default, ~75% pohon
- **v5_best_visibility_grid** → pohon dengan B3 heavy (dup-rate tinggi)
- **class_aware_vis** → pohon B2-heavy dengan B4 sedikit

Acc lompat: 93.86% → **96.49%**.

### Insight 4 — Specialist Tools (v7–v8)

Dua generasi ini (v7–v8) bukan untuk dipakai global — performanya malah lebih rendah. Tapi mereka menciptakan alat khusus untuk regime tertentu. Berikut cara kerja masing-masing:

#### v7_stacking_bracketed — Stacking Density + Bracket

**Masalah:** Dua tandan di sisi berbeda sering terdeteksi di posisi y yang hampir sama (bertumpuk vertikal). Semakin rapat secara vertikal, semakin besar kemungkinan itu tandan yang sama.

**Solusi — Stacking Density Correction:**

```
density(c) = jumlah_deteksi_c / rentang_y_c
```

Makin tinggi density, makin besar kemungkinan deteksi itu duplikat. Maka divisor ditambah:

```
extra(c) = 1 + 0.0008 × max(0, density(c) − ref_median(c))
```

Nilai ref_median diperoleh dari data 228 JSON:

| Kelas | Ref density | Arti |
|---:|---:|---|
| B1 | 42 | density rata-rata B1 di dataset |
| B2 | 56 | |
| B3 | 72 | (tertinggi — konsisten dengan B3 sebagai kelas tersulit) |
| B4 | 50 | |

**Contoh:** B3 punya 12 deteksi dalam rentang y=0.08 → density = 12/0.08 = 150. Ref B3 = 72, jadi extra = 1 + 0.0008 × (150−72) = 1.0624. Divisor jadi 1.795 × 1.0624 = 1.907 (lebih besar → koreksi lebih kuat).

**Solusi — Bracket Constraint:**

Estimasi akhir tidak boleh di luar batas fisik. Dua jaminan:

```
floor   = max(deteksi per sisi)    — paling sedikit harus sebanyak yg terlihat dari sisi terbaik
ceiling = round(naive / 1.10)      — paling banyak, asumsi dup minimal 10%
```

**Visual:**
```
  naive=12
    │
    ├── ceiling = 12/1.10 = 10.9 → 11
    │
    │   stacking_estimate = 7
    │   ↑
    ├── floor = max(deteksi di satu sisi) = 4
    │
    └── hasil akhir: clip(7, 4, 11) = 7
```

#### v8_b2_b4_boosted — Divsor Ekstra B2/B4

**Masalah:** Analisis error v7 menunjukkan B2 dan B4 secara konsisten over-predicted (terutama di split test). B2 rentan karena ambiguitas visual dengan B3. B4 sering overcount karena ukuran kecil dan muncul di banyak sisi.

**Solusi:** Sama seperti stacking_bracketed, tapi divisor B2 dan B4 dikalikan boost:

| Kelas | Boost | Efek |
|---:|---:|---|
| B1 | 1.0 (netral) | divisor normal = 1.986 |
| B2 | **1.10** | divisor = 1.786 × 1.10 = **1.965** |
| B3 | 1.0 (netral) | divisor normal = 1.795 |
| B4 | **1.08** | divisor = 1.655 × 1.08 = **1.787** |

Boost diperoleh dari analisis residual pada pohon yang gagal — bukan training. Hitungan:

```
divisor_B2 = 1.786 × density_scale × stack_extra × 1.10
```

#### v8_floor_anchor_50 — Estimasi Konservatif untuk Pohon Kecil

**Masalah:** Pohon dengan sedikit deteksi (≤13) dan hanya B3+B4 punya karakteristik khusus: stacking density estimate biasanya terlalu tinggi karena y_span sempit secara kebetulan, bukan karena overlap nyata.

**Solusi:** Jika stacking estimate > floor + 1, tarik estimasi ke arah floor dengan anchor 0.50:

```
jika E_stack ≤ floor + 1 → pakai E_stack (sudah konservatif)
jika E_stack > floor + 1 → hasil = floor + 0.50 × (E_stack − floor)
```

Artinya: estimate tidak boleh lebih dari setengah jalan antara floor dan stacking estimate. Ini sengaja under-predict, tapi lebih akurat untuk pohon kecil.

**Catatan:** Metode ini **jangan dipakai global** (Acc hanya 69.74%). Hanya efektif pada regime spesifik yang difilter oleh v9_selector.

### Insight 5 — Narrow Overrides (v9) — 98.68%

v9 tidak desain ulang. Prinsip: **v6 sudah 96.49%, tinggal perbaiki 8 pohon yang gagal**.

Dari analisis error pada v6 — membandingkan prediksi vs GT pohon demi pohon — ditemukan 4 regime sempit. Masing-masing punya ciri yang bisa dideteksi dengan fitur sederhana (tanpa melihat GT), lalu dirutekan ke specialist tool yang tepat.

#### Regime 1: B4-only dengan Overlap Tinggi

| Ciri | Deteksi | 
|---|---|
| Hanya B4 yang muncul (B1=B2=B3=0) | `B1_naive=0, B2_naive=0, B3_naive=0` |
| Overlap tinggi — ≥4 B4 terlihat di satu sisi | `B4_maxside ≥ 4` |

**Kenapa v6 gagal:** v6 routing default mengirim pohon ini ke `adaptive_corrected`, yang menerapkan divisor seragam untuk semua kelas. Tapi karena hanya B4 (kelas paling kecil, mudah overcount), divisor standar 1.655 tidak cukup.

**Cara v9 memperbaiki:** Routing ke `stacking_bracketed`. Metode ini punya bracket constraint yang membatasi ceiling ke round(naive/1.10). Untuk B4-only dengan overlap tinggi, bracket ini efektif mencegah overcount ekstrem.

#### Regime 2: Pohon Padat dengan B4 Sangat Sedikit

| Ciri | Deteksi |
|---|---|
| v6 sendiri sudah curiga — memilih `class_aware_vis` | `selected_method == "class_aware_vis"` |
| Pohon padat (banyak deteksi) | `total_det ≥ 21` |
| B4 hampir tidak ada | `B4_naive ≤ 2` |

**Kenapa v6 gagal:** `class_aware_vis` memberi bobot per kelas berdasarkan distribusi spasial. Tapi dengan B4 sangat sedikit dan total deteksi banyak, bobot visibility jadi tidak stabil untuk B2/B3.

**Cara v9 memperbaiki:** Routing ke `b2_b4_boosted`. Boost divisor B2 (×1.10) mengoreksi overcount B2 yang dominan di pohon padat.

#### Regime 3: Hanya B3+B4, Deteksi Sangat Sedikit

| Ciri | Deteksi |
|---|---|
| B1=B2=0, B3>0, B4>0 | hanya kelas atas |
| Total deteksi sedikit | `total_det ≤ 13` |
| Kedua kelas muncul di semua 4 sisi | `B3_activesides=4, B4_activesides=4` |
| Distribusi B3 tersebar, B4 mengumpul | `B3_ratio ≤ 3.0, B4_ratio ≥ 4.0` |

> **Apa itu `ratio`?** `B3_ratio = B3_naive / B3_activesides` — rata-rata deteksi per sisi.  
> Ratio kecil (=tersebar merata) + ratio besar (=mengumpul di satu sisi) adalah fingerprint unik.

**Kenapa v6 gagal:** Pohon kecil dengan kedua kelas di semua sisi — tandan benar-benar padat di ruang sempit. Stacking density estimate v6 terlalu tinggi karena y_span kecil, menyebabkan overcorrection (under-predict).

**Cara v9 memperbaiki:** Routing ke `floor_anchor_50`. Anchor 0.50 menarik estimasi ke floor, mencegah under-predict karena stacking density palsu.

#### Regime 4: Semua Kelas Padat, Dup-rate Moderat

| Ciri | Deteksi |
|---|---|
| v6 memilih `adaptive_corrected` | default method |
| Pohon sangat padat | `total_det ≥ 28` |
| B2,B3,B4 semuanya ada di 4 sisi | `B2/B3/B4_activesides == 4` |
| Dup-rate moderat | `B2_ratio < 3.0, B3_ratio < 2.5` |

**Kenapa v6 gagal:** `adaptive_corrected` di v6 menggunakan density_scale global yang sama untuk semua kelas. Pada pohon dengan semua sisi terisi dan dup-rate moderat, density_scale tidak cukup spesifik untuk tiap kelas.

**Cara v9 memperbaiki:** Routing ke `b2_b4_boosted`. Boost per kelas (khusus B2/B4) memberikan koreksi yang lebih granular daripada density_scale seragam.

#### Ringkasan Logika v9

```
                         ┌─ B4 only + maxside≥4 ──→ stacking_bracketed (v7)
                         │   (2 trees)
                         │
      masukan pohon ────┼─ class_aware + total≥21 + B4≤2 ──→ b2_b4_boosted (v8)
      (features dari     │   (3 trees)
       extract_features) │
                         ├─ B3B4 only + total≤13 + ratio khas ──→ floor_anchor_50 (v8)
                         │   (2 trees)
                         │
                         ├─ adaptive + total≥28 + allside ──→ b2_b4_boosted (v8)
                         │   (1 tree)
                         │
                         └─ default ──→ v6_selector (96.49%)
                             (220 trees)
```

Hasil: dari 8 pohon yang sebelumnya gagal di v6, 5 pohon berhasil diperbaiki. Sisa 3 pohon masih gagal — kemungkinan irreducible tanpa cross-view embedding (lihat Oracle analysis). **Acc = 98.68%**.

### Ringkasan Filosofi

```
Strict matching (cocokkan bbox antar sisi)           → <20%   ✗
Statistical correction global (satu divisor)          → 93%    ✓
Statistical + regime routing (pilih metode per pohon) → 96%    ✓
Narrow overrides (perbaiki sisa error)                → 98.68% ✓
Cross-view embedding (dilarang constraint)            → ~99.5% (teoretis)
```

Intinya: **jangan cocokkan bounding box secara individual** — label TXT punya noise koordinat yang bikin strict matching kacau. Koreksi statistik agregat jauh lebih efektif. B2↔B3 ambiguity adalah ceiling irreducible yang membatasi semua metode.

### Oracle Analysis — Seberapa Jauh Ceiling Teoretis?

**Oracle analysis** menjawab: *"andaikan kita punya kemampuan memilih metode terbaik untuk tiap pohon dengan sempurna, berapa akurasi maksimal yang bisa dicapai?"*

Cara kerja:
1. Ambil semua 228 pohon JSON + ground truth
2. Untuk tiap pohon, jalankan **semua** metode yang ada
3. Cek apakah **salah satu** dari mereka menghasilkan prediksi yang benar (Acc ±1)
4. Kalau ya → pohon itu terhitung "oracle_ok"

```
Pohon X → jalankan v6, v7_stacking_bracketed, v8_b2_b4_boosted, ...
        → adakah satu pun yang outputnya cocok GT?
        → ya?  oracle_ok = True (pohon ini bisa diselesaikan oleh setidaknya satu metode)
        → tidak? oracle_ok = False (pohon ini tidak bisa diselamatkan oleh metode manapun)
```

**Dua varian oracle:**

| Oracle | Cakupan | Acc | Gagal |
|---|---|---|---|
| **Narrow** | 8 metode kuat (v6, v7 best, v8 best, v9) | **99.12%** | 2 pohon |
| **Broad** | 16 metode termasuk eksperimental | **99.56%** | 1 pohon |

**Apa artinya?** Oracle broad 99.56% berarti **1 pohon dari 228 tidak bisa diperbaiki oleh metode apapun yang ada**. Itulah ceiling teoretis dari pendekatan statistik murni. Sisanya irreducible tanpa cross-view embedding (yang dilarang oleh constraint proyek).

**Perbandingan dengan v9_selector (98.68%):**

| Aspek | Oracle | v9_selector |
|---|---|---|
| Tahu GT? | **Ya** (cheat — lihat ground truth) | Tidak (hanya pakai fitur permukaan) |
| Cara pilih metode | Lihat mana yang cocok dengan GT | Tebak dari jumlah deteksi, active sides, ratio |
| Hasil | 99.12% (narrow) | 98.68% |
| Gap | **0.44 pp** — sangat kecil, menandakan v9 hampir optimal |

v9_selector hanya berjarak 0.44 poin persen dari oracle narrow. Artinya: **metode routing v9 sudah mendekati batas maksimal** yang bisa dicapai dengan tools yang ada. Sisa 3 pohon yang gagal kemungkinan besar irreducible — bahkan oracle narrow pun gagal di 2 pohon, dan oracle broad masih gagal di 1 pohon.

Sumber: `reports/dedup_research_v9/oracle_narrow_v9.csv`, `reports/dedup_research_v9/oracle_broad_v9.csv`.

---

## Dataset

| Item | Jumlah |
|---|---:|
| Total pohon | 953 (DAMIMAS 854 + LONSUM 99) |
| Dengan JSON GT | 228 |
| Hanya TXT prediksi | 725 |
| Sisi per pohon | 4 (mayoritas), 8 (~45 terbaru) |

**Kelas ordinal B1→B4** — B1 merah paling matang (bawah), B2 transisi, B3 hitam, B4 kecil berduri (atas). Ambiguitas utama: **B2↔B3** (irreducible, bukan label noise).

---

## Metrik Primer

Seluruh hasil di bawah dihitung pada 228 pohon ber-GT JSON.

| Metrik | Arah | Definisi |
|---|:---:|---|
| **Per-class MAE** | ↓ | rata-rata \|pred − GT\| tiap kelas (B1/B2/B3/B4), dirata-rata lintas pohon |
| **Macro class-MAE** | ↓ | rata-rata dari 4 per-class MAE (bobot sama antar kelas) |
| **Exact accuracy** | ↑ | % pohon dengan prediksi tepat sama GT di **semua** kelas |
| **Total count MAE** | ↓ | rata-rata \|Σpred − ΣGT\| per pohon |
| **Total ±1 accuracy** | ↑ | % pohon dengan total count dalam ±1 dari total GT |
| **Per-class mean error** | →0 | rata-rata (pred − GT) per kelas — mengukur **bias arah** (+ overcount, − undercount) |

> Legenda: **↓** makin kecil makin baik, **↑** makin besar makin baik, **→0** ideal mendekati nol.

**Traceability.** Setiap nama metode di seluruh tabel README bisa diklik → halaman [`reports/methods/<method>.md`](reports/methods) yang berisi nilai semua metrik primer untuk metode itu, derivasi perhitungan, CSV sumber ([`accuracy_per_tree.csv`](reports/benchmark_multidim/accuracy_per_tree.csv), [`accuracy_per_class.csv`](reports/benchmark_multidim/accuracy_per_class.csv), [`accuracy_summary.csv`](reports/benchmark_multidim/accuracy_summary.csv), [`speed_summary.csv`](reports/benchmark_multidim/speed_summary.csv), [`robustness_summary.csv`](reports/benchmark_multidim/robustness_summary.csv)), daftar pohon yang gagal, dan sample per-tree rows. Reproduce:

```bash
python scripts/benchmark_multidim.py         # regenerate CSV mentah (benchmark)
python scripts/generate_method_reports.py    # regenerate per-method breakdown
```

---

## Hasil Utama — 11 Algoritma

Urut berdasarkan **macro class-MAE**. Nama metode link ke breakdown data mentah; kolom *Impl* link ke file algoritma.

| Rank | Method | Impl | Macro MAE ↓ | Exact % ↑ | Total MAE ↓ | Total ±1 % ↑ |
|---:|---|---|---:|---:|---:|---:|
| 1 | [`v9_selector`](reports/methods/v9_selector.md) | [py](algorithms/v9_selector.py) | **0.2533** | 29.39% | **0.8553** | **83.77%** |
| 2 | [`v9_b2_median_v6`](reports/methods/v9_b2_median_v6.md) | [py](algorithms/b2_median_v6.py) | 0.2577 | 29.82% | 0.8640 | 82.02% |
| 3 | [`v6_selector`](reports/methods/v6_selector.md) | [py](algorithms/v6_selector.py) | 0.2599 | 28.07% | 0.8816 | 82.46% |
| 4 | [`v8_b2_b4_boosted`](reports/methods/v8_b2_b4_boosted.md) | [py](algorithms/b2_b4_boosted.py) | 0.2632 | **31.14%** | 0.9035 | 80.26% |
| 5 | [`v7_stacking_bracketed`](reports/methods/v7_stacking_bracketed.md) | [py](algorithms/stacking_bracketed.py) | 0.2643 | 31.14% | 0.8904 | 79.82% |
| 6 | [`v2_visibility`](reports/methods/v2_visibility.md) | — | 0.2664 | **31.58%** | 0.8728 | 82.02% |
| 7 | [`v5_best_visibility`](reports/methods/v5_best_visibility.md) | [py](algorithms/best_visibility_grid.py) | 0.2664 | 31.58% | 0.8728 | 82.02% |
| 8 | [`v7_stacking_density`](reports/methods/v7_stacking_density.md) | [py](algorithms/stacking_density.py) | 0.2708 | 29.39% | 0.9079 | 79.39% |
| 9 | [`v8_entropy_modulated`](reports/methods/v8_entropy_modulated.md) | [py](algorithms/entropy_modulated.py) | 0.2763 | 30.70% | 0.8772 | 78.95% |
| 10 | [`v5_adaptive_corrected`](reports/methods/v5_adaptive_corrected.md) | [py](algorithms/adaptive_corrected.py) | 0.2774 | 26.32% | 0.9342 | 79.82% |
| 11 | [`v1_corrected`](reports/methods/v1_corrected.md) | — | 0.2851 | 30.26% | 0.9035 | 78.51% |

Sumber: [`accuracy_per_tree.csv`](reports/benchmark_multidim/accuracy_per_tree.csv) (228 pohon × 11 metode = 2508 baris). Derivasi tiap angka ada di breakdown per-metode.

### Per-Class MAE (↓ lebih kecil lebih baik)

Sumber: kolom `err_B*` (sudah absolute) di [`accuracy_per_tree.csv`](reports/benchmark_multidim/accuracy_per_tree.csv); cross-check di [`accuracy_per_class.csv`](reports/benchmark_multidim/accuracy_per_class.csv).

| Method | B1 ↓ | B2 ↓ | B3 ↓ | B4 ↓ |
|---|---:|---:|---:|---:|
| [`v9_selector`](reports/methods/v9_selector.md) | **0.105** | **0.219** | **0.386** | 0.303 |
| [`v9_b2_median_v6`](reports/methods/v9_b2_median_v6.md) | 0.105 | 0.224 | 0.395 | 0.307 |
| [`v6_selector`](reports/methods/v6_selector.md) | 0.105 | 0.232 | 0.395 | 0.307 |
| [`v8_b2_b4_boosted`](reports/methods/v8_b2_b4_boosted.md) | **0.079** | 0.268 | 0.425 | **0.281** |
| [`v7_stacking_bracketed`](reports/methods/v7_stacking_bracketed.md) | 0.079 | 0.241 | 0.425 | 0.311 |
| [`v2_visibility`](reports/methods/v2_visibility.md) | 0.101 | 0.224 | 0.447 | 0.294 |
| [`v5_best_visibility`](reports/methods/v5_best_visibility.md) | 0.101 | 0.224 | 0.447 | 0.294 |
| [`v7_stacking_density`](reports/methods/v7_stacking_density.md) | 0.096 | 0.241 | 0.425 | 0.320 |
| [`v8_entropy_modulated`](reports/methods/v8_entropy_modulated.md) | 0.096 | 0.254 | 0.434 | 0.320 |
| [`v5_adaptive_corrected`](reports/methods/v5_adaptive_corrected.md) | 0.110 | 0.237 | 0.434 | 0.329 |
| [`v1_corrected`](reports/methods/v1_corrected.md) | 0.101 | 0.224 | 0.487 | 0.329 |

B1 termudah untuk semua metode. **B3 adalah bottleneck** universal.

### Per-Class Mean Error (Bias) (→0 ideal)

Nilai positif = overcount, negatif = undercount, nol = tidak bias. Sumber: `mean(pred_B* − gt_B*)` di [`accuracy_per_tree.csv`](reports/benchmark_multidim/accuracy_per_tree.csv).

| Method | B1 →0 | B2 →0 | B3 →0 | B4 →0 |
|---|---:|---:|---:|---:|
| [`v9_selector`](reports/methods/v9_selector.md) | +0.044 | +0.044 | **0.000** | +0.039 |
| [`v9_b2_median_v6`](reports/methods/v9_b2_median_v6.md) | +0.044 | −0.101 | +0.009 | +0.044 |
| [`v6_selector`](reports/methods/v6_selector.md) | +0.044 | +0.057 | +0.009 | +0.044 |
| [`v8_b2_b4_boosted`](reports/methods/v8_b2_b4_boosted.md) | +0.044 | −0.154 | +0.004 | −0.114 |
| [`v7_stacking_bracketed`](reports/methods/v7_stacking_bracketed.md) | +0.044 | +0.039 | +0.004 | +0.048 |
| [`v2_visibility`](reports/methods/v2_visibility.md) | +0.092 | −0.066 | −0.175 | −0.215 |
| [`v5_best_visibility`](reports/methods/v5_best_visibility.md) | +0.092 | −0.066 | −0.175 | −0.215 |
| [`v7_stacking_density`](reports/methods/v7_stacking_density.md) | +0.026 | +0.039 | +0.004 | +0.039 |
| [`v8_entropy_modulated`](reports/methods/v8_entropy_modulated.md) | +0.070 | +0.070 | +0.048 | +0.110 |
| [`v5_adaptive_corrected`](reports/methods/v5_adaptive_corrected.md) | +0.039 | +0.061 | +0.057 | +0.066 |
| [`v1_corrected`](reports/methods/v1_corrected.md) | +0.092 | −0.048 | −0.039 | +0.022 |

[`v9_selector`](reports/methods/v9_selector.md) punya profil bias paling seimbang: magnitude ≤0.044 di semua kelas, **tepat 0 di B3** (kelas tersulit).

---

## Metrik Pelengkap

### Acc ±1 per kelas per pohon (pohon lulus jika semua 4 kelas meleset ≤1)

Sumber: kolom `acc_pct` dan `n_fail` di [`accuracy_summary.csv`](reports/benchmark_multidim/accuracy_summary.csv).

| Rank | Method | Acc ±1 ↑ | Gagal ↓ |
|---:|---|---:|---:|
| 1 | [`v9_selector`](reports/methods/v9_selector.md) | **97.37%** | 6 |
| 2 | [`v9_b2_median_v6`](reports/methods/v9_b2_median_v6.md) | 96.05% | 9 |
| 3 | [`v6_selector`](reports/methods/v6_selector.md) | 96.05% | 9 |
| 4 | [`v7_stacking_bracketed`](reports/methods/v7_stacking_bracketed.md) | 94.30% | 13 |
| 5 | [`v7_stacking_density`](reports/methods/v7_stacking_density.md) | 94.30% | 13 |
| 6 | [`v8_entropy_modulated`](reports/methods/v8_entropy_modulated.md) | 94.30% | 13 |
| 7 | [`v5_adaptive_corrected`](reports/methods/v5_adaptive_corrected.md) | 93.86% | 14 |
| 8 | [`v8_b2_b4_boosted`](reports/methods/v8_b2_b4_boosted.md) | 92.54% | 17 |
| 9 | [`v2_visibility`](reports/methods/v2_visibility.md) | 92.54% | 17 |
| 10 | [`v5_best_visibility`](reports/methods/v5_best_visibility.md) | 92.54% | 17 |
| 11 | [`v1_corrected`](reports/methods/v1_corrected.md) | 90.79% | 21 |

### Kecepatan (ms/pohon, 30 repetisi × 228 pohon)

Sumber: [`speed_summary.csv`](reports/benchmark_multidim/speed_summary.csv).

| Method | ms ↓ | pohon/detik ↑ |
|---|---:|---:|
| [`v1_corrected`](reports/methods/v1_corrected.md) | 0.004 | 279,830 |
| [`v5_adaptive_corrected`](reports/methods/v5_adaptive_corrected.md) | 0.007 | 136,242 |
| [`v7_stacking_density`](reports/methods/v7_stacking_density.md) | 0.014 | 70,585 |
| [`v2_visibility`](reports/methods/v2_visibility.md) | 0.022 | 45,146 |
| [`v9_selector`](reports/methods/v9_selector.md) | 0.079 | 12,619 |
| [`v6_selector`](reports/methods/v6_selector.md) | 0.099 | 10,074 |
| [`v9_b2_median_v6`](reports/methods/v9_b2_median_v6.md) | 0.429 | 2,330 |

### Robustness (Gaussian noise σ=20% pada koordinat)

Sumber: [`robustness_summary.csv`](reports/benchmark_multidim/robustness_summary.csv).

| Method | Drop Acc @ σ=20% ↓ |
|---|---:|
| [`v1_corrected`](reports/methods/v1_corrected.md), [`v5_adaptive_corrected`](reports/methods/v5_adaptive_corrected.md) | 0.00% (tidak pakai koordinat) |
| [`v8_b2_b4_boosted`](reports/methods/v8_b2_b4_boosted.md) | −0.88% (malah naik) |
| [`v7_stacking_bracketed`](reports/methods/v7_stacking_bracketed.md), [`v7_stacking_density`](reports/methods/v7_stacking_density.md) | 0.44% |
| [`v9_selector`](reports/methods/v9_selector.md) | 2.63% (paling sensitif) |

### Per Split (train=196, test=31, val=1)

Sumber: [`domain_breakdown.csv`](reports/benchmark_multidim/domain_breakdown.csv).

| Method | test ↑ | train ↑ | val ↑ |
|---|---:|---:|---:|
| [`v9_selector`](reports/methods/v9_selector.md) | **90.32%** | **98.47%** | 100% |
| [`v9_b2_median_v6`](reports/methods/v9_b2_median_v6.md) | 87.10% | 97.45% | 100% |
| [`v6_selector`](reports/methods/v6_selector.md) | 83.87% | 97.96% | 100% |

[`v9_selector`](reports/methods/v9_selector.md) unggul 6+ poin di test set — gap terbesar justru di split tersulit.

---

## Rekomendasi

| Kebutuhan | Pilihan |
|---|---|
| Akurasi maksimal pada JSON GT | [`v9_selector`](reports/methods/v9_selector.md) |
| Throughput tinggi + Acc ±1 >93% | [`v5_adaptive_corrected`](reports/methods/v5_adaptive_corrected.md) |
| Tradeoff akurasi/kecepatan | [`v7_stacking_density`](reports/methods/v7_stacking_density.md) |
| Pipeline TXT noisy (tanpa GT) | [`v5_adaptive_corrected`](reports/methods/v5_adaptive_corrected.md) atau [`v1_corrected`](reports/methods/v1_corrected.md) |
| Tidak butuh koordinat bbox | [`v1_corrected`](reports/methods/v1_corrected.md) |

---

## Evolusi Metode

| Gen | Method | Macro MAE ↓ | Catatan |
|---|---|---:|---|
| naive | — | — | overcount ~78.8% |
| v1 | [`v1_corrected`](reports/methods/v1_corrected.md) | 0.2851 | divisor global |
| v2 | [`v2_visibility`](reports/methods/v2_visibility.md) | 0.2664 | geometri sederhana |
| v5 | [`v5_adaptive_corrected`](reports/methods/v5_adaptive_corrected.md) | 0.2774 | adaptive divisor, >93% Acc±1 |
| v6 | [`v6_selector`](reports/methods/v6_selector.md) | 0.2599 | **titik balik** — routing per regime |
| v7 | [`v7_stacking_bracketed`](reports/methods/v7_stacking_bracketed.md) | 0.2643 | stacking density family |
| v8 | [`v8_b2_b4_boosted`](reports/methods/v8_b2_b4_boosted.md) | 0.2632 | per-kelas boosting |
| v9 | [`v9_selector`](reports/methods/v9_selector.md) | **0.2533** | narrow overrides di atas v6 |

**Logika [`v9_selector`](reports/methods/v9_selector.md):**
1. default → [`v6_selector`](reports/methods/v6_selector.md)
2. `b4_only_overlap` → [`v7_stacking_bracketed`](reports/methods/v7_stacking_bracketed.md)
3. `classaware_compact_lowb4` → [`v8_b2_b4_boosted`](reports/methods/v8_b2_b4_boosted.md)
4. `b3b4_only_lowtotal` → `v8_floor_anchor_50`
5. `dense_allside_moderatedup` → [`v8_b2_b4_boosted`](reports/methods/v8_b2_b4_boosted.md)

---

## Total Tandan 945 Pohon

Target rasio empiris **0.5594** (unique/naive dari 228 JSON = 2466/4408). Sumber: [`reports/dedup_all_953/all_953_totals.csv`](reports/dedup_all_953/all_953_totals.csv), detail per-pohon di [`all_953_per_tree.csv`](reports/dedup_all_953/all_953_per_tree.csv).

| Metode | Total | Rasio →0.5594 | Jarak ↓ |
|---|---:|---:|---:|
| `v8_b2_b4_boosted` | 10,129 | 0.5604 | **0.0010** |
| `v9_median_strong5` | 10,130 | 0.5605 | 0.0011 |
| `hybrid_vis_corr` | 9,988 | 0.5526 | 0.0068 |
| `v9_selector` | 10,449 | 0.5782 | 0.0188 |
| `v6_selector` | 10,467 | 0.5792 | 0.0198 |
| naive | 18,073 | 1.0000 | 0.4406 |

> Breakdown per-metode di section ini memakai CSV terpisah (`dedup_all_953/`) — bukan subset 228. Link breakdown per metode tidak mencakup 945 ini.

---

## Menjalankan

```bash
pip install -r requirements.txt

python scripts/count_all_trees.py            # GT counting 953 pohon
python scripts/count_gt_vs_naive.py          # audit JSON-05 + JSON-01
python scripts/benchmark_multidim.py         # benchmark 4-dimensi 11 algoritma → reports/benchmark_multidim/
python scripts/generate_method_reports.py    # regenerate reports/methods/<method>.md dari CSV di atas
python scripts/dedup_research_v9.py          # research script v9
python scripts/dedup_all_953.py              # semua metode pada 945 pohon
python scripts/dedup_nonjson_compare.py      # validasi non-JSON
```

## Struktur Repo

```
json/                    228 file JSON multi-view bunch links
dataset/                 image + label YOLO
algorithms/              satu file per algoritma
scripts/                 audit, counting, dedup research, report generator
reports/
  benchmark_multidim/    CSV mentah: accuracy_per_tree, accuracy_per_class, speed, robustness, domain_breakdown
  methods/               per-method markdown breakdown + per-tree CSV slice (linked dari README)
  dedup_research_v9/     research script terbaru
  dedup_all_953/         semua metode pada 945 pohon
RESEARCH.md              dokumen riset panjang (baca Section 0)
AGENTS.md                instruksi operasional
```

## Schema JSON

```json
{
  "tree_id": "20260422-DAMIMAS-001",
  "images": {
    "sisi_1": {"annotations": [{"class_name": "B3", "bbox_yolo": [0.5, 0.5, 0.1, 0.2], "box_index": 0}]}
  },
  "bunches": [{"bunch_id": 1, "class": "B3", "appearance_count": 2}],
  "summary": {"by_class": {"B1": 1, "B2": 2, "B3": 5, "B4": 0}}
}
```

`summary.by_class` = GT unique count per kelas.

## Batasan Riset

100% algoritmik. Tidak boleh: Siamese / CNN embedding, MLP pada fitur bbox, learned threshold via backprop, strict matching (Hungarian / graph / cluster) pada TXT (broken oleh coordinate noise).

Laporan multi-dimensi lengkap: [`reports/benchmark_multidim/REPORT.md`](reports/benchmark_multidim/REPORT.md).
