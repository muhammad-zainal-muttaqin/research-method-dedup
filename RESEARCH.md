# Research: Multi-View Oil Palm Fruit Detection & Classification
**Dataset: DAMIMAS | Ditulis: 2026-04-23 | Revisi: 2026-04-23 (v2: 2026-04-23)**

---

## 0. REVISI — Status & Reframe (BACA DULU)

> Dokumen ini awalnya ditulis greenfield (asumsi proyek baru). Faktanya proyek sudah punya banyak eksperimen sebelumnya yang mendokumentasikan plateau. Section 1–26 di bawah dipertahankan sebagai infra reference & dokumentasi konseptual, **tetapi banyak yang sudah tertutup secara empiris**. Section 0 ini override segala konflik.

### 0.1 Source of Truth Sekarang

Kalau ada konflik antara Section 1–26 dan dokumen di bawah → **ikuti dokumen ini**, bukan Section 1–26:

- `CONTEXT.md` (workspace)
- Ledger eksperimen aktif: `C:/Users/Zainal/Desktop/autoresearch/results.tsv`, `C:/Users/Zainal/Desktop/bbc-autoresearch-v1/experiments/results.tsv`
- Laporan formal E0: `C:/Users/Zainal/Desktop/bbc-autoresearch-v1/LAPORAN_EKSPERIMEN.md`
- Audit dataset: `D:/Work/Assisten Dosen/YOLOBench/analysis_dataset_640/...`

### 0.2 Baseline Aktif yang Harus Dilampaui

| Rejim | Run | mAP50-95 | Catatan |
|---|---|---:|---|
| Standard val | **AR29** YOLO11l 640 b16 | **0.264** | Active fair baseline |
| Upper-bound (train+test, bukan fair) | **AR34** YOLO11l 80 ep 640 b16 | **0.269** | Bukan benchmark fair |
| E0 final | `p3_final_yolo11s_s42` | 0.265 | Verdict: **INSUFFICIENT** |
| Single-class stage-1 (bukti detector mampu) | – | **0.390** | Task 1-kelas, bukan solusi 4-kelas |

Plateau seluruh rejim 4-kelas modern: **0.24–0.27 mAP50-95**.

> **Decision metric utama: `mAP50-95`.** Bukan `mAP@0.5` (Section 8.1 lama harus dibaca dengan ini in mind).

### 0.3 Bottleneck Struktural (Empirically Confirmed)

1. **B2/B3 ambiguity** — linear probe precision B2=0.394, B3=0.420; E0 confusion B2→B3 ≈34%. Hipotesis kuat: **label-ceiling**, bukan optimization.
2. **B4 small-object** — median rel_area 0.0072 (paling kecil); B4→background ≈42% di E0.
3. **Domain imbalance** — DAMIMAS ~90% image / ~94% instance; LONSUM minoritas (B1 LONSUM hanya 17 instance).
4. **Bbox quality** pada slice tersulit — audit shortlist: 3 DROP + 9 high-priority + 21 review.

### 0.4 Mapping Kelas Canonical (override Section 1.2 lama)

Dari `CONTEXT.md Section 1.7`:

| Kelas | Visual | Posisi | Kematangan |
|---|---|---|---|
| **B1** | Merah, besar, bulat | **Paling bawah** | **Paling matang** |
| **B2** | Mostly hitam, transisi ke merah, besar bulat | Atas B1 | Hampir matang |
| **B3** | Full hitam, masih berduri, lonjong | Atas B2 | Mengkal |
| **B4** | Paling kecil, paling dalam di tandan, berduri, hitam→hijau | **Paling atas** | **Paling belum matang** |

(Konsisten dengan Section 1.2 lama — tapi tegaskan ulang: urutan biologis `B1 → B4` = matang → belum matang.)

### 0.5 Section yang SUDAH TERTUTUP (Jangan Re-run tanpa Angle Struktural Baru)

| Section lama | Konten | Status faktual |
|---|---|---|
| Section 4.1 Section 4.2 Section 4.3 | Class weights, focal loss, naive oversampling B1/B4 | **CLOSED** — sudah dicoba, tidak menembus plateau |
| Section 5 (CORAL/ordinal) | Ordinal head, CORAL loss | **CLOSED** — DINOv2 CORN sudah dicoba di two-stage |
| Section 6.1 (WBF lintas view naif) | WBF basis IoU lintas view | Sanity baseline OK; bukan solusi 4-kelas |
| Section 9.2 Exp 1–6 | Knob tuning + imbalance recipe | **CLOSED** — dijawab AR29 series |
| Section 16 train script imbalance | Implementasi knob | Tetap valid sebagai utility, bukan jalur baru |
| Section 17 CORAL crop classifier | Two-stage classifier | **CLOSED** — DINOv2 CE/CORN, EfficientNet, hierarchical, wide-context — semua gagal |
| Section 18 KD teacher→student | Knowledge distillation | Layak hanya sebagai alat mobile export, **bukan** solusi 4-kelas ceiling |
| Section 20 Ablation exp01–exp10 | Matrix knob lama | Sebagian besar sudah dijawab; lihat **Section 30** untuk matrix baru |
| Section 22 Mobile export | TFLite INT8 pipeline | Tetap valid kapan pun ada model layak deploy |
| Section 25 Failure modes | Debug playbook knob | Tetap berguna sebagai referensi, bukan agenda eksperimen |

### 0.6 Section yang Sekarang Jadi HEADLINE

**[v2 UPDATE — 2026-04-23]** Arahan dosen (Bu Fatma): *"tidak usah end-to-end dulu. Inputnya (ground truth deteksi+klasifikasi)"*

Implikasi langsung:
- **Prioritas 1 sekarang: JSON-05** — counting pipeline multi-view dengan **GT label sebagai input** (bukan prediksi model)
- JSON-01 s/d JSON-04 (label audit + retrain path) **turun ke prioritas 2** — baru relevan setelah counting pipeline berjalan dan ada kebutuhan konkret meningkatkan detector
- Dataset yang dipakai: **228 pohon dengan JSON** (bukan 854 pohon penuh), karena hanya 228 yang sudah selesai di-link

**Section 23 — Multi-View Aggregation (JSON bunch-linking)** sebelumnya disebut "Stage 2 nanti". Sekarang **fokus utama v2**. Detail:

- **Section 29 JSON-05** — eksperimen counting (GT-based, no GPU, prioritas pertama)
- **Section 23** — pipeline MultiViewAggregator
- **Section 27** JSON sebagai Label-Audit Tool (defer, run setelah JSON-05)
- **Section 28** JSON sebagai Multi-View Supervision Signal (defer)
- **Section 30** Updated Ablation Matrix & Decision Tree

### 0.7 Aturan Perbandingan Hasil Baru

- **Wajib** breakdown per-domain (DAMIMAS vs LONSUM) dan per-class (B1/B2/B3/B4)
- **Wajib** bootstrap CI 95% terhadap AR29 (gap < 0.005 mAP50-95 = noise)
- **Wajib** specify rejim (standard val vs train+test vs legacy split — jangan campur)
- Klaim "improvement" tanpa CI overlap test = invalid

---

### 0.8 Quick Start — Baca Ini Sebelum Section 1–26 [v2]

> **Dokumen ini ~37K token. Section 1–26 adalah referensi infra, bukan agenda baru.**

#### Urutan Baca yang Benar (v2)

```
0.1–0.8  (sudah kamu baca)
    ↓
Section 29 JSON-05  (ACTION PERTAMA: counting pipeline, no GPU)
    ↓
Section 23          (MultiViewAggregator implementation detail)
    ↓
Section 30.2        (Decision Tree — setelah JSON-05 punya hasil)
    ↓
Section 27/28/29 JSON-01–04  (defer: label audit + retrain — baru kalau counting pipeline sudah jalan)
    ↓
Section 1–26  (buka hanya kalau perlu detail infra spesifik)
```

#### Action Pertama Sekarang: JSON-05 (GT-Based Counting)

**Arahan:** Input = ground truth label (bukan prediksi model). Tidak perlu retrain.  
**Dataset:** 228 pohon dengan JSON bunch-link.  
**Cost:** ~2 jam, tidak butuh GPU.  
**Tugas konkret:**
1. Untuk setiap pohon (dari 228 JSON), hitung **ground truth count per kelas** dari `summary.total_unique_bunches` per kelas
2. Bandingkan dengan **naive sum** (jumlah bbox dari semua 4 sisi tanpa dedup)
3. Hitung **Count MAE per kelas** — ini baseline counting pipeline

**Output:** `reports/json_05/count_mae_gt.csv` dengan kolom `tree_id, B1_gt, B2_gt, B3_gt, B4_gt, B1_naive, B2_naive, B3_naive, B4_naive, MAE_B1, ..., MAE_overall`  
**Decision rule:** lihat Section 29 JSON-05 dan Section 30.2

#### Peta Section (Fokus Utama v2)

| Section | Isi | Prioritas |
|---|---|---|
| **29 JSON-05** | Counting pipeline GT-based | **PERTAMA** |
| **23** | MultiViewAggregator implementation | **Setelah JSON-05** |
| **29 JSON-01** | Label audit cross-view | Defer |
| **29 JSON-02/03/04** | Retrain path (consensus, 3-class, consistency loss) | Defer |
| **30** | Decision tree + ablation matrix | Referensi navigasi |

#### Yang Tidak Perlu Dibuka Dulu

Section 3–22: arsitektur, augmentasi, imbalance, CORAL, KD, mobile export — sudah tertutup atau defer. Buka hanya kalau ada kebutuhan infra spesifik.

---

### 0.9 Scope JSON-05 (v2 — GT-Based Counting, Bukan End-to-End)

> Clarifikasi agar tidak scope-creep ke detection improvement:

**Yang DILAKUKAN di JSON-05:**
- Input: file JSON bunch-link (228 pohon) + label YOLO TXT original (GT)
- Task: hitung unique bunch count per kelas per pohon via JSON dedup
- Baseline: naive sum (semua bbox dari 4 sisi dijumlah tanpa dedup)
- Metric: Count MAE per kelas, Count MAE overall, per-pohon breakdown
- Laporan: visualisasi distribusi error, kelas mana yang overcounting paling parah

**Yang TIDAK dilakukan di JSON-05:**
- Tidak run model inference (tidak butuh GPU)
- Tidak modifikasi label / dataset
- Tidak bandingkan ke AR29 (beda task: counting bukan detection mAP)
- Tidak handle 626 pohon yang belum punya JSON (defer)

**Definition of Done JSON-05:**
- [ ] Script `scripts/count_gt_vs_naive.py` jalan tanpa error pada 228 JSON
- [ ] `reports/json_05/count_mae_gt.csv` tersimpan
- [ ] Summary: MAE naive per kelas (ini upper bound seberapa buruk overcounting tanpa dedup)
- [ ] Decision: apakah JSON dedup signifikan menurunkan MAE? → kalau yes, lanjut implement pipeline inference-based

---

## 1. Profil Dataset

### 1.1 Struktur Umum

| Item | Nilai |
|------|-------|
| Total pohon | 854 |
| Gambar per pohon | 4 sisi (sisi_1 sampai sisi_4) |
| Total gambar | 3,992 JPEG |
| Resolusi | 960 × 1280 px (portrait) |
| Device | Samsung SM-A566B |
| Format anotasi | YOLO TXT (normalized xywh) |
| Train / Val / Test | 2,780 / 620 / 592 |

### 1.2 Kelas Buah (Ordinal Maturity)

| ID | Nama | Warna Tandan | Ciri Khas | Posisi di Pohon | Waktu ke Panen |
|----|------|-------------|----------|----------------|----------------|
| 0  | B1   | **Kemerahan** (dominan merah) | Tandan besar, matang penuh | **Paling bawah** | ~1 bulan |
| 1  | B2   | **Setengah merah, masih ada hitam** | Transisi warna, campuran | Di atas B1 | ~2 bulan |
| 2  | B3   | **Sepenuhnya hitam** | Besar, gelap penuh | Di atas B2 | ~3 bulan |
| 3  | B4   | **Hitam kecil, ada duri tajam** | Buah muda, ukuran kecil, berduri | **Paling atas** | ~4 bulan |

> **Kritis:** B1–B4 adalah skala **ordinal**, bukan nominal. Jarak antar kelas bermakna: salah prediksi B1→B4 (selisih 3 bulan) 3× lebih berbahaya dari B1→B2 (selisih 1 bulan).

> **Implikasi visual penting:**
> - **Warna adalah diskriminator utama B1 vs B3/B4** — merah vs hitam sangat kontras, model seharusnya bisa membedakan ini dengan baik
> - **B2↔B3 paling sulit** — transisi warna merah→hitam, batas ambigu
> - **B4 unik dari ukuran dan tekstur** (kecil + berduri) tapi warnanya mirip B3
> - **Posisi vertikal** mengandung informasi implisit: B1 selalu paling bawah, B4 paling atas — bisa dieksploitasi sebagai spatial prior

### 1.3 Distribusi Kelas

| Kelas | Deteksi (Train) | Persentase | Rasio ke B3 |
|-------|----------------|-----------|-------------|
| B1    | 1,548          | 12.2%     | 0.26× |
| B2    | 2,895          | 22.9%     | 0.49× |
| B3    | 5,853          | 46.3%     | 1.00× |
| B4    | 2,347          | 18.6%     | 0.40× |
| **Total** | **12,643** | **100%** | — |

> **B3 dominan** karena mayoritas pohon yang difoto sedang dalam tahap 3 bulan pra-panen. Ini mencerminkan kondisi lapangan riil tapi menciptakan class imbalance.

### 1.4 Statistik Objek per Gambar

| Split | Gambar | Total Bbox | Avg Bbox/Gambar |
|-------|--------|-----------|----------------|
| Train | 2,780  | 12,643    | 4.55 |
| Val   | 620    | ~2,800    | ~4.5 |
| Test  | 592    | ~2,700    | ~4.5 |

### 1.5 Struktur JSON Bunch-Linking

228 dari 854 pohon (26.7%) memiliki JSON metadata yang menghubungkan bounding box yang sama terlihat dari beberapa sisi:

```
dataset_combined_1_yolo/json/  → 46 JSON (pohon 1–45)
dataset_combined_2_yolo/json/  → 72 JSON (pohon 244–576)
dataset_combined_3_yolo/json/  → 113 JSON (pohon 577–809)
```

Struktur JSON per pohon:
```json
{
  "tree_id": "DAMIMAS_A21B_0001",
  "images": {
    "sisi_1": {
      "filename": "DAMIMAS_A21B_0001_1.jpg",
      "annotations": [{"class": "B3", "yolo_bbox": [x, y, w, h]}, ...]
    },
    "sisi_2": {...},
    "sisi_3": {...},
    "sisi_4": {...}
  },
  "bunches": [
    {
      "bunch_id": 1,
      "class": "B3",
      "appearance_count": 2,
      "appearances": [
        {"side": "sisi_1", "box_index": 0},
        {"side": "sisi_4", "box_index": 1}
      ]
    }
  ],
  "summary": {
    "total_unique_bunches": 8,
    "total_detections": 17,
    "duplicates_linked": 9
  }
}
```

> **Insight penting:** 1 tandan sawit rata-rata terlihat dari 2+ sisi. Tanpa deduplication, count per pohon akan overcounting. JSON ground truth menyediakan "unique bunch count" yang akurat.

---

## 2. Formulasi Masalah

### 2.1 Task Definition

**Input:** Gambar tunggal (960×1280 JPEG) dari satu sisi pohon sawit  
**Output:** Sekumpulan bounding box, masing-masing dengan:
- Lokasi (x_center, y_center, width, height) — normalized
- Kelas (B1 / B2 / B3 / B4)
- Confidence score

**Tahap 1 (target sekarang):** Per-image object detection + classification  
**Tahap 2 (nanti):** Multi-view aggregation → count unik per kelas per pohon

### 2.2 Karakteristik Unik Problem Ini

1. **Ordinal labels** — B1 < B2 < B3 < B4 dalam skala kematangan. Standard cross-entropy tidak mengeksploitasi informasi urutan ini.

2. **Multi-view dari objek yang sama** — 4 foto per pohon dari sudut berbeda. Objek yang sama bisa muncul di 2–3 sisi. Ground truth JSON sudah meng-encode linking ini.

3. **High intra-class variance** — Pencahayaan outdoor, sudut kamera bervariasi, kondisi daun menutupi buah.

4. **Objek kecil di foto wide-angle** — Pohon sawit besar, buah relatif kecil dalam frame.

5. **Heavy class imbalance** — B3 mendominasi 46%, B1 hanya 12%.

---

## 3. Review Arsitektur yang Relevan

### 3.1 YOLO Variants

#### YOLOv8 (Ultralytics, 2023)
- **Kelebihan:** Paling mature, dokumentasi lengkap, komunitas besar, banyak tutorial
- **Kekurangan:** Sedikit kalah dari v11 untuk small objects
- **Ukuran tersedia:** n (3.2M param) / s (11.2M) / m (25.9M) / l (43.7M) / x (68.2M)
- **Rekomendasi size:** `yolov8m` untuk akurasi, `yolov8s` jika GPU terbatas

#### YOLOv11 (Ultralytics, 2024) ← **Direkomendasikan**
- **Kelebihan:** Lebih akurat dari v8 dengan param lebih sedikit, better small object detection
- **Kekurangan:** Lebih baru, resource komunitas belum sebanyak v8
- **Ukuran tersedia:** n / s / m / l / x
- **Rekomendasi size:** `yolo11m` untuk akurasi

#### YOLOv9 (2024, GELAN architecture)
- **Kelebihan:** GELAN (Generalized Efficient Layer Aggregation Network) — gradient flow lebih baik
- **Kekurangan:** Tidak di Ultralytics ecosystem, setup lebih rumit
- **Cocok jika:** Ingin eksperimen architecture, tapi butuh setup manual

#### RT-DETR (Real-Time Detection Transformer, 2023)
- **Kelebihan:** End-to-end transformer, no NMS post-processing, akurasi tinggi
- **Kekurangan:** Butuh lebih banyak GPU memory, lebih lambat dari YOLO
- **Cocok untuk:** Jika akurasi menjadi prioritas mutlak dan ada GPU yang kuat

### 3.2 Perbandingan untuk Dataset Ini

| Model | mAP COCO | Param | FPS (T4) | Mobile Export | Cocok? |
|-------|---------|-------|---------|--------------|-------|
| YOLOv8n | 37.3 | 3.2M | 140 | ✅ TFLite/ONNX | Kandidat mobile |
| YOLOv8s | 44.9 | 11.2M | 128 | ✅ | **Mobile + akurasi** |
| YOLOv8m | 50.2 | 25.9M | 82 | ⚠️ Berat untuk mobile | Train accuracy ref |
| YOLOv11n | 39.5 | 2.6M | 80 | ✅ | **Kandidat mobile terbaik** |
| YOLOv11s | 47.0 | 9.4M | 72 | ✅ | **Recommended mobile** |
| YOLOv11m | 51.5 | 20.1M | 68 | ⚠️ | Accuracy reference saja |
| RT-DETR-L | 53.0 | 32M | 75 | ❌ Tidak cocok mobile | Skip |

### 3.3 Strategi: Train Besar → Deploy Kecil (Knowledge Distillation)

Karena GPU unlimited tapi target adalah **mobile phone**, strategi terbaik:

```
STEP 1: Train teacher model (YOLOv11m) → akurasi maksimal
STEP 2: Train student model (YOLOv11n/s) dengan knowledge distillation
         Student belajar dari soft labels teacher, bukan hard labels saja
STEP 3: Export student → TFLite INT8 (Android) atau CoreML (iOS)
```

**Keuntungan dibanding langsung train YOLOv11n:**
- Student yang di-distill dari teacher yang baik ~3-5% mAP lebih tinggi
- Ukuran model tetap kecil untuk mobile

### 3.4 Mobile Export Pipeline

```
best.pt (PyTorch)
    │
    ├── ONNX (.onnx)           → Cross-platform, bisa di Android/iOS via ONNX Runtime
    ├── TFLite (.tflite)       → Android native (TensorFlow Lite)
    │     └── INT8 quantized   → 4× lebih kecil, sedikit loss akurasi
    └── CoreML (.mlmodel)      → iOS native

Ukuran estimasi YOLOv11n:
  - Float32 (PyTorch): ~5.4MB
  - INT8 quantized: ~1.4MB
  - Inference on Snapdragon 8: ~15-25ms per gambar
```

---

## 4. Strategi Handling Class Imbalance

### 4.1 Class-Weighted Loss

YOLO menggunakan BCE loss untuk classification. Bisa tambahkan weight:

```
Weight_inverse_frequency:
  B1: 12,643 / (4 × 1,548) = 2.04×
  B2: 12,643 / (4 × 2,895) = 1.09×
  B3: 12,643 / (4 × 5,853) = 0.54×
  B4: 12,643 / (4 × 2,347) = 1.35×
```

Normalize: B1=3.78, B2=2.02, B3=1.00, B4=2.50

### 4.2 Focal Loss

Focal Loss: `FL(p) = -α(1-p)^γ * log(p)`

- `γ=2.0` (default) mengurangi kontribusi easy examples
- B3 yang mudah diklasifikasi (banyak contoh) bobotnya berkurang otomatis
- B1 yang susah (sedikit contoh) mendapat bobot lebih besar

Built-in di YOLO via parameter `fl_gamma`. Rekomendasi: `fl_gamma: 1.5`

### 4.3 Oversampling Minor Class

Untuk B1 (hanya 12%): duplikat gambar yang mengandung B1 di training set.

```python
# Logika: scan train labels, jika ada kelas 0 (B1) → copy ke daftar oversample
images_with_B1 = [img for img in train_images if has_class(img, 0)]
# Oversample 2× agar B1 efektif ~24%
```

### 4.4 Augmentasi Spesifik: COLOR-AWARE (KRITIS)

> **Peringatan:** Karena warna (merah vs hitam) adalah fitur pembeda utama B1 vs B3, augmentasi warna yang terlalu agresif bisa **merusak label**. Hue jitter besar bisa membuat B1 (merah) terlihat seperti B3 (hitam) secara visual.

| Augmentasi | Parameter Aman | Tujuan | Catatan |
|-----------|---------------|--------|---------|
| Mosaic | ON | Variasi konteks, bantu minor class | Aman |
| MixUp | alpha=0.1 (rendah) | Soft labels, robustness | Hati-hati mixing B1+B3 |
| **HSV Hue** | `hsv_h: 0.015` (RENDAH) | Variasi pencahayaan | **JANGAN tinggi** — bisa ubah merah→hitam |
| **HSV Saturation** | `hsv_s: 0.4` | Variasi kejenuhan warna | Moderat, aman |
| **HSV Value** | `hsv_v: 0.4` | Variasi kecerahan | Moderat, aman |
| Random flip horizontal | ON | Sisi pohon bisa mirror | Aman |
| Random flip vertical | **OFF** | | **Jangan** — posisi vertikal B1(bawah)↔B4(atas) bermakna |
| Scale (zoom) | 0.5–1.5 | Handle small objects (B4 kecil) | Aman |
| Random erasing | p=0.2 | Simulasi occlusion daun | Aman |
| Color jitter (brightness) | 0.3 | Variasi pencahayaan lapangan | Moderat |

> **Kritis — Jangan lakukan:**
> - Hue jitter besar (hsv_h > 0.05) → B1 bisa jadi terlihat seperti B3
> - Vertical flip → B1 (yang harusnya bawah) muncul di atas seperti B4
> - Grayscale augmentation → hilangkan fitur warna sepenuhnya

---

## 4b. Eksploitasi Posisi Vertikal sebagai Spatial Prior

Karena B1 selalu di bawah dan B4 selalu di atas (secara biologis tandan sawit tumbuh dari atas ke bawah), **y_center dari bounding box mengandung informasi kelas implisit**.

### Analisis yang Perlu Dilakukan

Sebelum eksploitasi, verifikasi hipotesis ini di dataset:
```python
# Hitung rata-rata y_center per kelas dari semua train labels
# Expected: mean_y(B1) > mean_y(B3) > mean_y(B4)
# (y=1.0 = bawah gambar dalam koordinat image standar,
#  tapi YOLO origin = kiri atas, jadi bawah = y tinggi)
```

### Cara Mengeksploitasi Spatial Prior

**Opsi 1: Positional Bias di Loss (simple)**
- Tambahkan penalty jika B1 diprediksi di y_center rendah (atas gambar)
- Implementasi: custom loss term

**Opsi 2: Y-coordinate sebagai auxiliary feature**
- Concat y_center bbox ke classification feature vector sebelum softmax
- Model belajar bahwa y_center besar → likely B1

**Opsi 3: Tidak lakukan — biarkan model belajar sendiri**
- YOLO melihat spatial context lewat receptive field
- Mungkin sudah cukup tanpa explicit prior

> **Rekomendasi:** Opsi 3 dulu (baseline). Analisis apakah confusion matrix B1↔B4 tinggi — jika iya, coba Opsi 2.

---

## 5. Ordinal Classification: Pendekatan Alternatif

### 5.1 Masalah Standard Softmax untuk Ordinal Data

Standard YOLO classification head menggunakan cross-entropy:
```
L = -Σ y_i * log(p_i)
```

Ini memperlakukan B1→B4 sama dengan B1→B2. Untuk ordinal data ini suboptimal.

### 5.2 CORAL Loss (Consistent Rank Logits)

Dekomposisi ordinal ke binary tasks:
```
P(Y >= 1), P(Y >= 2), P(Y >= 3)
```

Loss:
```
L_CORAL = -Σ_k [1_{y>=k} * log(σ(s_k)) + 1_{y<k} * log(1-σ(s_k))]
```

Implementasi: tambahkan ordinal head setelah backbone YOLO, training loss = bbox_loss + CORAL_loss.

### 5.3 Weighted MSE pada Class Index

Paling sederhana: treat class index (0,1,2,3) sebagai nilai kontinu, gunakan MSE regression head:
```
L_ordinal = (predicted_index - true_index)^2
```

Misalkan B1=0, B4=3 → error B1→B4 = 9, B1→B2 = 1. Naturally weighted.

### 5.4 Rekomendasi

| Tahap | Approach |
|-------|---------|
| Baseline | Standard YOLO + cross-entropy |
| Experiment 1 | Tambah `fl_gamma` + class weights |
| Experiment 2 | Custom CORAL head (jika B1↔B4 confusion masih tinggi) |

---

## 6. Handling Multi-View (Tahap 2, Nanti)

### 6.1 Cross-View Deduplication: Weighted Box Fusion (WBF)

Setelah per-image inference, jalankan WBF untuk merge prediksi dari 4 sisi:

```
Input:  4 set boxes {B1_side1, B1_side2, B1_side3, B1_side4}
Process: IoU-based clustering → vote → merge
Output: 1 set boxes unik per pohon
```

Library: `pip install ensemble-boxes`

**Masalah:** IoU lintas view tidak bisa langsung — bounding box di sisi_1 dan sisi_3 adalah objek yang sama tapi koordinat berbeda (sudut kamera beda). Solusi:
- Gunakan visual feature similarity (embedding YOLO backbone) bukan IoU koordinat
- Atau gunakan JSON ground truth untuk supervisi linking (jika ada)

### 6.2 Evaluation Menggunakan JSON Ground Truth

JSON menyediakan ground truth "unique bunch count". Bisa evaluasi:

```
Metric: Count Accuracy per pohon per kelas
  Ground truth (JSON): {B1: 3, B2: 4, B3: 8, B4: 2}
  Prediction (setelah dedup): {B1: 2, B2: 5, B3: 9, B4: 2}
  Error: MAE per class
```

### 6.3 Struktur Pipeline Tahap 2

```
4 gambar pohon X
    │
    ├─→ YOLO inference → boxes_sisi_1
    ├─→ YOLO inference → boxes_sisi_2
    ├─→ YOLO inference → boxes_sisi_3
    └─→ YOLO inference → boxes_sisi_4
                │
                ↓
    Cross-view deduplication
    (WBF atau feature similarity)
                │
                ↓
    Unique bunch count per class
    {B1: N1, B2: N2, B3: N3, B4: N4}
                │
                ↓
    Evaluasi vs JSON ground truth
```

---

## 7. Tantangan Teknis yang Diantisipasi

### 7.1 Small Object Detection

Buah sawit di foto wide-angle bisa sangat kecil. Strategi:

| Strategi | Detail |
|---------|--------|
| Tingkatkan `imgsz` | Default 640 → coba 1280. Memory 4× lebih besar. |
| SAHI (Sliced Inference) | Bagi gambar jadi tile kecil, inference tiap tile, merge |
| Multi-scale training | `--multi_scale True` di YOLO |
| Anchor tuning | Jalankan `autoanchor` untuk dataset ini |

### 7.2 Occlusion

Tandan sawit sering terhalang daun atau tandan lain:

- **Augmentasi occlusion:** Random erasing di training
- **Deformable convolutions:** Lebih robust ke partial occlusion
- **Keypoint-based approach:** Jika bbox tidak reliable, coba center-point detection (CenterNet)

### 7.3 Intra-Class Variance Tinggi

B2 dan B3 mungkin visually mirip (warna dan tekstur overlap):

- **Hard negative mining:** Perhatikan confusion B2↔B3 di training
- **Pretrained backbone lebih kuat:** Fine-tune dari model yang dilatih di data buah/agricultural
- **Color space augmentation:** HSV jitter untuk handle variasi pencahayaan outdoor

### 7.4 Dataset Size Relatif Kecil

3,992 gambar cukup untuk fine-tuning tapi kecil untuk training from scratch:

- **Wajib:** Start dari pretrained weights (COCO-pretrained YOLO)
- **Transfer learning:** Freeze backbone layers pertama, train head saja di awal
- **Data augmentation agresif:** Mosaic, mixup, flip, color jitter wajib ON

---

## 8. Evaluation Framework

### 8.1 Metrics Tahap 1 (Per-Image)

| Metric | Keterangan | Target |
|--------|-----------|--------|
| mAP@0.5 | Standard YOLO metric | > 0.80 |
| mAP@0.5:0.95 | Lebih ketat | > 0.60 |
| mAP per class | B1, B2, B3, B4 terpisah | Gap B1-B3 < 15% |
| Precision per class | Berapa deteksi yang benar | > 0.75 semua |
| Recall per class | Berapa ground truth yang ketangkap | > 0.75 semua |
| **MAE class index** | Mean absolute error ordinal: \|pred_idx - true_idx\| | < 0.5 |
| **B1↔B4 confusion rate** | Seberapa sering prediksi meleset 3 level | < 5% |

### 8.2 Confusion Matrix Analysis

Confusion matrix untuk dataset ordinal harus diinterpretasikan berbeda:
```
Ideal:           Acceptable:         Buruk:
B1 B2 B3 B4     B1 B2 B3 B4        B1 B2 B3 B4
90  8  2  0     85 12  3  0        70  5  5 20  ← B1 ke B4 tinggi
 7 85  7  1      8 80 12  0         5 60  5 30
 2  8 85  5      3 10 78  9         5  5 60 30
 0  1  6 93      0  0  8 92         5 20 20 55
```

### 8.3 Metrics Tahap 2 (Per-Pohon)

| Metric | Keterangan |
|--------|-----------|
| Count MAE per class | \|predicted_count - true_count\| per B1-B4 |
| Total count accuracy | % pohon dengan total count error ≤ 1 |
| Dedup precision | Berapa unique bunches yang benar dideteksi |

---

## 9. Eksperimen yang Direkomendasikan

### 9.1 Training Config (GPU Unlimited)

```yaml
# ===== TEACHER MODEL (accuracy reference) =====
model: yolo11m.pt
data: dataset_combined/data.yaml
epochs: 150
imgsz: 1280          # High res — penting untuk B4 yang kecil
batch: 32            # GPU unlimited, pakai besar
lr0: 0.01
lrf: 0.01
cos_lr: true
warmup_epochs: 3
mosaic: 1.0
mixup: 0.1
close_mosaic: 15
hsv_h: 0.015         # RENDAH — warna kritis (merah B1 vs hitam B3)
hsv_s: 0.4
hsv_v: 0.4
flipud: 0.0          # MATIKAN — posisi vertikal B1↔B4 bermakna
fliplr: 0.5
scale: 0.5
erasing: 0.2
fl_gamma: 1.5        # Focal loss untuk class imbalance
cls: 0.5

# ===== STUDENT MODEL (target mobile deployment) =====
model: yolo11n.pt    # atau yolo11s.pt untuk akurasi lebih baik
# Semua hyperparameter sama, tapi tambah:
epochs: 200          # Student butuh lebih lama konvergen
# + knowledge distillation dari teacher best.pt
```

### 9.2 Urutan Eksperimen

```
Exp 1: YOLOv11m Teacher Baseline
  - imgsz=1280, batch=32, epoch=150
  - Default augment + hsv_h rendah + flipud=0
  - Tujuan: raih akurasi maksimal, lihat confusion matrix

Exp 2: Teacher + Imbalance Handling
  - Tambah fl_gamma=1.5
  - Class weights: B1=3.78, B2=2.02, B3=1.0, B4=2.50
  - Oversampling gambar dengan B1 (×2)
  - Tujuan: perbaiki mAP B1

Exp 3: Student YOLOv11n (Knowledge Distillation)
  - Gunakan soft labels dari Exp 2 teacher
  - epoch=200, imgsz=1280
  - Tujuan: model kecil dengan akurasi mendekati teacher

Exp 4: SAHI Inference (eval only, tanpa retrain)
  - Load best student dari Exp 3
  - Inference dengan SAHI tile 640×640 stride 320
  - Tujuan: lihat apakah B4 (kecil) detection naik

Exp 5 (opsional): CORAL Ordinal Head
  - Custom YOLOv11 + CORAL loss pada classification head
  - Evaluasi MAE ordinal metric vs Exp 2
  - Lakukan hanya jika B1↔B4 confusion masih > 5%

Exp 6: Mobile Export & Benchmark
  - Export Exp 3 student → TFLite INT8
  - Benchmark di device Android: latency, memory, akurasi
```

### 9.3 Ablation Study Design

| Faktor | Nilai yang diuji | Metrik utama |
|--------|----------------|-------------|
| Model size | yolo11n vs yolo11s vs yolo11m | mAP + size |
| imgsz | 640 vs 1280 | mAP B4 (small) |
| hsv_h | 0.015 vs 0.1 vs 0.5 | Confusion B1↔B3 |
| flipud | ON vs **OFF** | Confusion B1↔B4 |
| Imbalance strategy | None vs focal vs oversample vs keduanya | mAP B1 |
| Loss function | Cross-entropy vs CORAL | MAE ordinal |
| Knowledge distillation | Tanpa vs dengan teacher | mAP student |

---

## 10. Referensi Penelitian Relevan

1. **Redmon et al. (2016)** — YOLOv1: You Only Look Once, real-time object detection
2. **Jocher et al. (2023)** — YOLOv8: Ultralytics, state-of-the-art detection
3. **Wang et al. (2024)** — YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information
4. **Cao et al. (2020)** — Rank consistent ordinal regression for neural networks with application to age estimation
5. **Bochkovskiy et al. (2020)** — YOLOv4: Optimal Speed and Accuracy of Object Detection
6. **Lin et al. (2020)** — Focal Loss for Dense Object Detection (RetinaNet)
7. **Sohan et al. (2023)** — A Review on YOLOv8 and Its Advancements
8. **Akiva et al. (2020)** — Finding Berries: Segmentation and Counting of Cranberries using Point Supervision and Shape Priors (relevan: counting small fruits)
9. **Sa et al. (2016)** — DeepFruits: A Fruit Detection System Using Deep Neural Networks (relevan: fruit detection domain)
10. **Zeng et al. (2023)** — SAHI: Slicing Aided Hyper Inference for small object detection

---

## 11. Roadmap Kerja

```
Fase A — Teacher Training (GPU unlimited, akurasi maksimal)
  ├── Setup: Python 3.10+, PyTorch 2.x, Ultralytics pip install
  ├── Train YOLOv11m teacher:
  │     imgsz=1280, batch=32, epoch=150
  │     hsv_h=0.015, flipud=0, fl_gamma=1.5
  ├── Analisis: confusion matrix, mAP per class, MAE ordinal
  └── Decision: apakah B1↔B4 confusion > 5%? → tentukan perlu CORAL

Fase B — Optimization Teacher
  ├── Implement oversampling B1 (×2) di train set
  ├── Experiment class weights jika focal loss belum cukup
  ├── Jika B1↔B4 masih tinggi: tambah CORAL loss head
  └── Simpan best_teacher.pt

Fase C — Student Distillation (target mobile)
  ├── Train YOLOv11n student dengan soft labels dari best_teacher.pt
  ├── epoch=200, imgsz=1280
  ├── Evaluasi: mAP student vs teacher (target gap < 5%)
  └── Simpan best_student.pt

Fase D — Mobile Export & Benchmark
  ├── Export best_student.pt → ONNX → TFLite INT8
  ├── Test di Android device: latency, RAM, akurasi
  ├── Bandingkan: Float32 vs INT8 (akurasi tradeoff)
  └── Final model untuk deployment

Fase E — Multi-View Pipeline (Tahap 2, nanti)
  ├── Load best_student.pt
  ├── Inference 4 sisi per pohon → 4 prediction sets
  ├── Cross-view deduplication (WBF atau feature similarity)
  ├── Hitung unique bunch count per class per pohon
  └── Evaluasi vs JSON ground truth (228 pohon)
```

---

## 12. Status Pertanyaan Terbuka

| # | Pertanyaan | Status | Jawaban |
|---|-----------|--------|---------|
| 1 | Definisi visual B1-B4? | ✅ TERJAWAB | B1=merah (bawah), B2=setengah merah, B3=hitam penuh, B4=hitam kecil berduri (atas) |
| 2 | JSON coverage cukup? | ⚠️ PERLU KEPUTUSAN | 26.7% coverage — evaluasi Tahap 2 hanya valid untuk 228 pohon tersebut |
| 3 | GPU available? | ✅ TERJAWAB | **Unlimited** → pakai imgsz=1280, batch=32, model besar |
| 4 | Target deployment? | ✅ TERJAWAB | **Mobile phone** → strategy: train besar (teacher) + distill ke yolo11n (student) |
| 5 | Data tambahan untuk B1? | ❓ BELUM | Apakah bisa foto lebih banyak pohon yang B1-nya dominan? |

### Keputusan Desain Final

| Aspek | Keputusan |
|-------|----------|
| **Teacher model** | YOLOv11m, imgsz=1280, epoch=150 |
| **Student/deploy model** | YOLOv11n, knowledge distillation, export TFLite INT8 |
| **Augmentasi warna** | hsv_h=0.015 (sangat rendah), flipud=OFF |
| **Imbalance** | Focal loss (γ=1.5) + oversampling B1 ×2 |
| **Loss function** | Cross-entropy baseline → CORAL jika B1↔B4 confusion > 5% |
| **Spatial prior** | Biarkan model belajar sendiri (verifikasi setelah baseline) |

---

## 13. Insight Khusus: Color-Based Feature Engineering

Karena B1 (merah) vs B3/B4 (hitam) sangat kontras secara warna, ada potensi untuk:

### 13.1 Pre-training pada Agricultural Color Dataset
- Fine-tune dari checkpoint yang sudah dilatih di data buah/tanaman
- Backbone sudah sensitif terhadap warna buah → konvergen lebih cepat
- Kandidat: model YOLO yang dilatih di dataset manggis, tomat, atau palm oil lainnya

### 13.2 Color Channel Attention
- Tambahkan channel attention yang fokus pada channel R (red) untuk membantu deteksi B1
- Channel R tinggi = likely B1 atau B2
- Bisa diimplementasikan sebagai lightweight attention module setelah input

### 13.3 HSV Color Space Input (Eksperimental)
- Konversi gambar ke HSV sebelum masuk model
- Channel H (Hue) langsung encode merah vs hitam
- Pro: informasi warna lebih eksplisit
- Con: YOLO pretrained di RGB, perlu fine-tune dari awal atau adaptor layer

### 13.4 Dual-Stream: RGB + Color Mask (Kompleks)
- Stream 1: RGB input → YOLO backbone biasa
- Stream 2: Red-mask (pixel merah di foto) → auxiliary feature
- Merge sebelum detection head
- Hanya relevan jika baseline sangat rendah untuk B1

---

# Bagian II: Implementasi

Section 14–26 berisi rincian eksekusi: struktur project, kode siap-pakai, workflow ablation step-by-step, mobile export, dan stage 2 multi-view. Snippet kode di-tulis level "tinggal copy lalu sesuaikan path", bukan production-grade — tetap fokus riset.

---

## 14. Environment & Project Structure

### 14.1 Hardware & OS

| Item | Spek Asumsi |
|------|-------------|
| GPU | 1× NVIDIA A100 / RTX 4090 / 3090 (24GB+) — minimum 16GB untuk imgsz=1280 batch=16 |
| RAM | 32GB+ |
| Disk | 50GB free (dataset + checkpoints + ablation outputs) |
| OS | Linux Ubuntu 22.04 (rekomendasi) atau Windows 11 + WSL2 |
| CUDA | 12.1+ |
| Python | 3.10 atau 3.11 |

### 14.2 `requirements.txt`

```txt
# Core deep learning
torch==2.3.0
torchvision==0.18.0
ultralytics==8.3.0

# Detection / multi-view
ensemble-boxes==1.0.9
sahi==0.11.18

# Quantization & export
onnx==1.16.0
onnxruntime-gpu==1.18.0
onnx-simplifier==0.4.36
tensorflow==2.16.1
coremltools==7.2

# Data / metrics / logging
numpy==1.26.4
pandas==2.2.2
scikit-learn==1.5.0
matplotlib==3.9.0
seaborn==0.13.2
opencv-python==4.10.0.84
Pillow==10.3.0
PyYAML==6.0.1
tqdm==4.66.4
wandb==0.17.0
albumentations==1.4.10
```

### 14.3 Setup Commands

```bash
# Conda env
conda create -n damimas python=3.10 -y
conda activate damimas

# Torch dengan CUDA 12.1
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121

# Sisanya
pip install -r requirements.txt

# Verifikasi GPU
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 14.4 Project Tree

```
damimas-yolo/
├── configs/
│   ├── exp01_baseline_v8m_640.yaml
│   ├── exp02_baseline_v11m_640.yaml
│   ├── exp03_v11m_1280.yaml
│   ├── exp04_v11m_1280_focal.yaml
│   ├── exp05_v11m_1280_focal_weights.yaml
│   ├── exp06_v11m_1280_focal_weights_oversample.yaml
│   ├── exp07_v11m_1280_full_imbalance_coral.yaml
│   ├── exp08_student_v11n_kd.yaml
│   ├── exp09_student_v11s_kd.yaml
│   └── exp10_student_v11n_sahi_eval.yaml
├── data/
│   └── dataset_combined/        # symlink ke D:/Work/.../dataset_combined
├── scripts/
│   ├── verify_dataset.py
│   ├── dataset_stats.py
│   ├── oversample_minor.py
│   ├── train_teacher.py
│   ├── train_student_kd.py
│   ├── eval_full.py
│   ├── sahi_inference.py
│   ├── export_mobile.py
│   ├── eval_multiview.py
│   └── run_ablation.py
├── src/
│   ├── __init__.py
│   ├── seed.py
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── coral.py
│   │   └── focal_weighted.py
│   ├── heads/
│   │   ├── __init__.py
│   │   └── coral_head.py
│   ├── trainers/
│   │   ├── __init__.py
│   │   └── kd_trainer.py
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── ordinal.py
│   │   └── confusion.py
│   └── pipeline/
│       ├── __init__.py
│       └── multiview_count.py
├── runs/                        # Ultralytics auto-output
├── exports/                     # ONNX / TFLite / CoreML
├── reports/                     # ablation_summary.csv, plots
├── requirements.txt
└── README.md
```

### 14.5 Reproducibility — `src/seed.py`

```python
import os, random
import numpy as np
import torch

def seed_everything(seed: int = 42, deterministic: bool = False):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    else:
        torch.backends.cudnn.benchmark = True
```

> **Catatan:** Ultralytics punya argumen `deterministic=True` di `model.train()`. AMP + multi-worker dataloader tetap punya non-determinisme kecil — accept gap <0.5% mAP antar run sebagai noise.

---

## 15. Data Preparation Scripts

### 15.1 `scripts/verify_dataset.py`

```python
"""Cek konsistensi dataset_combined: jumlah image vs label, parse data.yaml,
distribusi kelas per split."""
from pathlib import Path
import yaml
from collections import Counter

ROOT = Path("data/dataset_combined")
SPLITS = ["train", "val", "test"]

def main():
    cfg = yaml.safe_load((ROOT / "data.yaml").read_text())
    names = cfg["names"]
    print(f"Classes: {names}")
    for split in SPLITS:
        imgs = sorted((ROOT / "images" / split).glob("*.jpg"))
        lbls = sorted((ROOT / "labels" / split).glob("*.txt"))
        assert len(imgs) == len(lbls), f"{split}: {len(imgs)} img != {len(lbls)} lbl"
        # validasi paired stem
        img_stems = {p.stem for p in imgs}
        lbl_stems = {p.stem for p in lbls}
        assert img_stems == lbl_stems, f"{split}: stem mismatch"
        # distribusi kelas
        cnt = Counter()
        for lbl in lbls:
            for line in lbl.read_text().strip().splitlines():
                cls = int(line.split()[0])
                cnt[cls] += 1
        total = sum(cnt.values())
        print(f"\n[{split}] images={len(imgs)} bboxes={total}")
        for c in sorted(cnt):
            print(f"  {names[c]}: {cnt[c]} ({100*cnt[c]/total:.1f}%)")

if __name__ == "__main__":
    main()
```

### 15.2 `scripts/dataset_stats.py`

```python
"""Hitung statistik: bbox area per class, y_center per class (verifikasi spatial
prior B1 bawah / B4 atas), aspect ratio."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("data/dataset_combined")
NAMES = ["B1", "B2", "B3", "B4"]
OUT = Path("reports"); OUT.mkdir(exist_ok=True)

def parse_split(split):
    rows = []
    for lbl in (ROOT / "labels" / split).glob("*.txt"):
        for line in lbl.read_text().strip().splitlines():
            c, xc, yc, w, h = line.split()
            rows.append({
                "split": split, "class": NAMES[int(c)],
                "xc": float(xc), "yc": float(yc),
                "w": float(w), "h": float(h),
                "area": float(w) * float(h),
                "ar": float(w) / max(float(h), 1e-6),
            })
    return rows

def main():
    rows = []
    for s in ["train", "val", "test"]:
        rows += parse_split(s)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "bbox_stats.csv", index=False)

    # ===== Verifikasi spatial prior =====
    # YOLO origin = top-left, yc=0 atas, yc=1 bawah
    # Hipotesis: yc(B1) > yc(B4)
    print("\n=== Mean y_center per class (1=bawah gambar) ===")
    print(df.groupby("class")["yc"].agg(["mean", "median", "std"]))

    # ===== Bbox area per class =====
    print("\n=== Mean bbox area (normalized, kecil = small object) ===")
    print(df.groupby("class")["area"].agg(["mean", "median", "min", "max"]))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for cls in NAMES:
        sub = df[df["class"] == cls]
        axes[0].hist(sub["yc"], bins=40, alpha=0.5, label=cls, density=True)
        axes[1].hist(np.log10(sub["area"] + 1e-6), bins=40, alpha=0.5, label=cls, density=True)
    axes[0].set_title("y_center distribution per class"); axes[0].legend(); axes[0].set_xlabel("y_center (0=top, 1=bottom)")
    axes[1].set_title("log10(bbox area) per class"); axes[1].legend(); axes[1].set_xlabel("log10(area)")
    plt.tight_layout(); plt.savefig(OUT / "spatial_size_priors.png", dpi=140)
    print(f"\nSaved → {OUT / 'spatial_size_priors.png'}")

if __name__ == "__main__":
    main()
```

**Decision rule:** Jika `mean_yc(B1) - mean_yc(B4) > 0.15` → spatial prior nyata, pertimbangkan eksploitasi (Section 4b Opsi 2). Jika gap kecil → biarkan model belajar implisit.

### 15.3 `scripts/oversample_minor.py`

```python
"""Oversample images yang mengandung kelas minor (default: B1, kelas 0).
Output ke dataset baru agar split asli tidak rusak."""
from pathlib import Path
import shutil
import argparse

SRC = Path("data/dataset_combined")
DST = Path("data/dataset_combined_oversampled")

def has_class(label_path: Path, target_cls: int) -> bool:
    for line in label_path.read_text().strip().splitlines():
        if int(line.split()[0]) == target_cls:
            return True
    return False

def main(target_cls: int = 0, factor: int = 2):
    DST.mkdir(parents=True, exist_ok=True)
    # Copy data.yaml dengan path direvisi
    yaml_text = (SRC / "data.yaml").read_text().replace(
        str(SRC.resolve()), str(DST.resolve())
    )
    (DST / "data.yaml").write_text(yaml_text)

    for split in ["train", "val", "test"]:
        for sub in ["images", "labels"]:
            (DST / sub / split).mkdir(parents=True, exist_ok=True)
        # Copy semua file
        for img in (SRC / "images" / split).glob("*.jpg"):
            shutil.copy2(img, DST / "images" / split / img.name)
            lbl_src = SRC / "labels" / split / (img.stem + ".txt")
            shutil.copy2(lbl_src, DST / "labels" / split / lbl_src.name)
        # Oversample HANYA di train
        if split != "train":
            continue
        n_added = 0
        for img in (SRC / "images" / split).glob("*.jpg"):
            lbl = SRC / "labels" / split / (img.stem + ".txt")
            if has_class(lbl, target_cls):
                for k in range(1, factor):
                    new_stem = f"{img.stem}_aug{k}"
                    shutil.copy2(img, DST / "images" / split / f"{new_stem}.jpg")
                    shutil.copy2(lbl, DST / "labels" / split / f"{new_stem}.txt")
                    n_added += 1
        print(f"[{split}] oversample +{n_added} (target_cls={target_cls}, factor={factor})")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--target_cls", type=int, default=0)  # 0=B1
    p.add_argument("--factor", type=int, default=2)
    args = p.parse_args()
    main(args.target_cls, args.factor)
```

> **Catatan:** Oversample dengan duplikasi file (bukan augmentasi on-the-fly tambahan). Augmentasi mosaic+mixup dari YOLO trainer tetap aktif → tiap epoch duplikat terlihat berbeda secara efektif.

### 15.4 Stratified sanity check

```python
# Dijalankan setelah verify_dataset.py — pastikan val & test contain semua 4 kelas
# (Kalau ada split tanpa kelas tertentu, mAP per class jadi NaN)
```

Cukup pakai output `verify_dataset.py` Section 15.1; jika ada kelas dengan count=0 di val/test → re-split manual.

---

## 16. Training — Teacher (Stage 1)

### 16.1 `data.yaml` Template

```yaml
# data/dataset_combined/data.yaml (atau dataset_combined_oversampled/)
path: D:/Work/Assisten Dosen/Folder Linked Dataset/dataset_combined
train: images/train
val: images/val
test: images/test
nc: 4
names: [B1, B2, B3, B4]
```

### 16.2 Class Weights — Compute

```python
# src/losses/focal_weighted.py
import numpy as np

def inverse_frequency_weights(counts: dict, normalize_to: str = "B3"):
    """counts = {'B1': 1548, 'B2': 2895, 'B3': 5853, 'B4': 2347}
    Return dict {class: weight} dinormalisasi terhadap normalize_to (=1.0)."""
    n_total = sum(counts.values())
    n_cls = len(counts)
    raw = {c: n_total / (n_cls * v) for c, v in counts.items()}
    base = raw[normalize_to]
    return {c: w / base for c, w in raw.items()}

# Hasil utk dataset DAMIMAS (train):
# {'B1': 3.78, 'B2': 2.02, 'B3': 1.00, 'B4': 2.49}
```

### 16.3 `scripts/train_teacher.py`

```python
"""Training teacher YOLOv11m. Config diparse dari yaml di configs/.
Mendukung: focal loss, class weights (via monkey-patch), oversampled dataset path,
custom callback log per-class mAP & MAE ordinal."""
import argparse, yaml, json
from pathlib import Path
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.metrics import ConfusionMatrix
import sys; sys.path.insert(0, ".")
from src.seed import seed_everything

NAMES = ["B1", "B2", "B3", "B4"]

def patch_class_weights(model, weights_tensor):
    """Override BCE pos_weight di v8DetectionLoss untuk classification."""
    from ultralytics.utils.loss import v8DetectionLoss
    orig_init = v8DetectionLoss.__init__
    def new_init(self, model_):
        orig_init(self, model_)
        self.bce = torch.nn.BCEWithLogitsLoss(
            pos_weight=weights_tensor.to(self.device), reduction="none"
        )
    v8DetectionLoss.__init__ = new_init

def on_val_end_callback(validator):
    """Hitung MAE ordinal & B1↔B4 confusion dari confusion matrix Ultralytics."""
    cm = validator.confusion_matrix.matrix  # (nc+1, nc+1) — kolom terakhir/baris = background
    cm = cm[:4, :4]  # ambil hanya kelas (drop background)
    if cm.sum() == 0:
        return
    # MAE ordinal: |i - j| weighted by cm[i,j]
    indices = np.arange(4)
    mae = 0.0; total = 0
    for i in range(4):
        for j in range(4):
            mae += abs(i - j) * cm[i, j]
            total += cm[i, j]
    mae /= max(total, 1)
    # B1↔B4 confusion: salah ≥3 level
    b14 = (cm[0, 3] + cm[3, 0]) / max(total, 1)
    print(f"\n[ordinal] MAE_class_index={mae:.3f}  B1↔B4_rate={b14:.4f}")
    out_dir = Path(validator.save_dir)
    (out_dir / "ordinal_metrics.json").write_text(json.dumps(
        {"mae_ordinal": float(mae), "b1_b4_rate": float(b14)}, indent=2))

def main(cfg_path: str):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    seed_everything(cfg.get("seed", 42))

    # Class weights (opsional)
    if cfg.get("use_class_weights", False):
        cw = cfg["class_weights"]  # dict {B1: 3.78, ...}
        w_tensor = torch.tensor([cw[n] for n in NAMES], dtype=torch.float32)
        patch_class_weights(None, w_tensor)
        print(f"Patched class weights: {w_tensor.tolist()}")

    model = YOLO(cfg["model"])
    model.add_callback("on_val_end", on_val_end_callback)

    results = model.train(
        data=cfg["data"],
        epochs=cfg["epochs"],
        imgsz=cfg["imgsz"],
        batch=cfg["batch"],
        lr0=cfg.get("lr0", 0.01),
        lrf=cfg.get("lrf", 0.01),
        cos_lr=cfg.get("cos_lr", True),
        warmup_epochs=cfg.get("warmup_epochs", 3),
        mosaic=cfg.get("mosaic", 1.0),
        mixup=cfg.get("mixup", 0.1),
        close_mosaic=cfg.get("close_mosaic", 15),
        hsv_h=cfg.get("hsv_h", 0.015),
        hsv_s=cfg.get("hsv_s", 0.4),
        hsv_v=cfg.get("hsv_v", 0.4),
        flipud=cfg.get("flipud", 0.0),
        fliplr=cfg.get("fliplr", 0.5),
        scale=cfg.get("scale", 0.5),
        erasing=cfg.get("erasing", 0.2),
        cls=cfg.get("cls", 0.5),
        fl_gamma=cfg.get("fl_gamma", 0.0),  # 0=disable, 1.5=focal
        project=cfg.get("project", "runs/detect"),
        name=cfg["exp_id"],
        device=cfg.get("device", 0),
        amp=cfg.get("amp", True),
        deterministic=cfg.get("deterministic", False),
        seed=cfg.get("seed", 42),
        workers=cfg.get("workers", 8),
        cache=cfg.get("cache", False),
    )
    print(f"\nDone. Best weights: {results.save_dir}/weights/best.pt")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    main(p.parse_args().config)
```

### 16.4 Config Files (contoh)

**`configs/exp02_baseline_v11m_640.yaml`:**
```yaml
exp_id: exp02_baseline_v11m_640
model: yolo11m.pt
data: data/dataset_combined/data.yaml
epochs: 100
imgsz: 640
batch: 32
lr0: 0.01
cos_lr: true
mosaic: 1.0
mixup: 0.1
close_mosaic: 10
hsv_h: 0.015
hsv_s: 0.4
hsv_v: 0.4
flipud: 0.0
fliplr: 0.5
fl_gamma: 0.0
use_class_weights: false
device: 0
seed: 42
```

**`configs/exp06_v11m_1280_focal_weights_oversample.yaml`:**
```yaml
exp_id: exp06_v11m_1280_focal_weights_oversample
model: yolo11m.pt
data: data/dataset_combined_oversampled/data.yaml   # hasil oversample_minor.py
epochs: 150
imgsz: 1280
batch: 16            # 1280 perlu memory lebih
lr0: 0.01
cos_lr: true
warmup_epochs: 3
mosaic: 1.0
mixup: 0.1
close_mosaic: 15
hsv_h: 0.015
hsv_s: 0.4
hsv_v: 0.4
flipud: 0.0
fliplr: 0.5
scale: 0.5
erasing: 0.2
fl_gamma: 1.5
cls: 0.5
use_class_weights: true
class_weights: {B1: 3.78, B2: 2.02, B3: 1.00, B4: 2.49}
device: 0
amp: true
seed: 42
workers: 8
```

### 16.5 Running

```bash
# Single GPU
python scripts/train_teacher.py --config configs/exp06_v11m_1280_focal_weights_oversample.yaml

# Multi-GPU (DDP via Ultralytics built-in)
# Edit config: device: [0,1,2,3]
# Atau via CLI:
yolo detect train model=yolo11m.pt data=... device=0,1,2,3 imgsz=1280 batch=64

# Resume training
python scripts/train_teacher.py --config configs/exp06... # otomatis resume jika last.pt ada
```

---

## 17. Custom Ordinal Head & CORAL Loss

### 17.1 Ringkasan

YOLO classification branch normal output `(B, A, nc)` logits → softmax. Untuk ordinal, ganti dengan `(B, A, nc-1)` logits → sigmoid kumulatif. Karena modifikasi head saja, backbone+neck pretrained tetap dipakai.

### 17.2 `src/heads/coral_head.py`

```python
import torch
import torch.nn as nn

class CoralOrdinalHead(nn.Module):
    """Drop-in replacement untuk classification branch.
    Output: K-1 logits per anchor; logit_k = score - bias_k untuk k=0..K-2.
    Decoding: pred_class = sum(sigmoid(logits) > 0.5).
    """
    def __init__(self, in_channels: int, num_classes: int = 4):
        super().__init__()
        self.num_classes = num_classes
        # Single shared score head
        self.score = nn.Conv2d(in_channels, 1, kernel_size=1)
        # Learnable rank biases (CORAL ensures monotonic via shared score)
        self.bias = nn.Parameter(torch.zeros(num_classes - 1))

    def forward(self, x):
        # x: (B, in_channels, H, W) atau (B, A, in_channels)
        score = self.score(x)  # (B, 1, H, W)
        # broadcast: (B, K-1, H, W)
        return score - self.bias.view(1, -1, 1, 1)
```

### 17.3 `src/losses/coral.py`

```python
import torch
import torch.nn.functional as F

def class_idx_to_levels(y: torch.Tensor, K: int) -> torch.Tensor:
    """y: (N,) int dalam [0, K-1].
    Return: (N, K-1) cumulative one-hot.
      class 0 → [0,0,0]
      class 1 → [1,0,0]
      class 2 → [1,1,0]
      class 3 → [1,1,1]
    """
    levels = torch.zeros(y.size(0), K - 1, device=y.device)
    for k in range(K - 1):
        levels[:, k] = (y > k).float()
    return levels

def coral_loss(logits: torch.Tensor, levels: torch.Tensor,
               importance_weights: torch.Tensor = None):
    """logits: (N, K-1)  levels: (N, K-1)
    importance_weights: (K-1,) bobot per task (opsional, untuk imbalance)."""
    if importance_weights is None:
        importance_weights = torch.ones(logits.size(1), device=logits.device)
    val = -torch.sum(
        (F.logsigmoid(logits) * levels +
         (F.logsigmoid(logits) - logits) * (1 - levels)) * importance_weights,
        dim=1,
    )
    return val.mean()

def decode_ordinal(logits: torch.Tensor) -> torch.Tensor:
    """logits: (N, K-1) → predicted class index (N,)."""
    return (torch.sigmoid(logits) > 0.5).sum(dim=1)
```

### 17.4 Integrasi ke YOLO — Opsi B (Recommended)

Daripada bedah `tasks.py` Ultralytics (rapuh saat upgrade), **fine-tune classification head terpisah**: pakai best.pt teacher → ekstrak crop tiap GT bbox → train classifier ordinal kecil di atas crop.

```python
# scripts/train_ordinal_head.py
"""Phase 2 training: ekstrak crop dari best teacher, fine-tune CORAL head."""
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import sys; sys.path.insert(0, ".")
from src.heads.coral_head import CoralOrdinalHead
from src.losses.coral import class_idx_to_levels, coral_loss, decode_ordinal

class CropDataset(Dataset):
    def __init__(self, root: Path, split: str, transform=None):
        self.samples = []  # list of (image_path, bbox_yolo, class)
        for lbl in (root / "labels" / split).glob("*.txt"):
            img = root / "images" / split / (lbl.stem + ".jpg")
            for line in lbl.read_text().strip().splitlines():
                c, xc, yc, w, h = map(float, line.split())
                self.samples.append((img, (xc, yc, w, h), int(c)))
        self.tf = transform

    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        img_path, (xc, yc, w, h), cls = self.samples[i]
        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        x1 = max(int((xc - w/2) * W), 0); y1 = max(int((yc - h/2) * H), 0)
        x2 = min(int((xc + w/2) * W), W); y2 = min(int((yc + h/2) * H), H)
        crop = img.crop((x1, y1, x2, y2))
        if self.tf: crop = self.tf(crop)
        return crop, cls

def build_ordinal_classifier(num_classes=4):
    """Backbone ResNet18 + CORAL head — kecil tapi sufficient untuk crop."""
    import torchvision.models as M
    backbone = M.resnet18(weights=M.ResNet18_Weights.IMAGENET1K_V1)
    feat_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()
    head = CoralOrdinalHead(feat_dim, num_classes)
    # Adapt: head expects (B,C,1,1) — wrap
    class Wrap(nn.Module):
        def __init__(self, b, h):
            super().__init__(); self.b = b; self.h = h
        def forward(self, x):
            f = self.b(x).unsqueeze(-1).unsqueeze(-1)  # (B,C,1,1)
            return self.h(f).squeeze(-1).squeeze(-1)   # (B, K-1)
    return Wrap(backbone, head)

def train(epochs=20, lr=1e-3, batch=64, device="cuda"):
    tf = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])
    tf_val = transforms.Compose([
        transforms.Resize((128, 128)), transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])
    root = Path("data/dataset_combined")
    train_ds = CropDataset(root, "train", tf)
    val_ds = CropDataset(root, "val", tf_val)
    train_dl = DataLoader(train_ds, batch, shuffle=True, num_workers=4)
    val_dl = DataLoader(val_ds, batch, shuffle=False, num_workers=4)
    model = build_ordinal_classifier(4).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for ep in range(epochs):
        model.train()
        for crops, ys in train_dl:
            crops, ys = crops.to(device), ys.to(device)
            logits = model(crops)                                 # (B, K-1)
            levels = class_idx_to_levels(ys, K=4)
            loss = coral_loss(logits, levels)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        # Val
        model.eval()
        mae = 0.0; n = 0; b14 = 0
        with torch.no_grad():
            for crops, ys in val_dl:
                crops, ys = crops.to(device), ys.to(device)
                preds = decode_ordinal(model(crops))
                mae += (preds - ys).abs().sum().item()
                b14 += ((preds - ys).abs() == 3).sum().item()
                n += ys.size(0)
        print(f"ep {ep:02d} val MAE={mae/n:.3f} B1↔B4={b14/n:.4f}")
    torch.save(model.state_dict(), "runs/ordinal_head/best.pt")

if __name__ == "__main__":
    train()
```

**Inference combined:** YOLO deteksi bbox → crop → ordinal classifier → kelas final. Confidence detection tetap dari YOLO; classification confidence dari sigmoid product.

> **Trade-off:** Opsi B menambah latency (ekstra forward pass per bbox). Untuk mobile, kalau crop classifier dijaga kecil (ResNet18 quantized ~2MB), tambahan latency bisa dibatasi <5ms total per frame.

---

## 18. Knowledge Distillation (Teacher → Student)

### 18.1 Strategi

**Response-based KD** (paling simple & efektif untuk YOLO):
- Soft cls logits: KLDiv dengan temperature T
- Soft bbox: SmoothL1 antara teacher dan student box prediction (hanya pada anchor yang teacher confident, conf > τ)

```
L_total = L_yolo_hard(student, gt) + α * (L_cls_kd + λ_box * L_box_kd)
```

Default: α=1.0, T=4.0, λ_box=2.0, τ=0.25.

### 18.2 `src/trainers/kd_trainer.py`

```python
"""Subclass DetectionTrainer Ultralytics, inject teacher prediction ke loss."""
import torch
import torch.nn.functional as F
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics import YOLO

class KDLoss(v8DetectionLoss):
    def __init__(self, model, teacher, alpha=1.0, T=4.0, lambda_box=2.0, tau=0.25):
        super().__init__(model)
        self.teacher = teacher
        self.teacher.eval()
        for p in self.teacher.parameters(): p.requires_grad_(False)
        self.alpha = alpha
        self.T = T
        self.lambda_box = lambda_box
        self.tau = tau

    def __call__(self, preds, batch):
        loss_hard, loss_items = super().__call__(preds, batch)
        with torch.no_grad():
            t_preds = self.teacher(batch["img"])
        # preds: list of feature maps from heads
        # Asumsi struktur preds = [pred_p3, pred_p4, pred_p5], setiap pred shape
        # (B, no, H, W) di mana no = nc + reg_max*4
        kd_cls, kd_box, n_anc = 0.0, 0.0, 0
        for s_p, t_p in zip(preds, t_preds):
            s_p = s_p.permute(0, 2, 3, 1)  # (B, H, W, no)
            t_p = t_p.permute(0, 2, 3, 1)
            nc = self.nc
            s_cls = s_p[..., :nc]
            t_cls = t_p[..., :nc]
            s_box = s_p[..., nc:]
            t_box = t_p[..., nc:]
            # Mask: anchor yang teacher confident
            t_conf = t_cls.sigmoid().max(dim=-1).values  # (B,H,W)
            mask = t_conf > self.tau
            if mask.sum() == 0: continue
            s_cls_m = s_cls[mask]; t_cls_m = t_cls[mask]
            s_box_m = s_box[mask]; t_box_m = t_box[mask]
            # KL on soft cls
            kd_cls = kd_cls + F.kl_div(
                F.log_softmax(s_cls_m / self.T, dim=-1),
                F.softmax(t_cls_m / self.T, dim=-1),
                reduction="batchmean"
            ) * (self.T ** 2)
            # SmoothL1 on box (DFL bins atau xywh — sesuaikan)
            kd_box = kd_box + F.smooth_l1_loss(s_box_m, t_box_m, reduction="mean")
            n_anc += 1
        if n_anc > 0:
            kd_total = self.alpha * (kd_cls + self.lambda_box * kd_box)
            loss_hard = loss_hard + kd_total
            loss_items = torch.cat([loss_items,
                torch.tensor([kd_cls.detach(), kd_box.detach()], device=loss_items.device)])
        return loss_hard, loss_items

class KDTrainer(DetectionTrainer):
    def __init__(self, *args, teacher_weights: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = YOLO(teacher_weights).model.to(self.device)

    def init_criterion(self):
        return KDLoss(self.model, self.teacher,
                      alpha=self.args.kd_alpha if hasattr(self.args,'kd_alpha') else 1.0,
                      T=self.args.kd_T if hasattr(self.args,'kd_T') else 4.0)
```

### 18.3 `scripts/train_student_kd.py`

```python
import argparse, yaml
from pathlib import Path
import sys; sys.path.insert(0, ".")
from src.seed import seed_everything
from src.trainers.kd_trainer import KDTrainer

def main(cfg_path):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    seed_everything(cfg.get("seed", 42))
    overrides = dict(
        model=cfg["model"],            # e.g. yolo11n.pt
        data=cfg["data"],
        epochs=cfg["epochs"],
        imgsz=cfg["imgsz"],
        batch=cfg["batch"],
        device=cfg.get("device", 0),
        project="runs/detect",
        name=cfg["exp_id"],
        amp=True,
    )
    trainer = KDTrainer(overrides=overrides, teacher_weights=cfg["teacher_weights"])
    trainer.train()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    main(p.parse_args().config)
```

**`configs/exp08_student_v11n_kd.yaml`:**
```yaml
exp_id: exp08_student_v11n_kd
model: yolo11n.pt
teacher_weights: runs/detect/exp06_v11m_1280_focal_weights_oversample/weights/best.pt
data: data/dataset_combined_oversampled/data.yaml
epochs: 200
imgsz: 1280
batch: 32
device: 0
seed: 42
kd_alpha: 1.0
kd_T: 4.0
```

> **Note teknis:** Ultralytics structure head output bisa berubah antar versi. Sebelum run, tambahkan `print(s_p.shape)` di KDLoss untuk verifikasi shape; sesuaikan slicing `s_box` jika DFL diaktifkan (default v11 pakai DFL dengan reg_max=16).

---

## 19. Evaluation & Custom Metrics

### 19.1 Standard mAP

```bash
yolo detect val model=runs/detect/exp06.../weights/best.pt data=data/dataset_combined/data.yaml imgsz=1280
# Output: per-class mAP, P, R, mAP@0.5, mAP@0.5:0.95, confusion_matrix.png
```

### 19.2 `scripts/eval_full.py`

```python
"""Evaluasi lengkap: mAP per class, ordinal MAE, B1↔B4 confusion, ordinal-weighted CM."""
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO

NAMES = ["B1", "B2", "B3", "B4"]
K = 4

def ordinal_metrics_from_cm(cm: np.ndarray):
    """cm: (K, K) confusion matrix kelas-only."""
    if cm.sum() == 0:
        return dict(mae=float("nan"), b1b4_rate=float("nan"))
    mae, total = 0.0, 0
    for i in range(K):
        for j in range(K):
            mae += abs(i - j) * cm[i, j]; total += cm[i, j]
    return dict(mae=float(mae / total),
                b1b4_rate=float((cm[0, 3] + cm[3, 0]) / total))

def ordinal_weighted_cm(cm: np.ndarray):
    """Confusion matrix dengan bobot |i-j| di off-diagonal."""
    W = np.abs(np.subtract.outer(np.arange(K), np.arange(K)))
    return cm * W

def plot_cm(cm, title, out_path, fmt="d"):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt=fmt, xticklabels=NAMES, yticklabels=NAMES,
                cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    plt.tight_layout(); plt.savefig(out_path, dpi=140); plt.close()

def bootstrap_ci(values, n_boot=2000, ci=95):
    rng = np.random.default_rng(42)
    arr = np.array(values)
    boots = [rng.choice(arr, len(arr), replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [(100-ci)/2, 100-(100-ci)/2])
    return float(arr.mean()), float(lo), float(hi)

def main(weights, data, imgsz, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    model = YOLO(weights)
    metrics = model.val(data=data, imgsz=imgsz, save_json=True,
                        plots=True, project=str(out), name="val")
    cm_full = metrics.confusion_matrix.matrix  # (K+1, K+1)
    cm = cm_full[:K, :K].astype(np.int64)
    ordinal = ordinal_metrics_from_cm(cm)
    cm_w = ordinal_weighted_cm(cm)
    plot_cm(cm, "Confusion Matrix (raw count)", out / "cm_raw.png")
    plot_cm(cm_w, "Confusion Matrix (ordinal-weighted by |i-j|)",
            out / "cm_ordinal_weighted.png", fmt=".0f")
    summary = {
        "weights": str(weights),
        "mAP@0.5": float(metrics.box.map50),
        "mAP@0.5:0.95": float(metrics.box.map),
        "per_class_mAP@0.5": {NAMES[i]: float(v) for i, v in enumerate(metrics.box.maps)},
        "ordinal_MAE": ordinal["mae"],
        "B1_B4_confusion_rate": ordinal["b1b4_rate"],
        "ordinal_weighted_cm_sum": int(cm_w.sum()),
    }
    (out / "eval_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--data", default="data/dataset_combined/data.yaml")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--out_dir", default="reports/eval")
    args = p.parse_args()
    main(args.weights, args.data, args.imgsz, args.out_dir)
```

### 19.4 Bootstrap Significance

Saat membandingkan exp A vs B, run val 5× dengan seed berbeda (atau 5 fold) → bootstrap mean+CI:

```python
# Pseudo: kumpulkan list mAP dari 5 run, panggil bootstrap_ci(values)
# Klaim "B > A signifikan" jika CI lower bound (B) > CI upper bound (A)
```

---

## 20. Workflow Ablation (Step-by-Step)

### 20.1 Ablation Matrix

| Exp ID | Model | imgsz | Focal | ClassW | Oversample | KD | Note |
|--------|-------|-------|-------|--------|-----------|----|----|
| exp01 | yolov8m | 640 | – | – | – | – | Sanity, framework comparison |
| exp02 | yolo11m | 640 | – | – | – | – | v11 baseline |
| exp03 | yolo11m | 1280 | – | – | – | – | High res baseline |
| exp04 | yolo11m | 1280 | γ=1.5 | – | – | – | Focal only |
| exp05 | yolo11m | 1280 | γ=1.5 | ✅ | – | – | Focal + weights |
| exp06 | yolo11m | 1280 | γ=1.5 | ✅ | ×2 B1 | – | Full imbalance ★ |
| exp07 | yolo11m | 1280 | γ=1.5 | ✅ | ×2 B1 | – | + CORAL crop classifier |
| exp08 | yolo11n | 1280 | inherit | inherit | inherit | from exp06 | Mobile target ★ |
| exp09 | yolo11s | 1280 | inherit | inherit | inherit | from exp06 | Mobile (akurasi+) |
| exp10 | exp08 weights | – | – | – | – | – | SAHI eval only |

★ = milestone utama.

### 20.2 `scripts/run_ablation.py`

```python
"""Loop semua config, jalankan train+eval, aggregate ke CSV."""
import argparse, subprocess, json, time
from pathlib import Path
import pandas as pd

CONFIGS_DIR = Path("configs")
RUNS_DIR = Path("runs/detect")
REPORTS = Path("reports"); REPORTS.mkdir(exist_ok=True)
CSV = REPORTS / "ablation_summary.csv"

def run_one(cfg_path: Path, is_kd: bool = False):
    script = "scripts/train_student_kd.py" if is_kd else "scripts/train_teacher.py"
    print(f"\n[ABLATION] running {cfg_path.name}")
    t0 = time.time()
    subprocess.run(["python", script, "--config", str(cfg_path)], check=True)
    elapsed = time.time() - t0
    return elapsed

def collect_metrics(exp_id: str):
    run_dir = RUNS_DIR / exp_id
    weights = run_dir / "weights" / "best.pt"
    if not weights.exists(): return None
    out_eval = REPORTS / "eval" / exp_id
    subprocess.run(["python", "scripts/eval_full.py",
                    "--weights", str(weights), "--out_dir", str(out_eval)], check=True)
    summary = json.loads((out_eval / "val" / "eval_summary.json").read_text()) \
              if (out_eval / "val" / "eval_summary.json").exists() else \
              json.loads((out_eval / "eval_summary.json").read_text())
    return summary

def main(only=None):
    rows = []
    cfgs = sorted(CONFIGS_DIR.glob("exp*.yaml"))
    for cfg in cfgs:
        if only and cfg.stem not in only: continue
        is_kd = "student" in cfg.stem
        elapsed = run_one(cfg, is_kd=is_kd)
        m = collect_metrics(cfg.stem)
        if m is None:
            print(f"[skip metrics] {cfg.stem}"); continue
        m["exp_id"] = cfg.stem
        m["train_seconds"] = elapsed
        rows.append(m)
        # Save incrementally
        pd.DataFrame(rows).to_csv(CSV, index=False)
    print(f"\nDone. Summary → {CSV}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="*", help="Subset exp_id, e.g. exp03 exp04")
    main(p.parse_args().only)
```

### 20.3 Decision Tree

```
START → exp01 (sanity)
   │
   ├── mAP@0.5 < 0.50? → STOP, debug data pipeline
   ├── mAP@0.5 ≥ 0.50 → lanjut exp02 (v11 baseline)
   │
   ↓
exp02 → exp03 (naik res 640→1280)
   │
   ├── mAP B4 naik signifikan? → confirmed: B4 small object, lanjut high-res
   ├── mAP tidak naik → batal, kembali imgsz=640 untuk efisiensi
   │
   ↓
exp03 → exp04 (focal) → exp05 (+ weights) → exp06 (+ oversample)
   │
   ├── per-class mAP B1 sudah dalam 15% gap dari B3? → STOP imbalance, ke exp08
   ├── B1 masih jauh? → coba exp07 (CORAL crop)
   │
   ↓
exp07 keputusan:
   ├── B1↔B4 confusion < 5%? → cukup, baseline cross-entropy
   ├── B1↔B4 ≥ 5%? → adopsi CORAL untuk inference final
   │
   ↓
exp06 best → exp08 (KD ke v11n) → exp09 (KD ke v11s, jika v11n gap > 5%)
   │
   ↓
exp10 SAHI eval (cek apakah B4 detection naik di test set)
   │
   ↓
Pick best mobile model → mobile export pipeline (Section 22)
```

### 20.4 Budget Estimasi

Asumsi single A100 (40GB):

| Exp | imgsz | epochs | Est. waktu |
|-----|-------|--------|-----------|
| exp01–02 | 640 | 100 | ~3 jam |
| exp03 | 1280 | 100 | ~10 jam |
| exp04–06 | 1280 | 150 | ~14 jam masing-masing |
| exp07 | – | 20 (crop) | ~1 jam |
| exp08 | 1280 | 200 | ~12 jam |
| exp09 | 1280 | 200 | ~14 jam |
| exp10 | – | – | <30 menit (eval only) |

Total ablation full: **~80 GPU-hour** (bisa diparalel jika multi-GPU).

---

## 21. SAHI Inference (Small Object Boost)

```python
# scripts/sahi_inference.py
"""Sliced inference untuk boost B4 (small object) detection."""
import argparse
from pathlib import Path
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction, predict
import json

def main(weights, source, out_dir, slice_h=640, slice_w=640, overlap=0.2,
         conf_threshold=0.25, imgsz=1280):
    model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=weights,
        confidence_threshold=conf_threshold,
        device="cuda:0",
    )
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    results_coco = []
    for img_path in sorted(Path(source).glob("*.jpg")):
        pred = get_sliced_prediction(
            str(img_path), model,
            slice_height=slice_h, slice_width=slice_w,
            overlap_height_ratio=overlap, overlap_width_ratio=overlap,
            postprocess_type="NMS", postprocess_match_threshold=0.5,
        )
        for obj in pred.object_prediction_list:
            results_coco.append({
                "image": img_path.name,
                "bbox": obj.bbox.to_xywh(),
                "category": obj.category.id,
                "score": obj.score.value,
            })
    (out / "sahi_predictions.json").write_text(json.dumps(results_coco, indent=2))
    print(f"Saved {len(results_coco)} predictions → {out}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--source", required=True, help="Folder gambar")
    p.add_argument("--out_dir", default="reports/sahi")
    args = p.parse_args()
    main(args.weights, args.source, args.out_dir)
```

**Compare standard vs SAHI:**
```bash
# Standard
python scripts/eval_full.py --weights runs/.../best.pt --out_dir reports/eval/exp08
# SAHI
python scripts/sahi_inference.py --weights runs/.../best.pt --source data/dataset_combined/images/test --out_dir reports/sahi/exp08
# Konversi predictions.json → COCO format → evaluasi mAP per class B4
# (kerjakan manual via pycocotools.cocoeval)
```

---

## 22. Mobile Export Pipeline

### 22.1 PT → ONNX

```python
# scripts/export_mobile.py
import argparse
from ultralytics import YOLO

def export_onnx(weights, imgsz=640, out_dir="exports"):
    m = YOLO(weights)
    path = m.export(format="onnx", imgsz=imgsz, opset=12, simplify=True,
                    dynamic=False, half=False)
    print(f"ONNX → {path}")
    return path

def export_tflite(weights, imgsz=640, int8=True):
    m = YOLO(weights)
    path = m.export(format="tflite", imgsz=imgsz, int8=int8,
                    data="data/dataset_combined/data.yaml")  # data → calibration
    print(f"TFLite → {path}")
    return path

def export_coreml(weights, imgsz=640):
    m = YOLO(weights)
    path = m.export(format="coreml", imgsz=imgsz, nms=True)
    print(f"CoreML → {path}")
    return path

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--formats", nargs="+", default=["onnx", "tflite", "coreml"])
    args = p.parse_args()
    if "onnx" in args.formats: export_onnx(args.weights, args.imgsz)
    if "tflite" in args.formats: export_tflite(args.weights, args.imgsz, int8=True)
    if "coreml" in args.formats: export_coreml(args.weights, args.imgsz)
```

> **Penting:** Ultralytics handle calibration TFLite secara otomatis kalau `data` parameter diisi (sample 100–200 dari val). Kalau perlu kontrol manual, pakai `tf.lite.TFLiteConverter` dengan `representative_dataset` generator.

### 22.2 Manual TFLite INT8 (Kontrol Lebih)

```python
# Alternatif untuk kontrol lebih granular
import tensorflow as tf, numpy as np
from PIL import Image
from pathlib import Path

def representative_dataset(calib_dir, imgsz=640, n=200):
    files = list(Path(calib_dir).glob("*.jpg"))[:n]
    def gen():
        for f in files:
            img = Image.open(f).convert("RGB").resize((imgsz, imgsz))
            arr = np.array(img, dtype=np.float32) / 255.0
            arr = arr.transpose(2, 0, 1)[None]  # NCHW
            yield [arr.astype(np.float32)]
    return gen

converter = tf.lite.TFLiteConverter.from_saved_model("exports/best_saved_model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset(
    "data/dataset_combined/images/val", imgsz=640, n=200)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_model = converter.convert()
Path("exports/best_int8.tflite").write_bytes(tflite_model)
```

### 22.3 Verifikasi Numerical Fidelity

```bash
# Run val pada model terkompilasi
yolo detect val model=exports/best.tflite data=data/dataset_combined/data.yaml imgsz=640
# Bandingkan dengan PT → mAP loss harus < 2%
```

### 22.4 On-Device Benchmark (Android)

```bash
# Push TFLite ke device
adb push exports/best_int8.tflite /data/local/tmp/

# Pakai TFLite Benchmark Tool (download dari TF release)
adb push tflite_benchmark_model /data/local/tmp/
adb shell chmod +x /data/local/tmp/tflite_benchmark_model
adb shell /data/local/tmp/tflite_benchmark_model \
  --graph=/data/local/tmp/best_int8.tflite \
  --num_threads=4 \
  --use_gpu=true \
  --enable_op_profiling=true

# Output: latency mean / std, tier op breakdown.
# Snapdragon 8 Gen 2: expected 15–25 ms dengan GPU delegate, 30–50 ms CPU 4 thread.
```

### 22.5 NMS On-Device vs Off-Device

| Pendekatan | Pro | Con |
|------------|-----|-----|
| NMS embedded di TFLite (`nms=True`) | App tinggal pakai output, simpler | Beberapa runtime tidak support, accuracy kadang berbeda |
| NMS di app code (Java/Kotlin/Swift) | Kontrol penuh threshold | Perlu implement IoU lokal |

**Rekomendasi:** Embed NMS di model untuk Android (TFLite support OK), keep manual NMS untuk iOS CoreML jika ada masalah kompatibilitas.

---

## 23. Stage 2 — Multi-View Aggregation Pipeline

### 23.1 `src/pipeline/multiview_count.py`

```python
"""Pipeline: 4 gambar 1 pohon → unique bunch count per kelas.
Pendekatan: feature-similarity inter-view + IoU intra-view linking."""
import torch, json
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from typing import List, Dict

NAMES = ["B1", "B2", "B3", "B4"]
SIDES = ["sisi_1", "sisi_2", "sisi_3", "sisi_4"]

class MultiViewAggregator:
    def __init__(self, weights: str, sim_threshold: float = 0.75,
                 conf_threshold: float = 0.25, device: str = "cuda:0"):
        self.model = YOLO(weights)
        self.sim_thr = sim_threshold
        self.conf_thr = conf_threshold
        self.device = device
        self._features = []
        self._hook = None
        self._register_hook()

    def _register_hook(self):
        """Hook untuk capture feature embedding pre-head (neck output)."""
        # Cari layer terakhir neck Ultralytics (biasanya self.model.model.model[22] / Detect head)
        # Ekstrak input dari Detect layer
        target = self.model.model.model[-1]  # Detect head
        def hook(module, inputs, outputs):
            self._features = inputs[0]  # list of feature maps from neck
        self._hook = target.register_forward_hook(hook)

    @torch.no_grad()
    def detect_one(self, img_path: str) -> List[Dict]:
        """Return list of {bbox_xyxy, class, conf, embedding}."""
        results = self.model.predict(img_path, conf=self.conf_thr,
                                     device=self.device, verbose=False)
        r = results[0]
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            return []
        # Embedding per bbox: ROI pool dari feature map terbesar (P3)
        feat = self._features[0]  # (1, C, H, W)
        H, W = feat.shape[-2:]
        items = []
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy()
            cls = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())
            # Map bbox ke koordinat feature map
            x1, y1, x2, y2 = xyxy
            img_h, img_w = r.orig_shape
            fx1 = int(x1 / img_w * W); fy1 = int(y1 / img_h * H)
            fx2 = max(int(x2 / img_w * W), fx1+1); fy2 = max(int(y2 / img_h * H), fy1+1)
            roi = feat[0, :, fy1:fy2, fx1:fx2]
            emb = roi.mean(dim=(1, 2)).cpu().numpy() if roi.numel() > 0 else np.zeros(feat.shape[1])
            emb = emb / (np.linalg.norm(emb) + 1e-6)
            items.append(dict(bbox=xyxy, cls=cls, conf=conf, emb=emb, side=Path(img_path).stem))
        return items

    def link_across_views(self, all_dets: List[List[Dict]]) -> List[Dict]:
        """all_dets: list panjang 4 (per side). Return unique bunches."""
        flat = [d for view in all_dets for d in view]
        if not flat:
            return []
        # Greedy clustering by cosine similarity (intra-class only)
        clusters = []
        used = [False] * len(flat)
        for i, d in enumerate(flat):
            if used[i]: continue
            cluster = [i]; used[i] = True
            for j in range(i+1, len(flat)):
                if used[j]: continue
                if flat[j]["cls"] != d["cls"]: continue
                # Jangan link di view yang sama (gunakan IoU intra-view ternyata lebih baik)
                if flat[j]["side"] == d["side"]:
                    iou = self._iou(d["bbox"], flat[j]["bbox"])
                    if iou > 0.5:
                        cluster.append(j); used[j] = True
                else:
                    sim = float(np.dot(d["emb"], flat[j]["emb"]))
                    if sim > self.sim_thr:
                        cluster.append(j); used[j] = True
            members = [flat[k] for k in cluster]
            clusters.append(dict(
                cls=d["cls"],
                confidence=float(np.mean([m["conf"] for m in members])),
                support_views=sorted({m["side"] for m in members}),
                n_appearances=len(members),
            ))
        return clusters

    @staticmethod
    def _iou(a, b):
        x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
        x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        area_a = (a[2]-a[0]) * (a[3]-a[1])
        area_b = (b[2]-b[0]) * (b[3]-b[1])
        return inter / max(area_a + area_b - inter, 1e-6)

    def count_per_tree(self, tree_id: str, image_dir: Path) -> Dict[str, int]:
        all_dets = []
        for side in SIDES:
            img = image_dir / f"{tree_id}_{side[-1]}.jpg"
            if img.exists():
                all_dets.append(self.detect_one(str(img)))
        unique = self.link_across_views(all_dets)
        counts = {n: 0 for n in NAMES}
        for u in unique:
            counts[NAMES[u["cls"]]] += 1
        return counts
```

### 23.2 `scripts/eval_multiview.py`

```python
"""Bandingkan count per pohon vs JSON ground truth."""
import json, argparse
from pathlib import Path
import numpy as np
import sys; sys.path.insert(0, ".")
from src.pipeline.multiview_count import MultiViewAggregator, NAMES

def main(weights, json_dirs, image_root, out):
    agg = MultiViewAggregator(weights)
    rows = []
    for jd in json_dirs:
        for jp in Path(jd).glob("*.json"):
            gt = json.loads(jp.read_text())
            tree_id = gt["tree_id"]
            # GT counts dari unique bunches
            gt_counts = {n: 0 for n in NAMES}
            for b in gt["bunches"]:
                gt_counts[b["class"]] += 1
            pred_counts = agg.count_per_tree(tree_id, Path(image_root))
            row = dict(tree_id=tree_id)
            for n in NAMES:
                row[f"gt_{n}"] = gt_counts[n]
                row[f"pred_{n}"] = pred_counts[n]
                row[f"err_{n}"] = abs(gt_counts[n] - pred_counts[n])
            row["total_gt"] = sum(gt_counts.values())
            row["total_pred"] = sum(pred_counts.values())
            row["total_err"] = abs(row["total_gt"] - row["total_pred"])
            rows.append(row)
    # Aggregate
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(out + ".csv", index=False)
    summary = {
        "n_trees": len(df),
        "MAE_per_class": {n: float(df[f"err_{n}"].mean()) for n in NAMES},
        "MAE_total": float(df["total_err"].mean()),
        "pct_within_1": float((df["total_err"] <= 1).mean()),
    }
    Path(out + ".json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--json_dirs", nargs="+", required=True,
                   help="dataset_combined_{1,2,3}_yolo/json/")
    p.add_argument("--image_root", required=True,
                   help="Folder yang berisi semua gambar pohon (dataset_combined/images/all)")
    p.add_argument("--out", default="reports/multiview_eval")
    args = p.parse_args()
    main(args.weights, args.json_dirs, args.image_root, args.out)
```

### 23.3 WBF Sanity Baseline

```python
# Alternatif simpler — pakai ensemble_boxes WBF asumsikan view sebagai "model" berbeda
from ensemble_boxes import weighted_boxes_fusion

def wbf_per_tree(predictions_per_view, iou_thr=0.5, skip_box_thr=0.25):
    # predictions_per_view: list 4 dict {boxes_xyxy_norm, scores, labels}
    boxes_list = [p["boxes"] for p in predictions_per_view]
    scores_list = [p["scores"] for p in predictions_per_view]
    labels_list = [p["labels"] for p in predictions_per_view]
    boxes, scores, labels = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list,
        weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    return boxes, scores, labels
```

> **Catatan:** WBF berbasis IoU koordinat — secara teori salah untuk view berbeda (sudut beda), tapi sebagai sanity check untuk batas atas overcounting masih berguna.

---

## 24. Logging, Tracking, Reproducibility

### 24.1 Wandb Setup

```python
# Tambahkan ke train_teacher.py
import wandb
wandb.init(
    project="damimas-yolo",
    name=cfg["exp_id"],
    config=cfg,
    tags=[cfg["model"], f"imgsz{cfg['imgsz']}"],
)
# Ultralytics auto-log jika wandb terinstall — atau pasang callback eksplisit
```

### 24.2 Git + Config Hash

```python
import hashlib, subprocess, json
def fingerprint(cfg: dict) -> str:
    git = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    cfg_h = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
    return f"{git[:8]}-{cfg_h}"
# Simpan ke runs/{exp_id}/fingerprint.txt
```

### 24.3 Determinism Caveats

- `cudnn.deterministic=True` slowdown 5–15% pada conv besar
- AMP (mixed precision) introduces small variance bahkan dengan seed sama
- DataLoader `num_workers > 0` perlu `worker_init_fn` untuk seed konsisten
- Diterima: gap 0.3–0.5% mAP antar run sebagai noise

---

## 25. Failure Modes & Debug Playbook

| Gejala | Diagnosis | Fix |
|--------|----------|------|
| mAP B1 stuck < 0.50 | Imbalance tidak teratasi | Naikkan oversample factor ke ×3, fl_gamma ke 2.0, cek class_weights aktif |
| Loss NaN epoch 1–5 | Batch terlalu besar untuk imgsz, lr terlalu tinggi | Turunkan batch ke 8, lr0 ke 0.005, naikkan warmup ke 5 |
| mAP val anjlok setelah epoch 80 | Overfitting (mosaic OFF terlalu cepat) | Naikkan close_mosaic ke 20, tambah weight_decay |
| B1↔B4 confusion > 10% | Color cue tidak ditangkap, atau hue jitter terlalu besar | Verifikasi hsv_h ≤ 0.015, cek dataset_stats.py untuk distribusi warna B1 |
| Student gap > 5% dari teacher | KD signal lemah | Naikkan T ke 6, alpha ke 1.5, tambah feature distillation di neck (hint loss) |
| TFLite akurasi drop > 5% | Calibration data tidak representatif | Gunakan 500 sample dari train+val, swap ke per-channel quant |
| WBF count 2× truth | IoU threshold lintas view terlalu rendah | Naikkan sim_threshold ke 0.85, atau wajibkan minimal 2 view sebagai konfirmasi |
| Multi-view miss B4 | B4 kecil tidak masuk filter conf | Turunkan conf_thr ke 0.15 untuk B4 saja (per-class threshold) |
| GPU OOM imgsz=1280 batch=16 | Memory tinggi | Aktifkan `cache=False`, `workers=4`, gradient accumulation manual |

---

## 26. Verification Checklist (End-to-End)

- [ ] `python scripts/verify_dataset.py` → semua split image=label, distribusi sesuai Section 1.3
- [ ] `python scripts/dataset_stats.py` → confirm `mean_yc(B1) > mean_yc(B4)` (spatial prior)
- [ ] `python scripts/oversample_minor.py --target_cls 0 --factor 2` → folder `dataset_combined_oversampled/` siap
- [ ] Exp01 baseline jalan tanpa error → `runs/detect/exp01.../weights/best.pt` ada
- [ ] Exp06 (full imbalance) → mAP@0.5 ≥ 0.80 di test, B1 mAP gap dari B3 < 15%
- [ ] Exp08 student → mAP gap dari teacher exp06 < 5%
- [ ] `python scripts/export_mobile.py --weights ...exp08.../best.pt` → `exports/best_int8.tflite` size < 2MB
- [ ] On-device benchmark Snapdragon 8 → latency < 30ms / frame
- [ ] `python scripts/eval_multiview.py --weights ... --json_dirs dataset_combined_*_yolo/json` → MAE count per class < 1.0 pada 228 pohon
- [ ] `reports/ablation_summary.csv` lengkap dengan semua 10 exp
- [ ] Setiap milestone exp06 / exp08 memiliki `confusion_matrix.png` + `eval_summary.json` di `reports/eval/`

> **Definition of Done Stage 1+2:** Mobile model dengan latency <30ms, mAP@0.5 ≥ 0.78 (sedikit drop dari teacher), MAE count per pohon ≤ 1.0 untuk semua kelas.

---

# Bagian III: Branch JSON-Aware (Headline Sekarang)

> Ini bagian yang dirujuk oleh Section 0.6. Berdiri di atas insight CONTEXT.md bahwa **JSON multi-view adalah satu-satunya angle struktural yang belum pernah dieksplor**, sementara semua jalur "knob & arsitektur" sudah dijawab `INSUFFICIENT` oleh AR29/AR34/E0.

---

## 27. JSON sebagai Label-Audit Tool (Test Hipotesis Label-Ceiling)

### 27.1 Premis

1 tandan fisik yang sama dilihat dari ≥2 sisi pohon **harus** mendapat kelas yang sama. JSON bunch-linking sudah memberi mapping `bunch_id → list of (side, box_index)` untuk 228 pohon. Inkonsistensi label di sini adalah salah satu dari:

- **(a)** label noise (annotator beda kasih kelas berbeda untuk objek sama)
- **(b)** ambiguity nyata yang membuktikan boundary kelas tidak well-defined oleh manusia sendiri

Kedua-duanya = **upper-bound** untuk berapa tinggi model bisa naik. Ini yang E0 + ablation tidak bisa jawab tapi JSON bisa.

### 27.2 Workflow

1. Untuk setiap pohon dengan JSON, untuk setiap bunch dengan `appearance_count ≥ 2`, ekstrak kelas di setiap appearance
2. Hitung **consistency rate**: % bunch dengan label sama di semua view-nya
3. **Per kelas**, hitung "leak distribution" — saat tidak konsisten, kelas apa yang ikut muncul
4. Bandingkan inconsistency rate B2/B3 vs B1/B4

### 27.3 `scripts/audit_label_via_json.py`

```python
"""Audit konsistensi label cross-view dari JSON bunch-link.
Output: per-class inconsistency rate, leak pair matrix, shortlist bunch ambigu untuk re-review."""
import json
from pathlib import Path
from collections import Counter
import pandas as pd

JSON_DIRS = [
    "D:/Work/Assisten Dosen/Folder Linked Dataset/dataset_combined_1_yolo/json",
    "D:/Work/Assisten Dosen/Folder Linked Dataset/dataset_combined_2_yolo/json",
    "D:/Work/Assisten Dosen/Folder Linked Dataset/dataset_combined_3_yolo/json",
]
NAMES = ["B1", "B2", "B3", "B4"]
OUT = Path("reports/label_audit"); OUT.mkdir(parents=True, exist_ok=True)

def main():
    inconsistent_rows = []
    per_class_total = Counter()
    per_class_inconsistent = Counter()
    leak_pairs = Counter()  # (true_class, leaked_class)
    n_trees = 0; n_bunches_multi = 0

    for jd in JSON_DIRS:
        for jp in Path(jd).glob("*.json"):
            data = json.loads(jp.read_text())
            n_trees += 1
            for bunch in data["bunches"]:
                if bunch["appearance_count"] < 2:
                    continue
                n_bunches_multi += 1
                # Re-derive class per appearance dari images dict (tidak percaya bunch.class saja)
                labels = []
                for app in bunch["appearances"]:
                    side = app["side"]; idx = app["box_index"]
                    cls = data["images"][side]["annotations"][idx]["class"]
                    labels.append(cls)
                base = bunch["class"]
                per_class_total[base] += 1
                if len(set(labels)) > 1:
                    per_class_inconsistent[base] += 1
                    inconsistent_rows.append({
                        "tree": data["tree_id"],
                        "bunch_id": bunch["bunch_id"],
                        "json_class": base,
                        "labels": "|".join(labels),
                        "n_views": len(labels),
                    })
                    for a in labels:
                        for b in labels:
                            if a != b:
                                leak_pairs[(a, b)] += 1

    # Print summary
    print(f"\nTrees with JSON: {n_trees}")
    print(f"Multi-view bunches (appearance≥2): {n_bunches_multi}\n")
    print("Inconsistency rate per class:")
    rows = []
    for c in NAMES:
        tot = per_class_total[c]; inc = per_class_inconsistent[c]
        rate = inc / max(tot, 1)
        print(f"  {c}: {inc}/{tot} = {rate:.3%}")
        rows.append(dict(class_=c, total=tot, inconsistent=inc, rate=rate))
    pd.DataFrame(rows).to_csv(OUT / "per_class_inconsistency.csv", index=False)

    print("\nTop leak pairs (true → leaked):")
    leak_rows = []
    for (a, b), n in leak_pairs.most_common(20):
        print(f"  {a} → {b}: {n}")
        leak_rows.append(dict(true=a, leaked=b, count=n))
    pd.DataFrame(leak_rows).to_csv(OUT / "leak_pairs.csv", index=False)

    pd.DataFrame(inconsistent_rows).to_csv(OUT / "inconsistent_bunches.csv", index=False)
    print(f"\nShortlist for re-review → {OUT/'inconsistent_bunches.csv'}")

if __name__ == "__main__":
    main()
```

### 27.4 Hipotesis Falsifiable

| ID | Hipotesis | Kondisi konfirmasi | Implikasi |
|---|---|---|---|
| H-LBL-1 | B2/B3 inconsistency rate >> B1/B4 | rate(B2 ∪ B3) > 2× rate(B1 ∪ B4) | Konfirmasi label-ceiling pada B2/B3 — bukti bahwa E0 confusion 34% B2→B3 sebagian dari noise label |
| H-LBL-2 | Leak pair B2↔B3 dominan | top-1 leak pair = (B2,B3) atau (B3,B2) dengan count >> pair lain | Boundary B2/B3 yang ambigu, bukan random noise |
| H-LBL-3 | B1/B4 cross-view rate < 5% | konsistensi tinggi pada kelas paling matang & paling muda | Sanity check: warna ekstrim cukup unambiguous untuk annotator |

### 27.5 Decision Rule Pasca-Audit

- **Jika H-LBL-1 + H-LBL-2 confirmed** (mis. B2/B3 inconsistency > 15%): label-ceiling kuat → eksplisit dokumentasikan sebagai upper-bound argument; **lanjut ke Section 29 JSON-02 (consensus relabel) dan JSON-03 (3-class merge)**
- **Jika tidak confirmed** (B2/B3 inconsistency < 5%): label OK; bottleneck murni model → JSON multi-view masih layak untuk supervisi training (Section 28) atau B4 multi-view inference, **tapi reframing task tidak akan menolong**

> **Cost:** Audit ini cheap — read-only analisis, ~5 menit jalan, tidak butuh GPU. **Wajib dijalankan sebelum eksperimen JSON lain.**

---

## 28. JSON sebagai Multi-View Supervision Signal

### 28.1 Tiga Pendekatan (Urut Prioritas)

**Pendekatan A — Post-Hoc Consensus Relabeling (PRIORITAS 1, simpel)**

Untuk setiap bunch multi-view:
- Jika label konsisten → keep
- Jika tidak konsisten → vote majority. Tie → drop bbox tersebut dari training (atau tag sebagai "ambiguous", train dengan label-smoothing extra)

Output: dataset baru `dataset_combined_consensus/` dengan label yang sudah dibersihkan. Re-train YOLO11l baseline → bandingkan ke AR29.

```python
# scripts/consensus_relabel.py (sketch)
"""Generate cleaned dataset berdasarkan vote majority cross-view per bunch."""
import json, shutil
from pathlib import Path
from collections import Counter

SRC_DATA = Path("data/dataset_combined")
DST_DATA = Path("data/dataset_combined_consensus")
JSON_DIRS = [...]  # sama seperti Section 27
NAMES = ["B1", "B2", "B3", "B4"]

def main():
    # 1. Build mapping: (image_filename, box_index) → consensus_class atau "DROP"
    overrides = {}  # key: (image_stem, box_idx) → class_idx (or None=drop)
    for jd in JSON_DIRS:
        for jp in Path(jd).glob("*.json"):
            data = json.loads(jp.read_text())
            for bunch in data["bunches"]:
                if bunch["appearance_count"] < 2: continue
                labels = []
                for app in bunch["appearances"]:
                    side = app["side"]; idx = app["box_index"]
                    labels.append(data["images"][side]["annotations"][idx]["class"])
                cnt = Counter(labels)
                top, n_top = cnt.most_common(1)[0]
                if list(cnt.values()).count(n_top) > 1:
                    consensus = None  # tie → drop
                else:
                    consensus = NAMES.index(top)
                for app in bunch["appearances"]:
                    side = app["side"]; idx = app["box_index"]
                    img_stem = Path(data["images"][side]["filename"]).stem
                    overrides[(img_stem, idx)] = consensus

    # 2. Copy dataset, apply overrides ke labels/*.txt
    DST_DATA.mkdir(exist_ok=True)
    n_changed, n_dropped = 0, 0
    for split in ["train", "val", "test"]:
        (DST_DATA/"images"/split).mkdir(parents=True, exist_ok=True)
        (DST_DATA/"labels"/split).mkdir(parents=True, exist_ok=True)
        for img in (SRC_DATA/"images"/split).glob("*.jpg"):
            shutil.copy2(img, DST_DATA/"images"/split/img.name)
            lbl = SRC_DATA/"labels"/split/(img.stem + ".txt")
            new_lines = []
            for i, line in enumerate(lbl.read_text().strip().splitlines()):
                parts = line.split()
                cls = int(parts[0])
                key = (img.stem, i)
                if key in overrides:
                    new_cls = overrides[key]
                    if new_cls is None:
                        n_dropped += 1; continue
                    if new_cls != cls:
                        n_changed += 1
                        parts[0] = str(new_cls)
                new_lines.append(" ".join(parts))
            (DST_DATA/"labels"/split/(img.stem + ".txt")).write_text("\n".join(new_lines))
    # Copy data.yaml dengan path direvisi
    yaml_text = (SRC_DATA/"data.yaml").read_text().replace(
        str(SRC_DATA.resolve()), str(DST_DATA.resolve()))
    (DST_DATA/"data.yaml").write_text(yaml_text)
    print(f"Changed labels: {n_changed} | Dropped (tie): {n_dropped}")

if __name__ == "__main__":
    main()
```

> **Catatan:** Hanya 228/854 pohon punya JSON, jadi consensus hanya menyentuh subset bbox. Tetap berguna sebagai **proxy**: kalau gain mAP signifikan walau hanya 26.7% data tersentuh → label noise jelas penyebab.

**Pendekatan B — Training-Time Consistency Loss (PRIORITAS 2)**

Sample batch yang berisi pasangan view dari pohon yang sama (custom sampler). Tarik logit cls antara bbox yang link via JSON:

```
L_consist(a, b) = KLDiv(softmax(logits_a / T), softmax(logits_b / T)) * T^2
L_total = L_yolo + λ_consist * L_consist
```

Default: T=2.0, λ_consist=1.0. Hanya aktif untuk bbox yang punya cross-view link di JSON.

Skema:
1. Custom `MultiViewSampler` — setiap batch sampling 50% pasangan (view_a, view_b) dari pohon dengan JSON
2. Custom `KDLoss`-style hook — setelah forward kedua view, identifikasi bbox match via JSON, hitung KLDiv pada cls logits
3. Backward gabungan

Implementasi non-trivial — perlu modifikasi Ultralytics dataloader & loss. Estimasi 1–2 hari coding.

**Pendekatan C — Multi-View Cross-Attention Head (PRIORITAS 3, stretch)**

Forward 4 view bersamaan → classification head menerima feature pool dari semua view via cross-attention. Lebih dalam tapi kompleks. Skip dulu, evaluasi setelah B mature.

### 28.2 Mengapa Pendekatan A Dulu

- Cheap (re-run baseline = sudah ada infra dari Section 16)
- Falsifiable cepat — kalau A gagal, B/C kemungkinan juga tidak break ceiling
- Tidak butuh ubah arsitektur Ultralytics

---

## 29. Eksperimen Falsifiable JSON-Aware

| Exp ID | Hipotesis falsifiable | Cara verifikasi | Slice metric utama | Cost |
|---|---|---|---|---|
| **JSON-01** | B2/B3 cross-view inconsistency rate >> B1/B4 (label-ceiling) | Run Section 27 audit script | Inconsistency rate per kelas + leak pair matrix | ~5 menit, no GPU |
| **JSON-02** | Pendekatan A (consensus relabel) menaikkan mAP50-95 vs AR29 dengan margin > bootstrap CI | Section 28 Pendekatan A, re-train YOLO11l 640 b16 (replikasi AR29 setup), val standar | mAP50-95 overall + per-class B2/B3 + bootstrap CI | ~6 jam GPU |
| **JSON-03** | Merge B2+B3 jadi 1 kelas ("B23") menaikkan mAP B1/B4 karena task lebih mudah | Train 3-class YOLO11l 640 b16 (B1, B23, B4); evaluasi mAP B1/B4 vs baseline 4-class | mAP50-95 B1, B4, B23-merged + per-domain breakdown | ~6 jam GPU |
| **JSON-04** | Pendekatan B (consistency loss) > Pendekatan A pada slice multi-view | Custom training (Section 28-B); eval pada 228 pohon dengan JSON | mAP50-95 + cross-view prediction agreement rate | ~12 jam GPU + 1-2 hari coding |
| **JSON-05** | Multi-view post-inference dedup (pipeline Section 23) menurunkan count MAE vs sum naif | Run Section 23 MultiViewAggregator; bandingkan ke baseline = sum count per view | Count MAE per kelas per pohon (228 pohon GT) | ~2 jam, no train |

### 29.1 Stop Criteria

- **JSON-01 hasil rendah inconsistency** (< 5% di B2/B3) → label-ceiling **falsified**; JSON tidak akan menyelesaikan B2/B3 confusion. Lanjut hanya JSON-05 untuk Stage 2 deliverable; angle B2/B3 tidak punya solusi struktural di workspace ini → laporkan sebagai irreducible noise.
- **JSON-01 confirmed tinggi tapi JSON-02 tidak naik** > 0.005 mAP50-95 vs AR29 → label noise nyata tapi consensus tidak cukup; coba JSON-03.
- **JSON-03 menunjukkan B1/B4 mAP naik signifikan saat B2/B3 di-merge** → pertimbangkan **task reframing**: produk akhir = 3-class (B1, B23, B4) atau ordinal regression dengan tolerance ±1 step. Diskusikan dengan stakeholder apakah deliverable bisa diubah.
- **Semua JSON-01..04 negatif** → bottleneck struktural lebih dalam dari label/multi-view (misalnya keterbatasan resolusi sensor, occlusion fundamental); rekomendasi pivot ke koleksi data tambahan atau setup berbeda (close-up shots, multi-angle drone, dll.).

### 29.2 Wajib dilakukan tiap eksperimen JSON

- **Bootstrap CI 95%** vs AR29 (n_boot=2000)
- **Per-domain breakdown** (DAMIMAS vs LONSUM) — laporkan terpisah
- **Per-class mAP50-95** — bukan hanya overall
- **Per-size bucket** untuk B4 (small/medium berdasarkan rel_area threshold dari Section 3.4 audit)

---

## 30. Updated Ablation Matrix & Decision Tree (Override Section 20)

### 30.1 Matrix Aktif

| Exp ID | Tujuan | Status | Compare to |
|---|---|---|---|
| AR29 | Baseline standard val | **Confirmed** = 0.264 mAP50-95 | – |
| AR34 | Upper-bound train+test | **Confirmed** = 0.269 mAP50-95 | – |
| **JSON-01** | Label audit cross-view | **TODO (run dulu, cheap)** | – |
| **JSON-02** | Consensus relabel re-train | TODO (kalau JSON-01 → label-ceiling) | AR29 |
| **JSON-03** | 3-class B23-merged | TODO (kalau JSON-02 belum break) | AR29 (B1/B4 only) |
| **JSON-04** | Consistency loss training | TODO (kalau JSON-02 promising) | JSON-02 |
| **JSON-05** | Multi-view inference dedup | TODO (independent, paralel) | Baseline = naive sum |

### 30.2 Decision Tree Baru

```
JSON-01  (cheap, ~5 menit, no GPU)
   │
   ├── B2/B3 inconsistency > 15% & B1/B4 < 5%
   │     → label-ceiling kuat
   │     ↓
   │   JSON-02  (consensus relabel + retrain ~6 jam)
   │     ├── mAP50-95 naik > AR29 + 0.005 (CI overlap clear)
   │     │     → adopt sebagai label cleanup baseline
   │     │     → coba JSON-04 untuk push lebih jauh
   │     │
   │     └── tidak naik / dalam noise CI
   │           ↓
   │         JSON-03  (3-class merge ~6 jam)
   │           ├── B1/B4 mAP naik signifikan saat task disederhanakan
   │           │     → reframe task ke 3-class (atau ordinal ±1 tolerance)
   │           │     → diskusi stakeholder soal deliverable
   │           │
   │           └── tetap tidak naik
   │                 → bottleneck di luar label, kemungkinan data quality / resolusi
   │                 → pivot: koleksi data baru atau setup sensor berbeda
   │
   ├── B2/B3 inconsistency ~ B1/B4 (5–10%)
   │     → label OK; angle JSON tidak menolong B2/B3
   │     ↓
   │   skip JSON-02/03/04, langsung JSON-05 untuk Stage 2 deliverable
   │   B2/B3 ceiling = irreducible noise → laporkan honest
   │
   └── inconsistency rendah semua kelas
         → JSON tidak relevan untuk label noise
         → tetap jalankan JSON-05 sebagai Stage 2 deliverable saja
```

### 30.3 Reporting Template untuk Setiap JSON-XX Run

Wajib ada di `reports/json_xx/summary.md`:

```markdown
# Exp JSON-XX
- Date: YYYY-MM-DD
- Hypothesis: [text falsifiable]
- Setup: model, imgsz, batch, epochs, label source
- Compare to: AR29 / AR34 / JSON-YY

## Results
| Metric | This run | Baseline | Δ | Bootstrap 95% CI | Significant? |
|---|---|---|---|---|---|
| mAP50-95 overall | – | 0.264 | – | [lo, hi] | yes/no |
| mAP50-95 B1 | – | – | – | – | – |
| mAP50-95 B2 | – | – | – | – | – |
| mAP50-95 B3 | – | – | – | – | – |
| mAP50-95 B4 | – | – | – | – | – |
| DAMIMAS subset mAP | – | – | – | – | – |
| LONSUM subset mAP | – | – | – | – | – |

## Decision
- [ ] Hypothesis confirmed → next: ...
- [ ] Hypothesis falsified → close branch
- [ ] Inconclusive → ...

## Caveats
- ...
```

### 30.4 Yang TIDAK Boleh Dilakukan Lagi (Reminder)

Per CONTEXT.md Section 6 — jangan re-run kombinasi knob ini tanpa angle baru:
- imgsz 800, scale 0.7, BOX/CLS/DFL tweak, lr sweep, SGD vs AdamW, copy_paste, label_smoothing, model soup, long-run brute force
- Naive oversampling B1/B4
- HSV-only branch
- SAHI pada setup lama (versi baru Section 21 dengan model JSON-aware OK untuk evaluasi B4 spesifik)
- Two-stage 4-class classifier (DINOv2 CE/CORN, EfficientNet, hierarchical)
- YOLOv9e, RT-DETR-L, RF-DETR DINOv2, YOLO11x train+test sebagai jalan utama

Section 14–22 di Bagian II tetap valid sebagai **infra reference** (env setup, mobile export, eval skrip) — bukan sebagai jalur eksperimen baru.
