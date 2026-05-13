# Penghitungan Tandan Sawit Multi-Sisi

Pipeline ini menghitung jumlah tandan unik per pohon kelapa sawit dari foto yang diambil dari 4–8 sisi berbeda.

- **Dataset:** 953 pohon (DAMIMAS 854 + LONSUM 99)
- **Kelas:** B1 → B2 → B3 → B4 (ordinal, matang → belum matang)
- **Lisensi:** CC BY-NC 4.0

---

## Ringkasan Hasil Terbaik

### Juara Per Track

| Track | Metode | Macro MAE | Macro Acc ±1 | Keterangan |
|:---|:---|---:|---:|:---|
| A. Heuristik | M01_selector_b2b3 | 0.398 | **86.67%** | ✅ Produksi (valid per RULES.txt) |
| B. Deteksi | YOLO26n (lokal) | — | mAP50 = **0.521** | ✅ Detektor terbaik lokal |
| C. ML Counting (fitur GT) | SVM RBF | **0.318** | **96.1%** | ✅ ML terbaik dengan fitur sempurna |
| D. End-to-End | y26m → SVM | 1.118 | **71.6%** | ⚠️ Terbaik E2E, masih di bawah heuristik |

### Metrik Terbaik Keseluruhan

| Metrik | Nilai | Metode |
|:---|---:|:---|
| Macro Acc ±1 tertinggi — ML (fitur GT) | **96.1%** | SVM RBF |
| Macro Acc ±1 tertinggi — heuristik valid | **86.67%** | M01_selector_b2b3 |
| Macro Acc ±1 tertinggi — end-to-end | **71.6%** | y26m → SVM |
| Macro MAE terendah — ML (fitur GT) | **0.318** | SVM RBF |
| Macro MAE terendah — heuristik valid | **0.398** | M01_selector_b2b3 |
| mAP50 terbaik (lokal, batch=16) | **0.521** | YOLO26n |
| Tercepat | **0.005 ms/pohon** | M15_divide_global |

### Temuan Utama

Pipeline ML dengan fitur GT (Track C) mengungguli heuristik terbaik M01: SVM mencapai Macro Acc±1 = **96.1%** dibandingkan **86.67%** milik M01, sehingga desain fitur 13-dim terbukti memadai apabila detektor menghasilkan deteksi yang benar.

Namun, pipeline ujung-ke-ujung (Track D) hanya mencapai Macro Acc±1 = **71.6%** pada kombinasi terbaik (y26m → SVM), karena propagasi galat detektor YOLO merusak nilai `naive_sum`, `max_per_side`, dan `mean_per_side` sebelum masuk ke algoritma penghitung — sehingga model menerima input yang tidak akurat.

Temuan penting: seluruh 15 kombinasi E2E (5 detektor × 3 algoritma penghitung) menghasilkan Macro Acc±1 dalam rentang **64–72%**, tanpa perbedaan signifikan antar algoritma penghitung. Hal ini membuktikan bahwa bottleneck terletak pada kualitas detektor, bukan pada algoritma penghitungan.

> ❌ M60 dan M53 mencapai 90.24%, tetapi keduanya **tidak valid** per `exp_12 may 2026/RULES.txt` karena menggunakan tabel divisor yang diturunkan dari statistik training split (kalibrasi domain-spesifik), bukan dari prinsip geometri murni, sehingga tidak dapat digeneralisasi ke kebun lain.

---

## Track A: Penghitungan Heuristik (Tanpa Training)

Metode heuristik bekerja langsung pada deteksi bounding box per sisi tanpa memerlukan proses pelatihan apa pun.

| Peringkat | Metode | Macro Acc ±1 | Macro MAE | Profil Tepat | Valid? |
|:---:|:---|---:|---:|---:|:---:|
| — | M60_blind_strict | 90.24% | 0.302 | — | ❌ |
| — | M53_three_band_override | 90.24% | 0.304 | — | ❌ |
| 1 | **M01_selector_b2b3** | **86.67%** | **0.398** | 26.3% | ✅ |
| 2 | M05_blend_vis_divide | 86.04% | 0.408 | 25.3% | ✅ |
| 3 | M06_weight_visibility | 85.94% | 0.396 | 25.3% | ✅ |
| 4 | M15_divide_global | 84.37% | 0.416 | 23.3% | ✅ |

Tabel lengkap 29 metode tersedia di `reports/dedup_brand_new_953/accuracy_953.csv`.

> ❌ M60 dan M53 dinyatakan tidak valid per `exp_12 may 2026/RULES.txt` karena keduanya menggunakan tabel divisor yang diturunkan dari statistik training split. Kedua metode tersebut disimpan hanya sebagai referensi historis.

```bash
python scripts/dedup_brand_new_953.py
```

---

## Track B: Deteksi Objek (YOLO26)

Seluruh eksperimen deteksi dijalankan secara lokal dengan konfigurasi yang konsisten: `batch=16`, `imgsz=640`, `epochs=100`, `patience=50`, `seed=42`.

### Perbandingan Arsitektur

| Model | mAP50 | mAP50-95 | Kecepatan | Parameter |
|:---|---:|---:|---:|---:|
| **YOLO26n** | **0.521** | **0.237** | **0.2 ms** | 2.4 M |
| YOLO26s | 0.506 | 0.235 | 0.5 ms | 9.5 M |
| YOLO26m | 0.509 | 0.231 | 0.8 ms | 20.4 M |

### Ablasi Konfigurasi (YOLO26s sebagai basis)

| Eksperimen | Best Epoch | mAP50 | mAP50-95 | Catatan |
|:---|---:|---:|---:|:---|
| y26m vanilla (lokal) | 33 | 0.509 | 0.231 | Gap kecil dari RunPod (0.528) karena perbedaan environment |
| y26s vanilla (lokal) | 21 | 0.506 | 0.234 | Digunakan sebagai baseline E2E |
| y26s tanpa pretrained | 57 | **0.511** | 0.231 | Scratch = pretrained; COCO pretraining tidak wajib |
| y26s tanpa augmentasi | 6 | 0.465 | 0.216 | Overfit, early stop pada epoch 56 |

**Insight:** YOLO26n menjadi pilihan terbaik untuk produksi — mAP50 tertinggi (0.521) dengan kecepatan 4× lebih cepat dari y26m (0.2 ms vs 0.8 ms). Augmentasi bersifat esensial: tanpa augmentasi, mAP50 turun ke 0.465 dan model overfit pada epoch ke-6.

```bash
python -c "
from ultralytics import YOLO
YOLO('yolo26n.pt').train(data='local_data.yaml', epochs=100, batch=16, imgsz=640, seed=42)
"
```

---

## Track C: Penghitungan ML (Fitur dari GT)

Setiap pohon direpresentasikan sebagai vektor fitur 13 dimensi: `naive_sum` (B1–B4), `max_per_side` (B1–B4), `mean_per_side` (B1–B4), dan `n_sides`. Fitur diekstrak dari anotasi GT yang sempurna sebagai batas atas performa ML.

| Model | Macro MAE | Macro Acc ±1 | Acc±1 B1 | Acc±1 B2 | Acc±1 B3 | Acc±1 B4 | Profil Tepat |
|:---|---:|---:|---:|---:|---:|---:|---:|
| **SVM (RBF, GridSearchCV)** | **0.318** | **96.1%** | **100.0%** | 95.8% | 91.6% | 96.8% | 27.4% |
| RF (n=200, max_depth=10) | 0.353 | 95.3% | 96.8% | 96.8% | 90.5% | 96.8% | 27.4% |

SVM (96.1%) mengungguli heuristik terbaik M01 (86.67%), yang membuktikan bahwa desain fitur 13-dim sudah memadai apabila input berupa deteksi yang sempurna. Detail tersedia di `reports/counting_{svm,rf}/metrics.json`.

```bash
python scripts/run_counting_svm.py
python scripts/run_counting_rf.py
```

---

## Track D: Pipeline Ujung-ke-Ujung (Deteksi → Penghitungan)

Setiap detektor diuji dengan tiga algoritma penghitungan: SVM (RBF, GridSearchCV), RF (n=200, max_depth=10), dan M01 heuristik. Inferensi dijalankan pada seluruh 953 pohon; SVM dan RF dilatih ulang menggunakan fitur dari masing-masing detektor. Seluruh angka dilaporkan pada test set (n=95).

| Detektor | mAP50 | Penghitung | Macro MAE | Macro Acc ±1 | Acc±1 B1 | Acc±1 B2 | Acc±1 B3 | Acc±1 B4 | Profil Tepat |
|:---|---:|:---|---:|---:|---:|---:|---:|---:|---:|
| y26n vanilla | 0.521 | SVM | 1.145 | 70.0% | 90.5% | 68.4% | 56.8% | 64.2% | 0.0% |
| y26n vanilla | 0.521 | RF | 1.218 | 68.2% | 90.5% | 68.4% | 54.7% | 58.9% | 0.0% |
| y26n vanilla | 0.521 | M01 | 1.337 | 67.1% | 87.4% | 65.3% | 51.6% | 64.2% | 2.1% |
| y26s vanilla | 0.506 | SVM | 1.163 | 68.9% | 93.7% | 68.4% | 48.4% | 65.3% | 0.0% |
| y26s vanilla | 0.506 | RF | 1.216 | 66.6% | 96.8% | 68.4% | 48.4% | 52.6% | 1.1% |
| y26s vanilla | 0.506 | M01 | 1.403 | 65.5% | 89.5% | 66.3% | 38.9% | 67.4% | 2.1% |
| y26s scratch | 0.511 | SVM | 1.145 | 68.9% | 90.5% | 68.4% | 51.6% | 65.3% | 2.1% |
| y26s scratch | 0.511 | RF | 1.229 | 67.9% | 93.7% | 65.3% | 55.8% | 56.8% | 1.1% |
| y26s scratch | 0.511 | M01 | 1.266 | 69.2% | 91.6% | 63.2% | 52.6% | 69.5% | 2.1% |
| y26s no-aug | 0.465 | SVM | 1.126 | 70.5% | 91.6% | 69.5% | 56.8% | 64.2% | 1.1% |
| y26s no-aug | 0.465 | RF | 1.184 | 68.4% | 92.6% | 66.3% | 55.8% | 58.9% | 1.1% |
| y26s no-aug | 0.465 | M01 | 1.384 | 66.6% | 90.5% | 68.4% | 43.2% | 64.2% | 0.0% |
| **y26m vanilla** | 0.509 | **SVM** | **1.118** | **71.6%** | 92.6% | 63.2% | 60.0% | 70.5% | 2.1% |
| y26m vanilla | 0.509 | RF | 1.211 | 67.9% | 95.8% | 68.4% | 49.5% | 57.9% | 0.0% |
| y26m vanilla | 0.509 | M01 | 1.400 | 64.5% | 90.5% | 56.8% | 40.0% | 70.5% | 0.0% |
| **M01 heuristik (fitur GT — target)** | — | — | **0.398** | **86.7%** | — | — | — | — | 26.3% |

**Bottleneck:** Seluruh 15 kombinasi detektor × penghitung menghasilkan Macro Acc±1 dalam rentang sempit **64–72%**, jauh di bawah M01 berbasis GT (86.7%). Pilihan algoritma penghitung (SVM, RF, atau M01) tidak mengubah kesimpulan secara signifikan — bottleneck sejati adalah propagasi galat YOLO ke nilai `naive_sum`, `max_per_side`, dan `mean_per_side`. Sebagai pembanding, SVM dengan fitur GT mencapai 96.1% (Track C) menggunakan arsitektur fitur yang identik.

**Temuan tak terduga:** y26s-noaug (mAP50=0.465, detektor terlemah) menghasilkan SVM 70.5%, hanya 1.1 pp di bawah y26m (mAP50=0.509, SVM 71.6%). Hal ini mengindikasikan bahwa distribusi galat detektor — bukan besarnya mAP — yang menentukan kualitas fitur 13-dim untuk penghitungan.

```bash
# Jalankan pipeline E2E untuk satu detektor (inferensi + SVM + RF + M01):
python scripts/run_e2e_pipeline.py \
    --name y26m_vanilla_local \
    --weights baseline-run/weights/y26m_vanilla_local.pt
```

---

## Kesimpulan

| Kasus Penggunaan | Rekomendasi |
|:---|:---|
| Penghitungan produksi | **M01_selector_b2b3** — Macro Acc±1 = 86.67%, valid per RULES.txt |
| Deteksi saja (akurasi) | **YOLO26m** — mAP50 = 0.509 |
| Deteksi saja (kecepatan) | **YOLO26n** — mAP50 = 0.521, 0.2 ms/gambar |
| Baseline riset ML | **SVM pada fitur GT** — Macro Acc±1 = 96.1%, Macro MAE = 0.318 |
| Pipeline E2E terbaik | **y26m → SVM** — Macro Acc±1 = 71.6%, masih 15 pp di bawah heuristik |

---

## Panduan Cepat

```bash
# 1. Instalasi dependensi
pip install -r requirements.txt
pip install scikit-learn ultralytics huggingface_hub

# 2. Unduh dataset dari HuggingFace
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'ULM-DS-Lab/OilPalm-MultiView-BunchCount-YOLO',
    repo_type='dataset',
    local_dir='./Tested-Brand-New-Dataset-YOLO'
)"

# 3. Jalankan semua track
python scripts/dedup_brand_new_953.py    # Track A: heuristik
python scripts/run_counting_svm.py       # Track C: SVM dari fitur GT
python scripts/run_counting_rf.py        # Track C: RF dari fitur GT

# Track B: training detektor
python -c "from ultralytics import YOLO; YOLO('yolo26n.pt').train(
    data='local_data.yaml', epochs=100, batch=16, imgsz=640, seed=42,
    project='/workspace/runs/detect', name='y26n_vanilla_local')"

# Track D: pipeline E2E (inferensi + SVM + RF + M01 sekaligus)
python scripts/run_e2e_pipeline.py \
    --name y26n_vanilla_local \
    --weights baseline-run/weights/y26n_vanilla_local.pt
```

---

## Tautan

- [`RESEARCH.md`](RESEARCH.md) — Dokumen riset lengkap
- [`exp_12 may 2026/REPORT.md`](exp_12%20may%202026/REPORT.md) — Analisis mendalam M60
- [`baseline-run/SUMMARY.md`](baseline-run/SUMMARY.md) — Ringkasan hasil ML
- [`CLAUDE-TRAINING.md`](CLAUDE-TRAINING.md) — Panduan eksperimen ML di RunPod/Vast.ai
- [`exp_13 May 2026/PROGRESS.md`](exp_13%20May%202026/PROGRESS.md) — Log progres training
- [Dataset HuggingFace](https://huggingface.co/datasets/ULM-DS-Lab/OilPalm-MultiView-BunchCount-YOLO)

---

## Sitasi

```bibtex
@dataset{palm_bunch_2026,
  title   = {Multi-View Oil Palm Bunch Dataset},
  author  = {Muttaqin, M. Zainal},
  year    = {2026},
  publisher = {HuggingFace},
  url     = {https://huggingface.co/datasets/ULM-DS-Lab/OilPalm-MultiView-BunchCount-YOLO}
}
```

Lisensi: **CC BY-NC 4.0**
