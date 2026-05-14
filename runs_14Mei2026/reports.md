# Laporan Hasil Runs — 14 Mei 2026

Pipeline ini menghitung jumlah tandan unik per pohon kelapa sawit dari foto 4–8 sisi.

- **Dataset:** 953 pohon (DAMIMAS 854 + LONSUM 99) — dataset bersih, re-verified vs HuggingFace
- **Kelas:** B1 → B2 → B3 → B4 (ordinal, matang → belum matang)
- **Tanggal run:** 14 Mei 2026 (rerun penuh dari dataset yang sebelumnya corrupt)

---

## Ringkasan Juara Per Track

| Track | Metode | Macro MAE | Macro Acc ±1 | Keterangan |
|:---|:---|---:|---:|:---|
| A. Heuristik | [M02_selector_trifurc](reports/dedup_brand_new_953/accuracy_953.csv) | 0.388 | **86.88%** | ✅ Top baru (dataset bersih) |
| B. Deteksi | [YOLO26n vanilla](training/weights/y26n_vanilla_local_results.csv) | — | mAP50 = **0.521** | ✅ Tercepat & terbaik |
| C. ML Counting (fitur GT) | [Linear Regression](reports/counting_lr/metrics.json) | **0.287** | **96.8%** | ✅ Terbaik (mengalahkan SVM/RF) |
| D. E2E per-pohon | [y26m → SVM](reports/e2e_y26m_vanilla_local_svm/metrics.json) | **1.097** | **71.6%** | ✅ Best E2E |
| E. E2E per-gambar *(baru)* | [y26m vanilla](reports/e2e_per_image_y26m_vanilla_local/metrics.json) | **0.605** | **87.5%** | ✅ Jalur Baru |

---

## Track A: Heuristik Dedup (Tanpa Training)

Benchmark pada 953 pohon canonical (`Brand-New-Dataset-YOLO/`). Semua metode deterministik, tanpa training.

| Peringkat | Metode | Macro Acc ±1 | Macro MAE | Total ±1 | Profil Tepat |
|:---:|:---|---:|---:|---:|---:|
| 1 | [**M02_selector_trifurc**](reports/dedup_brand_new_953/accuracy_953.csv) | **86.88%** | 0.388 | 74.19% | 26.34% |
| 2 | [M01_selector_b2b3](reports/dedup_brand_new_953/accuracy_953.csv) | 86.78% | 0.388 | 74.19% | 26.34% |
| 3 | [M03_blend_geometric](reports/dedup_brand_new_953/accuracy_953.csv) | 86.36% | 0.388 | 74.61% | 26.86% |
| 4 | [M04_blend_floor_clamped](reports/dedup_brand_new_953/accuracy_953.csv) | 86.25% | 0.396 | 74.29% | 25.81% |
| 5 | [M05_blend_vis_divide](reports/dedup_brand_new_953/accuracy_953.csv) | 86.25% | 0.399 | 74.08% | 25.29% |

Tabel lengkap 29 metode: [`reports/dedup_brand_new_953/accuracy_953.csv`](reports/dedup_brand_new_953/accuracy_953.csv)  
Detail per pohon: [`reports/dedup_brand_new_953/per_tree.csv`](reports/dedup_brand_new_953/per_tree.csv)  
Evaluasi 4 dimensi: [`reports/benchmark_multidim/REPORT.md`](reports/benchmark_multidim/REPORT.md)

> **Catatan:** M02 menggeser M01 sebagai top setelah dataset bersih. Perbedaan sangat kecil (0.10 pp) — keduanya pada dasarnya setara.

```bash
python scripts/dedup_brand_new_953.py
```

---

## Track B: Deteksi Objek (YOLO26)

Semua eksperimen dijalankan lokal: `batch=16`, `imgsz=640`, `epochs=100`, `patience=50`, `seed=42`.

### Perbandingan Arsitektur

| Model | mAP50 | mAP50-95 | Best Epoch | Parameter | Training Log | Results |
|:---|---:|---:|---:|---:|:---|:---|
| [**YOLO26n**](training/weights/y26n_vanilla_local_args.yaml) | **0.521** | **0.237** | 38 | 2.4M | [log](training/logs/y26n_vanilla_local.txt) | [results.csv](training/weights/y26n_vanilla_local_results.csv) |
| [YOLO26s](training/weights/y26s_vanilla_local_args.yaml) | 0.507 | 0.225 | 31 | 9.5M | [log](training/logs/y26s_vanilla_local.txt) | [results.csv](training/weights/y26s_vanilla_local_results.csv) |
| [YOLO26m](training/weights/y26m_vanilla_local_args.yaml) | 0.509 | 0.231 | 33 | 20.4M | [log](training/logs/y26m_vanilla_local_retrain.txt) | [results.csv](training/weights/y26m_vanilla_local_results.csv) |

### Ablasi Konfigurasi (basis: YOLO26s)

| Eksperimen | Best Epoch | mAP50 | mAP50-95 | Log | Results | Catatan |
|:---|---:|---:|---:|:---|:---|:---|
| [y26s vanilla](training/weights/y26s_vanilla_local_args.yaml) | 31 | 0.507 | 0.225 | [log](training/logs/y26s_vanilla_local.txt) | [csv](training/weights/y26s_vanilla_local_results.csv) | Baseline |
| [y26s no-pretrained](training/weights/y26s_nopretrained_args.yaml) | 57 | 0.511 | 0.231 | [log](training/logs/y26s_nopretrained.txt) | [csv](training/weights/y26s_nopretrained_results.csv) | Scratch ≈ pretrained |
| [y26s no-augmentation](training/weights/y26s_noaug_args.yaml) | 6 | 0.465 | 0.216 | [log](training/logs/y26s_noaug.txt) | [csv](training/weights/y26s_noaug_results.csv) | Overfit, early stop |

**Insight:** YOLO26n pilihan terbaik — mAP50 tertinggi (0.521) dengan kecepatan 4× lebih cepat dari y26m. Augmentasi esensial: tanpa augmentasi mAP50 turun ke 0.465 dan overfit di epoch 6.

---

## Track C: ML Counting (Fitur dari GT)

Setiap pohon direpresentasikan sebagai vektor 13 dimensi: `naive_sum` (B1–B4), `max_per_side` (B1–B4), `mean_per_side` (B1–B4), dan `n_sides`. Fitur dari GT sempurna = batas atas performa ML.

| Model | Macro MAE | Macro Acc ±1 | Acc±1 B1 | Acc±1 B2 | Acc±1 B3 | Acc±1 B4 | Profil Tepat | Detail |
|:---|---:|---:|---:|---:|---:|---:|---:|:---|
| [**Linear Regression**](reports/counting_lr/metrics.json) | **0.287** | **96.8%** | **100.0%** | 96.8% | 92.6% | 97.9% | **36.8%** | [predictions](reports/counting_lr/predictions.csv) |
| [SVM (RBF, GridSearchCV)](reports/counting_svm/metrics.json) | 0.318 | 96.1% | 100.0% | 95.8% | 91.6% | 96.8% | 27.4% | [predictions](reports/counting_svm/predictions.csv) |
| [RF (n=200, max_depth=10)](reports/counting_rf/metrics.json) | 0.353 | 95.3% | 96.8% | 96.8% | 90.5% | 96.8% | 27.4% | [predictions](reports/counting_rf/predictions.csv) · [feature importance](reports/counting_rf/feature_importance.csv) |

> **Temuan mengejutkan:** [Linear Regression](reports/counting_lr/metrics.json) (96.8%) **mengalahkan SVM dan RF** — mengindikasikan hubungan antara fitur 13-dim dan target count bersifat cukup linear. Exact-profile accuracy LR (36.8%) juga jauh lebih tinggi, artinya prediksi vektor `[B1,B2,B3,B4]` lebih sering tepat secara simultan.

```bash
python scripts/run_counting_svm.py    # → reports/counting_svm/
python scripts/run_counting_rf.py     # → reports/counting_rf/
python scripts/run_counting_lr.py     # → reports/counting_lr/
```

---

## Track D: Pipeline E2E per-Pohon (Deteksi → Penghitungan)

Setiap detektor diuji dengan tiga algoritma penghitungan. Inferensi pada seluruh 953 pohon; SVM/RF dilatih ulang dari fitur detektor. Angka pada test set (n=95 pohon).

| Detektor | mAP50 | Penghitung | Macro Acc ±1 | Macro MAE | Detail |
|:---|---:|:---|---:|---:|:---|
| [**y26m vanilla**](training/weights/y26m_vanilla_local.pt) | 0.509 | [**SVM**](reports/e2e_y26m_vanilla_local_svm/metrics.json) | **71.6%** | **1.097** | [predictions](reports/e2e_y26m_vanilla_local_svm/predictions.csv) |
| [y26s vanilla](training/weights/y26s_vanilla_local.pt) | 0.507 | [RF](reports/e2e_y26s_vanilla_local_rf/metrics.json) | 71.3% | 1.158 | [predictions](reports/e2e_y26s_vanilla_local_rf/predictions.csv) |
| [y26n vanilla](training/weights/y26n_vanilla_local.pt) | 0.521 | [SVM](reports/e2e_y26n_vanilla_local_svm/metrics.json) | 70.8% | 1.124 | [predictions](reports/e2e_y26n_vanilla_local_svm/predictions.csv) |
| [y26s vanilla](training/weights/y26s_vanilla_local.pt) | 0.507 | [SVM](reports/e2e_y26s_vanilla_local_svm/metrics.json) | 70.8% | 1.108 | [predictions](reports/e2e_y26s_vanilla_local_svm/predictions.csv) |
| [y26s nopretrained](training/weights/y26s_nopretrained.pt) | 0.511 | [SVM](reports/e2e_y26s_nopretrained_svm/metrics.json) | 70.5% | 1.150 | [predictions](reports/e2e_y26s_nopretrained_svm/predictions.csv) |
| [y26s noaug](training/weights/y26s_noaug.pt) | 0.465 | [SVM](reports/e2e_y26s_noaug_svm/metrics.json) | 70.5% | 1.126 | [predictions](reports/e2e_y26s_noaug_svm/predictions.csv) |
| [y26m vanilla](training/weights/y26m_vanilla_local.pt) | 0.509 | [RF](reports/e2e_y26m_vanilla_local_rf/metrics.json) | 68.9% | 1.203 | [predictions](reports/e2e_y26m_vanilla_local_rf/predictions.csv) |
| [y26s nopretrained](training/weights/y26s_nopretrained.pt) | 0.511 | [RF](reports/e2e_y26s_nopretrained_rf/metrics.json) | 67.6% | 1.216 | [predictions](reports/e2e_y26s_nopretrained_rf/predictions.csv) |
| [y26s noaug](training/weights/y26s_noaug.pt) | 0.465 | [RF](reports/e2e_y26s_noaug_rf/metrics.json) | 68.4% | 1.184 | [predictions](reports/e2e_y26s_noaug_rf/predictions.csv) |
| [y26n vanilla](training/weights/y26n_vanilla_local.pt) | 0.521 | [RF](reports/e2e_y26n_vanilla_local_rf/metrics.json) | 67.1% | 1.176 | [predictions](reports/e2e_y26n_vanilla_local_rf/predictions.csv) |
| [y26m vanilla](training/weights/y26m_vanilla_local.pt) | 0.509 | [M01](reports/e2e_y26m_vanilla_local_m01/metrics.json) | 69.7% | 1.345 | [predictions](reports/e2e_y26m_vanilla_local_m01/predictions.csv) |
| [y26s vanilla](training/weights/y26s_vanilla_local.pt) | 0.507 | [M01](reports/e2e_y26s_vanilla_local_m01/metrics.json) | 66.6% | 1.321 | [predictions](reports/e2e_y26s_vanilla_local_m01/predictions.csv) |
| [y26s noaug](training/weights/y26s_noaug.pt) | 0.465 | [M01](reports/e2e_y26s_noaug_m01/metrics.json) | 66.6% | 1.384 | [predictions](reports/e2e_y26s_noaug_m01/predictions.csv) |
| [y26s nopretrained](training/weights/y26s_nopretrained.pt) | 0.511 | [M01](reports/e2e_y26s_nopretrained_m01/metrics.json) | 65.5% | 1.350 | [predictions](reports/e2e_y26s_nopretrained_m01/predictions.csv) |
| [y26n vanilla](training/weights/y26n_vanilla_local.pt) | 0.521 | [M01](reports/e2e_y26n_vanilla_local_m01/metrics.json) | 65.5% | 1.413 | [predictions](reports/e2e_y26n_vanilla_local_m01/predictions.csv) |
| **M01 heuristik (GT — target)** | — | [accuracy_953.csv](reports/dedup_brand_new_953/accuracy_953.csv) | **86.88%** | **0.388** | — |

**Bottleneck:** Seluruh 15 kombinasi berada dalam rentang **65–72%**, jauh di bawah heuristik berbasis GT (86.88%). Pilihan algoritma penghitung (SVM/RF/M01) tidak mengubah kesimpulan — bottleneck ada di propagasi error detektor YOLO ke fitur `naive_sum`, `max_per_side`, `mean_per_side`.

```bash
python scripts/run_e2e_pipeline.py --name y26m_vanilla_local \
    --weights runs_14Mei2026/training/weights/y26m_vanilla_local.pt
```

---

## Track E: E2E per-Gambar *(Baru)*

Evaluasi disederhanakan: YOLO deteksi per gambar → hitung deteksi langsung → bandingkan dengan GT per gambar (anotasi per sisi dari JSON). Tidak ada agregasi multi-sisi.

**Test set:** 688 gambar (n_images dari split test 412 pohon × rata-rata sisi).

| Detektor | mAP50 | Macro Acc ±1 | Macro MAE | Detail |
|:---|---:|---:|---:|:---|
| [**y26m vanilla**](reports/e2e_per_image_y26m_vanilla_local/metrics.json) | 0.509 | **87.5%** | **0.605** | [predictions](reports/e2e_per_image_y26m_vanilla_local/predictions.csv) |
| [y26s vanilla](reports/e2e_per_image_y26s_vanilla_local/metrics.json) | 0.507 | 87.4% | 0.614 | [predictions](reports/e2e_per_image_y26s_vanilla_local/predictions.csv) |
| [y26s noaug](reports/e2e_per_image_y26s_noaug/metrics.json) | 0.465 | 86.8% | 0.640 | [predictions](reports/e2e_per_image_y26s_noaug/predictions.csv) |
| [y26n vanilla](reports/e2e_per_image_y26n_vanilla_local/metrics.json) | 0.521 | 85.4% | 0.665 | [predictions](reports/e2e_per_image_y26n_vanilla_local/predictions.csv) |
| [y26s nopretrained](reports/e2e_per_image_y26s_nopretrained/metrics.json) | 0.511 | 84.7% | 0.699 | [predictions](reports/e2e_per_image_y26s_nopretrained/predictions.csv) |

**Temuan:** Per-image (87.5%) jauh lebih tinggi dari E2E per-pohon (71.6%) karena tidak ada error agregasi multi-sisi. Ini menunjukkan bahwa YOLO cukup akurat per gambar, namun agregasi lintas sisi menambah kompleksitas dan error.

```bash
python scripts/run_e2e_per_image.py --name y26m_vanilla_local \
    --weights runs_14Mei2026/training/weights/y26m_vanilla_local.pt
```

---

## Kesimpulan

| Kasus Penggunaan | Rekomendasi |
|:---|:---|
| Penghitungan produksi (heuristik) | [**M02_selector_trifurc**](reports/dedup_brand_new_953/accuracy_953.csv) — Macro Acc±1 = 86.88% |
| Deteksi saja (akurasi) | [**YOLO26n**](training/weights/y26n_vanilla_local_results.csv) — mAP50 = 0.521, tercepat |
| Baseline riset ML (fitur GT) | [**Linear Regression**](reports/counting_lr/metrics.json) — Macro Acc±1 = 96.8%, MAE = 0.287 |
| Pipeline E2E per-pohon terbaik | [**y26m → SVM**](reports/e2e_y26m_vanilla_local_svm/metrics.json) — Macro Acc±1 = 71.6% |
| Pipeline E2E per-gambar terbaik | [**y26m vanilla**](reports/e2e_per_image_y26m_vanilla_local/metrics.json) — Macro Acc±1 = 87.5% |

---

## Struktur Folder

```
runs_14Mei2026/
├── reports.md                    ← dokumen ini
├── reports/                      ← semua output benchmark (36 folder)
│   ├── dedup_brand_new_953/      ← Track A: heuristik 29 metode
│   ├── benchmark_multidim/       ← Track A: evaluasi 4 dimensi
│   ├── counting_svm/             ← Track C: SVM
│   ├── counting_rf/              ← Track C: RF
│   ├── counting_lr/              ← Track C: Linear Regression (baru)
│   ├── e2e_y26*/                 ← Track D: 15 kombinasi per-pohon
│   └── e2e_per_image_y26*/       ← Track E: per-gambar (baru)
├── training/
│   ├── runs/                     ← 5 YOLO training runs (plots, CM, results.csv)
│   ├── weights/                  ← 5 × best.pt + args.yaml + results.csv
│   └── logs/                     ← 15 training log .txt
├── predictions/                  ← 5 detektor × 953 JSON = 4765 file
└── scripts/                      ← run_counting_lr.py, run_e2e_per_image.py, _verify_hf.py
```

---

## Reproduksi

```bash
# Track A (tanpa images, ~5 menit)
pip install -r requirements.txt
python scripts/dedup_brand_new_953.py
python scripts/benchmark_multidim.py
python scripts/generate_method_reports.py

# Track C (tanpa images, ~2 menit)
python scripts/run_counting_svm.py
python scripts/run_counting_rf.py
python scripts/run_counting_lr.py   # baru

# Track D & E (butuh images + GPU)
python scripts/setup_dataset.py     # download images dari HuggingFace
python scripts/run_e2e_pipeline.py --name y26m_vanilla_local \
    --weights runs_14Mei2026/training/weights/y26m_vanilla_local.pt
python scripts/run_e2e_per_image.py --name y26m_vanilla_local \
    --weights runs_14Mei2026/training/weights/y26m_vanilla_local.pt
```

**Lisensi dataset:** CC BY-NC 4.0 · [HuggingFace](https://huggingface.co/datasets/ULM-DS-Lab/OilPalm-MultiView-BunchCount-YOLO)
