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
| A. Heuristik | [M01_selector_b2b3](algorithms/M01_selector_b2b3.py) | [0.375](reports/dedup_brand_new_953/accuracy_953.csv) | [**87.62%**](reports/dedup_brand_new_953/accuracy_953.csv) | ✅ Produksi (valid per [RULES.txt](archive/_to_review/exp_12%20may%202026/RULES.txt)) — post GT-fix 2026-05-16 |
| B. Deteksi | [YOLO26n (lokal)](ml-track/baseline-run/weights/y26n_vanilla_local_args.yaml) | — | mAP50 = [**0.521**](ml-track/baseline-run/weights/y26n_vanilla_local_results.csv) | ✅ Detektor terbaik lokal |
| C. ML Counting (fitur GT) | [SVM RBF](reports/counting_svm/metrics.json) | [**0.318**](reports/counting_svm/metrics.json) | [**96.1%**](reports/counting_svm/metrics.json) | ✅ ML terbaik dengan fitur sempurna |
| D. End-to-End | [y26m → SVM](reports/e2e_y26m_vanilla_local_svm/metrics.json) | [1.118](reports/e2e_y26m_vanilla_local_svm/metrics.json) | [**71.6%**](reports/e2e_y26m_vanilla_local_svm/metrics.json) | ⚠️ Terbaik E2E, masih di bawah heuristik |

### Metrik Terbaik Keseluruhan

| Metrik | Nilai | Metode |
|:---|---:|:---|
| Macro Acc ±1 tertinggi — ML (fitur GT) | [**96.1%**](reports/counting_svm/metrics.json) | [SVM RBF](reports/counting_svm/metrics.json) |
| Macro Acc ±1 tertinggi — heuristik valid | [**87.62%**](reports/dedup_brand_new_953/accuracy_953.csv) | [M01_selector_b2b3](algorithms/M01_selector_b2b3.py) |
| Macro Acc ±1 tertinggi — end-to-end | [**71.6%**](reports/e2e_y26m_vanilla_local_svm/metrics.json) | [y26m → SVM](reports/e2e_y26m_vanilla_local_svm/metrics.json) |
| Macro MAE terendah — ML (fitur GT) | [**0.318**](reports/counting_svm/metrics.json) | [SVM RBF](reports/counting_svm/metrics.json) |
| Macro MAE terendah — heuristik valid | [**0.368**](reports/dedup_brand_new_953/accuracy_953.csv) | [M07_weight_coverage](algorithms/M07_weight_coverage.py) |
| mAP50 terbaik (lokal, batch=16) | [**0.521**](ml-track/baseline-run/weights/y26n_vanilla_local_results.csv) | [YOLO26n](ml-track/baseline-run/weights/y26n_vanilla_local_args.yaml) |
| Tercepat | **0.005 ms/pohon** | M15_divide_global |

### Temuan Utama

Pipeline ML dengan fitur GT (Track C) mengungguli heuristik terbaik M01: SVM mencapai Macro Acc±1 = [**96.1%**](reports/counting_svm/metrics.json) dibandingkan [**86.67%**](reports/dedup_brand_new_953/accuracy_953.csv) milik M01, sehingga desain fitur 13-dim terbukti memadai apabila detektor menghasilkan deteksi yang benar.

Namun, pipeline ujung-ke-ujung (Track D) hanya mencapai Macro Acc±1 = [**71.6%**](reports/e2e_y26m_vanilla_local_svm/metrics.json) pada kombinasi terbaik (y26m → SVM), karena propagasi galat detektor YOLO merusak nilai `naive_sum`, `max_per_side`, dan `mean_per_side` sebelum masuk ke algoritma penghitung — sehingga model menerima input yang tidak akurat.

Temuan penting: seluruh [15 kombinasi E2E](ml-track/baseline-run/SUMMARY.md) (5 detektor × 3 algoritma penghitung) menghasilkan Macro Acc±1 dalam rentang **64–72%**, tanpa perbedaan signifikan antar algoritma penghitung. Hal ini membuktikan bahwa bottleneck terletak pada kualitas detektor, bukan pada algoritma penghitungan.

> ❌ M60 dan M53 mencapai 90.24%, tetapi keduanya **tidak valid** per [`archive/_to_review/exp_12 may 2026/RULES.txt`](archive/_to_review/exp_12%20may%202026/RULES.txt) karena menggunakan tabel divisor yang diturunkan dari statistik training split (kalibrasi domain-spesifik), bukan dari prinsip geometri murni, sehingga tidak dapat digeneralisasi ke kebun lain.

---

## Track A: Penghitungan Heuristik (Tanpa Training)

Metode heuristik bekerja langsung pada deteksi bounding box per sisi tanpa memerlukan proses pelatihan apa pun.

Tabel post GT-fix 2026-05-16 (semua metode naik ~0.8–1.6 pp setelah cleanup 48 trees GT — 8 wrap-around + 9 8-side over-link + 31 4-side auto-heal):

| Peringkat | Metode | Macro Acc ±1 | Macro MAE | Profil Tepat | Valid? |
|:---:|:---|---:|---:|---:|:---:|
| — | M60_blind_strict | 90.24% | 0.302 | — | ❌ |
| — | M53_three_band_override | 90.24% | 0.304 | — | ❌ |
| 1 | [**M01_selector_b2b3**](algorithms/M01_selector_b2b3.py) | [**87.62%**](reports/dedup_brand_new_953/accuracy_953.csv) | [0.375](reports/dedup_brand_new_953/accuracy_953.csv) | 27.1% | ✅ |
| 2 | M05_blend_vis_divide | 86.99% | 0.388 | 26.0% | ✅ |
| 3 | M06_weight_visibility | 86.88% | 0.371 | 26.0% | ✅ |
| 4 | [M07_weight_coverage](algorithms/M07_weight_coverage.py) | 86.88% | [**0.368**](reports/dedup_brand_new_953/accuracy_953.csv) | 26.6% | ✅ |
| 5 | M15_divide_global | 85.94% | 0.391 | 23.5% | ✅ |

Tabel lengkap 29 metode tersedia di [`reports/dedup_brand_new_953/accuracy_953.csv`](reports/dedup_brand_new_953/accuracy_953.csv).

> ❌ M60 dan M53 dinyatakan tidak valid per [`archive/_to_review/exp_12 may 2026/RULES.txt`](archive/_to_review/exp_12%20may%202026/RULES.txt) karena keduanya menggunakan tabel divisor yang diturunkan dari statistik training split. Kedua metode tersebut disimpan hanya sebagai referensi historis.

```bash
python scripts/dedup_brand_new_953.py
```

---

## Track B: Deteksi Objek (YOLO26)

Seluruh eksperimen deteksi dijalankan secara lokal dengan konfigurasi yang konsisten: `batch=16`, `imgsz=640`, `epochs=100`, `patience=50`, `seed=42`.

### Perbandingan Arsitektur

| Model | mAP50 | mAP50-95 | Kecepatan | Parameter | Bukti |
|:---|---:|---:|---:|---:|:---|
| [**YOLO26n**](ml-track/baseline-run/weights/y26n_vanilla_local_args.yaml) | [**0.521**](ml-track/baseline-run/weights/y26n_vanilla_local_results.csv) | **0.237** | **0.2 ms** | 2.4 M | [log](ml-track/baseline-run/y26n_vanilla_local_b16.txt) · [cm](ml-track/baseline-run/weights/y26n_vanilla_local/confusion_matrix_normalized.png) |
| [YOLO26s](ml-track/baseline-run/weights/y26s_vanilla_local_args.yaml) | [0.506](ml-track/baseline-run/weights/y26s_vanilla_local_results.csv) | 0.235 | 0.5 ms | 9.5 M | [log](ml-track/baseline-run/y26s_vanilla_local.txt) · [cm](ml-track/baseline-run/weights/y26s_vanilla_local/confusion_matrix_normalized.png) |
| [YOLO26m](ml-track/baseline-run/weights/y26m_vanilla_local_args.yaml) | [0.509](ml-track/baseline-run/weights/y26m_vanilla_local_results.csv) | 0.231 | 0.8 ms | 20.4 M | [log](ml-track/baseline-run/y26m_vanilla_local_retrain.txt) · [cm](ml-track/baseline-run/weights/y26m_vanilla_local/confusion_matrix_normalized.png) |

### Ablasi Konfigurasi (YOLO26s sebagai basis)

| Eksperimen | Best Epoch | mAP50 | mAP50-95 | Catatan |
|:---|---:|---:|---:|:---|
| [y26m vanilla (lokal)](ml-track/baseline-run/weights/y26m_vanilla_local_args.yaml) | [33](ml-track/baseline-run/weights/y26m_vanilla_local_results.csv) | [0.509](ml-track/baseline-run/weights/y26m_vanilla_local_results.csv) | 0.231 | [log](ml-track/baseline-run/y26m_vanilla_local_retrain.txt) · [cm](ml-track/baseline-run/weights/y26m_vanilla_local/confusion_matrix_normalized.png) · Gap kecil dari RunPod (0.528) karena perbedaan environment |
| [y26s vanilla (lokal)](ml-track/baseline-run/weights/y26s_vanilla_local_args.yaml) | [21](ml-track/baseline-run/weights/y26s_vanilla_local_results.csv) | [0.506](ml-track/baseline-run/weights/y26s_vanilla_local_results.csv) | 0.234 | [log](ml-track/baseline-run/y26s_vanilla_local.txt) · [cm](ml-track/baseline-run/weights/y26s_vanilla_local/confusion_matrix_normalized.png) · Digunakan sebagai baseline E2E |
| [y26s tanpa pretrained](ml-track/baseline-run/weights/y26s_nopretrained_args.yaml) | [57](ml-track/baseline-run/weights/y26s_nopretrained_results.csv) | [**0.511**](ml-track/baseline-run/weights/y26s_nopretrained_results.csv) | 0.231 | [log](ml-track/baseline-run/y26s_nopretrained.txt) · [cm](ml-track/baseline-run/weights/y26s_nopretrained/confusion_matrix_normalized.png) · Scratch = pretrained; COCO pretraining tidak wajib |
| [y26s tanpa augmentasi](ml-track/baseline-run/weights/y26s_noaug_args.yaml) | [6](ml-track/baseline-run/weights/y26s_noaug_results.csv) | [0.465](ml-track/baseline-run/weights/y26s_noaug_results.csv) | 0.216 | [log](ml-track/baseline-run/y26s_noaug.txt) · [cm](ml-track/baseline-run/weights/y26s_noaug/confusion_matrix_normalized.png) · Overfit, early stop pada epoch 56 |

**Insight:** [YOLO26n](ml-track/baseline-run/weights/y26n_vanilla_local_args.yaml) menjadi pilihan terbaik untuk produksi — mAP50 tertinggi ([0.521](ml-track/baseline-run/weights/y26n_vanilla_local_results.csv)) dengan kecepatan 4× lebih cepat dari y26m (0.2 ms vs 0.8 ms). Augmentasi bersifat esensial: tanpa augmentasi, mAP50 turun ke [0.465](ml-track/baseline-run/weights/y26s_noaug_results.csv) dan model overfit pada epoch ke-6.

```bash
python -c "
from ultralytics import YOLO
YOLO('yolo26n.pt').train(data='ml-track/local_data.yaml', epochs=100, batch=16, imgsz=640, seed=42)
"
```

---

## Track C: Penghitungan ML (Fitur dari GT)

Setiap pohon direpresentasikan sebagai vektor fitur 13 dimensi: `naive_sum` (B1–B4), `max_per_side` (B1–B4), `mean_per_side` (B1–B4), dan `n_sides`. Fitur diekstrak dari anotasi GT yang sempurna sebagai batas atas performa ML.

| Model | Macro MAE | Macro Acc ±1 | Acc±1 B1 | Acc±1 B2 | Acc±1 B3 | Acc±1 B4 | Profil Tepat |
|:---|---:|---:|---:|---:|---:|---:|---:|
| [**SVM (RBF, GridSearchCV)**](reports/counting_svm/metrics.json) | [**0.318**](reports/counting_svm/metrics.json) | [**96.1%**](reports/counting_svm/metrics.json) | **100.0%** | 95.8% | 91.6% | 96.8% | 27.4% |
| [RF (n=200, max_depth=10)](reports/counting_rf/metrics.json) | [0.353](reports/counting_rf/metrics.json) | [95.3%](reports/counting_rf/metrics.json) | 96.8% | 96.8% | 90.5% | 96.8% | 27.4% |

SVM ([96.1%](reports/counting_svm/metrics.json)) mengungguli heuristik terbaik M01 ([86.67%](reports/dedup_brand_new_953/accuracy_953.csv)), yang membuktikan bahwa desain fitur 13-dim sudah memadai apabila input berupa deteksi yang sempurna. Detail tersedia di [`reports/counting_svm/metrics.json`](reports/counting_svm/metrics.json) dan [`reports/counting_rf/metrics.json`](reports/counting_rf/metrics.json).

```bash
python scripts/run_counting_svm.py
python scripts/run_counting_rf.py
```

---

## Track D: Pipeline Ujung-ke-Ujung (Deteksi → Penghitungan)

Setiap detektor diuji dengan tiga algoritma penghitungan: SVM (RBF, GridSearchCV), RF (n=200, max_depth=10), dan M01 heuristik. Inferensi dijalankan pada seluruh 953 pohon; SVM dan RF dilatih ulang menggunakan fitur dari masing-masing detektor. Seluruh angka dilaporkan pada test set (n=95).

| Detektor | mAP50 | Penghitung | Macro MAE | Macro Acc ±1 | Acc±1 B1 | Acc±1 B2 | Acc±1 B3 | Acc±1 B4 | Profil Tepat |
|:---|---:|:---|---:|---:|---:|---:|---:|---:|---:|
| [y26n vanilla](reports/e2e_y26n_vanilla_local_svm/metrics.json) | 0.521 | SVM | 1.145 | 70.0% | 90.5% | 68.4% | 56.8% | 64.2% | 0.0% |
| [y26n vanilla](reports/e2e_y26n_vanilla_local_rf/metrics.json) | 0.521 | RF | 1.218 | 68.2% | 90.5% | 68.4% | 54.7% | 58.9% | 0.0% |
| [y26n vanilla](reports/e2e_y26n_vanilla_local_m01/metrics.json) | 0.521 | M01 | 1.337 | 67.1% | 87.4% | 65.3% | 51.6% | 64.2% | 2.1% |
| [y26s vanilla](reports/e2e_y26s_vanilla_local_svm/metrics.json) | 0.506 | SVM | 1.163 | 68.9% | 93.7% | 68.4% | 48.4% | 65.3% | 0.0% |
| [y26s vanilla](reports/e2e_y26s_vanilla_local_rf/metrics.json) | 0.506 | RF | 1.216 | 66.6% | 96.8% | 68.4% | 48.4% | 52.6% | 1.1% |
| [y26s vanilla](reports/e2e_y26s_vanilla_local_m01/metrics.json) | 0.506 | M01 | 1.403 | 65.5% | 89.5% | 66.3% | 38.9% | 67.4% | 2.1% |
| [y26s scratch](reports/e2e_y26s_nopretrained_svm/metrics.json) | 0.511 | SVM | 1.145 | 68.9% | 90.5% | 68.4% | 51.6% | 65.3% | 2.1% |
| [y26s scratch](reports/e2e_y26s_nopretrained_rf/metrics.json) | 0.511 | RF | 1.229 | 67.9% | 93.7% | 65.3% | 55.8% | 56.8% | 1.1% |
| [y26s scratch](reports/e2e_y26s_nopretrained_m01/metrics.json) | 0.511 | M01 | 1.266 | 69.2% | 91.6% | 63.2% | 52.6% | 69.5% | 2.1% |
| [y26s no-aug](reports/e2e_y26s_noaug_svm/metrics.json) | 0.465 | SVM | 1.126 | 70.5% | 91.6% | 69.5% | 56.8% | 64.2% | 1.1% |
| [y26s no-aug](reports/e2e_y26s_noaug_rf/metrics.json) | 0.465 | RF | 1.184 | 68.4% | 92.6% | 66.3% | 55.8% | 58.9% | 1.1% |
| [y26s no-aug](reports/e2e_y26s_noaug_m01/metrics.json) | 0.465 | M01 | 1.384 | 66.6% | 90.5% | 68.4% | 43.2% | 64.2% | 0.0% |
| [**y26m vanilla**](reports/e2e_y26m_vanilla_local_svm/metrics.json) | 0.509 | [**SVM**](reports/e2e_y26m_vanilla_local_svm/metrics.json) | [**1.118**](reports/e2e_y26m_vanilla_local_svm/metrics.json) | [**71.6%**](reports/e2e_y26m_vanilla_local_svm/metrics.json) | 92.6% | 63.2% | 60.0% | 70.5% | 2.1% |
| [y26m vanilla](reports/e2e_y26m_vanilla_local_rf/metrics.json) | 0.509 | RF | 1.211 | 67.9% | 95.8% | 68.4% | 49.5% | 57.9% | 0.0% |
| [y26m vanilla](reports/e2e_y26m_vanilla_local_m01/metrics.json) | 0.509 | M01 | 1.400 | 64.5% | 90.5% | 56.8% | 40.0% | 70.5% | 0.0% |
| [**M01 heuristik (fitur GT — target)**](reports/dedup_brand_new_953/accuracy_953.csv) | — | — | [**0.398**](reports/dedup_brand_new_953/accuracy_953.csv) | [**86.7%**](reports/dedup_brand_new_953/accuracy_953.csv) | — | — | — | — | 26.3% |

**Bottleneck:** Seluruh [15 kombinasi](ml-track/baseline-run/SUMMARY.md) detektor × penghitung menghasilkan Macro Acc±1 dalam rentang sempit **64–72%**, jauh di bawah M01 berbasis GT ([86.7%](reports/dedup_brand_new_953/accuracy_953.csv)). Pilihan algoritma penghitung (SVM, RF, atau M01) tidak mengubah kesimpulan secara signifikan — bottleneck sejati adalah propagasi galat YOLO ke nilai `naive_sum`, `max_per_side`, dan `mean_per_side`. Sebagai pembanding, SVM dengan fitur GT mencapai [96.1%](reports/counting_svm/metrics.json) (Track C) menggunakan arsitektur fitur yang identik.

**Temuan tak terduga:** y26s-noaug (mAP50=[0.465](ml-track/baseline-run/weights/y26s_noaug_results.csv), detektor terlemah) menghasilkan [SVM 70.5%](reports/e2e_y26s_noaug_svm/metrics.json), hanya 1.1 pp di bawah y26m (mAP50=[0.509](ml-track/baseline-run/weights/y26m_vanilla_local_results.csv), [SVM 71.6%](reports/e2e_y26m_vanilla_local_svm/metrics.json)). Hal ini mengindikasikan bahwa distribusi galat detektor — bukan besarnya mAP — yang menentukan kualitas fitur 13-dim untuk penghitungan.

```bash
# Jalankan pipeline E2E untuk satu detektor (inferensi + SVM + RF + M01):
python scripts/run_e2e_pipeline.py \
    --name y26m_vanilla_local \
    --weights ml-track/baseline-run/weights/y26m_vanilla_local.pt
```

---

## Kesimpulan

| Kasus Penggunaan | Rekomendasi |
|:---|:---|
| Penghitungan produksi | [**M01_selector_b2b3**](algorithms/M01_selector_b2b3.py) — Macro Acc±1 = [86.67%](reports/dedup_brand_new_953/accuracy_953.csv), valid per [RULES.txt](archive/_to_review/exp_12%20may%202026/RULES.txt) |
| Deteksi saja (akurasi) | [**YOLO26m**](ml-track/baseline-run/weights/y26m_vanilla_local_args.yaml) — mAP50 = [0.509](ml-track/baseline-run/weights/y26m_vanilla_local_results.csv) |
| Deteksi saja (kecepatan) | [**YOLO26n**](ml-track/baseline-run/weights/y26n_vanilla_local_args.yaml) — mAP50 = [0.521](ml-track/baseline-run/weights/y26n_vanilla_local_results.csv), 0.2 ms/gambar |
| Baseline riset ML | [**SVM pada fitur GT**](reports/counting_svm/metrics.json) — Macro Acc±1 = [96.1%](reports/counting_svm/metrics.json), Macro MAE = [0.318](reports/counting_svm/metrics.json) |
| Pipeline E2E terbaik | [**y26m → SVM**](reports/e2e_y26m_vanilla_local_svm/metrics.json) — Macro Acc±1 = [71.6%](reports/e2e_y26m_vanilla_local_svm/metrics.json), masih 15 pp di bawah heuristik |

---

## Validasi Ground Truth

GT JSON di `Brand-New-Dataset-YOLO/json/` harus memenuhi dua invariant struktural:

1. **Same-side uniqueness** — satu bunch tidak boleh muncul ≥ 2× di `side_index` yang sama (kamera satu sisi maksimal lihat bunch sekali). Detector: [`scripts/audit_same_side_dup.py`](scripts/audit_same_side_dup.py).
2. **Geometric adjacency (visibility cone)** — bunch hanya bisa terlihat dari sisi yg adjacent dgn home (rule update 2026-05-16 setelah validasi visual RA):
   - **4-sisi:** max distance = 1 (≤ 3 sisi visible). Mustahil di sisi opposite (distance 2).
     Contoh: home=`sisi_1` → visible {`sisi_4`, `sisi_1`, `sisi_2`}; mustahil `sisi_3`.
   - **8-sisi:** max distance = 3 (≤ 6 sisi visible — bunch besar/prominent). Mustahil ≥ 7 sisi.
     Normal: 5 sisi (distance ≤ 2). Edge case bunch besar: 6 sisi (distance ≤ 3).

   Detector: [`scripts/audit_impossible_visibility.py`](scripts/audit_impossible_visibility.py).

**Status audit (2026-05-16):**

| Audit | Trees affected | Bunches affected | Status |
|:---|---:|---:|:---|
| Same-side dup | 8 | 18 | ✅ FIXED (8 wrap-around trees per laporan RA) |
| Geometric violation (4-sisi 4/4) | 31 | 42 | ✅ AUTO-HEALED via [`scripts/heal_4side_visibility.py`](scripts/heal_4side_visibility.py) |
| Geometric violation (8-sisi) | 9 | 14 | ✅ CLEARED (4 manual fix + rule relaxation 2026-05-16) |
| Geometric warning | 469 | 802 | ℹ️ borderline (full visibility reach, accepted) |

**Status final:** 0 GT violations across all checks. Net +~62 unique bunches across ~48 trees. Backups: `archive/json_pre_wrap_fix_2026-05-15/`, `archive/json_pre_visibility_fix_2026-05-16/`, `archive/json_pre_visibility_heal_4side_2026-05-16/`.

Wrap-around fix detail (8 trees): backup di `archive/json_pre_wrap_fix_2026-05-15/`, runner di [`scripts/fix_wrap_around_links.py`](scripts/fix_wrap_around_links.py).

```bash
python scripts/audit_same_side_dup.py
python scripts/audit_impossible_visibility.py
```

---

## Reproduksi di Device Baru

Repo ini tracked **10.004 file** termasuk labels, JSON GT, split files, weights `.pt`, dan inference predictions. Yang **tidak** tracked: `Brand-New-Dataset-YOLO/images/` (~2.3 GB, distribusi via HuggingFace).

**Track heuristik (M01..M29) — zero download:**
```bash
git clone <repo>
pip install -r requirements.txt
python scripts/dedup_brand_new_953.py    # M01 86.67% Macro Acc±1
```
Cukup `labels/` + `json/` (sudah ter-track). Tidak butuh images.

**Track E2E ML (inferensi YOLO + counting RF/SVM) — butuh images:**
```bash
git clone <repo>
pip install -r requirements.txt
pip install huggingface_hub ultralytics scikit-learn

python scripts/setup_dataset.py          # idempotent: skip kalau images sudah ada
python scripts/run_e2e_pipeline.py --name y26m_vanilla_local \
    --weights ml-track/baseline-run/weights/y26m_vanilla_local.pt
```

---

## Panduan Cepat

```bash
# 1. Instalasi dependensi
pip install -r requirements.txt
pip install scikit-learn ultralytics huggingface_hub

# 2. Unduh dataset images dari HuggingFace (idempotent)
python scripts/setup_dataset.py

# 3. Jalankan semua track
python scripts/dedup_brand_new_953.py    # Track A: heuristik
python scripts/run_counting_svm.py       # Track C: SVM dari fitur GT
python scripts/run_counting_rf.py        # Track C: RF dari fitur GT

# Track B: training detektor
python -c "from ultralytics import YOLO; YOLO('yolo26n.pt').train(
    data='ml-track/local_data.yaml', epochs=100, batch=16, imgsz=640, seed=42,
    project='/workspace/runs/detect', name='y26n_vanilla_local')"

# Track D: pipeline E2E (inferensi + SVM + RF + M01 sekaligus)
python scripts/run_e2e_pipeline.py \
    --name y26n_vanilla_local \
    --weights ml-track/baseline-run/weights/y26n_vanilla_local.pt
```

---

## Tautan

- [`RESEARCH.md`](RESEARCH.md) — Dokumen riset lengkap
- [`archive/_to_review/exp_12 may 2026/REPORT.md`](archive/_to_review/exp_12%20may%202026/REPORT.md) — Analisis mendalam M60 (diarsipkan 2026-05-14)
- [`ml-track/baseline-run/SUMMARY.md`](ml-track/baseline-run/SUMMARY.md) — Ringkasan hasil ML (matriks E2E lengkap 15 kombinasi)
- [`ml-track/CLAUDE-TRAINING.md`](ml-track/CLAUDE-TRAINING.md) — Panduan eksperimen ML di RunPod/Vast.ai
- [`archive/_to_review/exp_13 May 2026/PROGRESS.md`](archive/_to_review/exp_13%20May%202026/PROGRESS.md) — Log progres training (diarsipkan 2026-05-14)
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
