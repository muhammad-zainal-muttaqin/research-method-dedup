# Multi-View Oil Palm Bunch Counting

Pipeline untuk menghitung jumlah tandan unik per pohon dari foto multi-sisi (4–8 sisi).
Dataset: **953 pohon** | Kelas: **B1→B2→B3→B4** (ordinal) | License: **CC BY-NC 4.0**

---

## Best Results Summary

### Per-Track Champion

| Track | Metode | Macro MAE | Macro Acc ±1 | Status |
|:---|:---|---:|---:|:---|
| A. Heuristik | M01_selector_b2b3 | 0.398 | 86.67% | ✅ Produksi (valid) |
| B. Deteksi | YOLO26m | — | mAP50 = **0.528** | ✅ Detektor terbaik |
| C. ML Counting (fitur GT) | SVM RBF | **0.318** | **96.1%** | ✅ ML terbaik |
| D. End-to-End | y26s → SVM | 1.163 | 68.9% | ⚠️ Batas bawah |

### Metrik Terbaik Keseluruhan

| Metrik | Nilai | Metode |
|:---|---:|:---|
| Macro Acc ±1 tertinggi — ML (fitur GT) | **96.1%** | SVM GT features |
| Macro Acc ±1 tertinggi — heuristik valid | **86.67%** | M01_selector_b2b3 |
| Macro MAE terendah — ML (fitur GT) | **0.318** | SVM GT features |
| Macro MAE terendah — heuristik valid | **0.398** | M01_selector_b2b3 |
| mAP50 terbaik | **0.528** | YOLO26m |
| Tercepat | **0.005 ms/pohon** | M15_divide_global |

### Temuan Utama

Pipeline ML dengan fitur GT (Track C) mengungguli heuristik terbaik M01: SVM mencapai Macro Acc±1 = **96.1%** dibandingkan **86.67%** milik M01, sehingga membuktikan bahwa desain fitur 13-dim sudah memadai apabila detektor menghasilkan deteksi yang benar.

Namun, pipeline ujung-ke-ujung (Track D) hanya mencapai Macro Acc±1 = **68.9%** karena propagasi galat detektor YOLO — setiap deteksi palsu (FP) dan deteksi yang terlewat (FN) merusak nilai `naive_sum`, `max_per_side`, dan `mean_per_side` sebelum masuk ke SVM/RF, sehingga model menerima input yang sudah tidak akurat.

> M60 dan M53 mencapai 90.24%, tetapi keduanya **tidak valid** per `exp_12 may 2026/RULES.txt` karena menggunakan tabel divisor yang diturunkan dari training split — bukan dari prinsip geometri murni — sehingga tidak dapat digeneralisasi ke kebun lain.

---

## Track A: Heuristic Counting (No Training)

| Peringkat | Metode | Macro Acc ±1 | Macro MAE | Profil Tepat | Valid? |
|:---:|:---|---:|---:|---:|:---:|
| — | M60_blind_strict | 90.24% | 0.302 | — | ❌ |
| — | M53_three_band_override | 90.24% | 0.304 | — | ❌ |
| 1 | **M01_selector_b2b3** | **86.67%** | **0.398** | 26.3% | ✅ |
| 2 | M05_blend_vis_divide | 86.04% | 0.408 | 25.3% | ✅ |
| 3 | M06_weight_visibility | 85.94% | 0.396 | 25.3% | ✅ |
| 4 | M15_divide_global | 84.37% | 0.416 | 23.3% | ✅ |

Tabel lengkap 29 metode: `reports/dedup_brand_new_953/accuracy_953.csv`

> ❌ M60 dan M53 tidak valid per `exp_12 may 2026/RULES.txt`: keduanya menggunakan tabel divisor yang diturunkan dari statistik training split (kalibrasi domain-spesifik), bukan dari prinsip geometri murni. Disimpan hanya sebagai referensi historis.

```bash
python scripts/dedup_brand_new_953.py
```

---

## Track B: Detection (YOLO26)

| Model | mAP50 | mAP50-95 | Speed | Params |
|---:|---:|---:|---:|---:|
| YOLO26n | 0.511 | 0.237 | 0.2ms | 2.4M |
| YOLO26s | 0.501 | 0.235 | 0.5ms | 9.5M |
| **YOLO26m** | **0.528** | **0.240** | 0.8ms | 20.4M |

### Ablasi

| Experiment | Best Epoch | mAP50 | mAP50-95 | Catatan |
|---:|---:|---:|---:|---|
| y26s vanilla retrain (lokal) | 21 | 0.506 | 0.234 | ≈ RunPod 0.501 — dipakai E2E pipeline |
| y26s no-pretrained | 57 | **0.511** | 0.231 | Scratch = pretrained! |
| y26s no-augmentation | 6 | 0.465 | 0.216 | Overfit, early stop ep=56 |

**Insight:** COCO pretrained tidak wajib — y26s dari scratch menghasilkan mAP50 sama dengan pretrained (0.511). Augmentasi esensial: tanpa augmentasi mAP50 drop ke 0.465 dan overfit di epoch 6.

```bash
yolo detect train model=yolo26m.pt data=local_data.yaml epochs=100 batch=16 imgsz=640
```

---

## Track C: ML Counting (Fitur dari GT)

Fitur 13 dimensi per pohon: `naive_sum` (B1–B4), `max_per_side` (B1–B4), `mean_per_side` (B1–B4), `n_sides`.

| Model | Macro MAE | Macro Acc ±1 | Acc±1 B1 | Acc±1 B2 | Acc±1 B3 | Acc±1 B4 | Profil Tepat |
|:---|---:|---:|---:|---:|---:|---:|---:|
| **SVM (RBF, GridSearchCV)** | **0.318** | **96.1%** | **100.0%** | 95.8% | 91.6% | 96.8% | 27.4% |
| RF (n=200, max_depth=10) | 0.353 | 95.3% | 96.8% | 96.8% | 90.5% | 96.8% | 27.4% |

Detail: `reports/counting_{svm,rf}/metrics.json`

```bash
python scripts/run_counting_svm.py
python scripts/run_counting_rf.py
```

---

## Track D: End-to-End (Deteksi → Penghitungan)

Setiap detektor diuji dengan tiga algoritma penghitungan: SVM (RBF, GridSearchCV), RF (n=200, max_depth=10), dan M01 heuristik. Inferensi dilakukan pada seluruh 953 pohon; SVM/RF dilatih ulang dari fitur detektor masing-masing. Semua angka dilaporkan pada test set (n=95).

| Detektor | mAP50 | Counting | Macro MAE | Macro Acc ±1 | Acc±1 B1 | Acc±1 B2 | Acc±1 B3 | Acc±1 B4 | Profil Tepat |
|:---|---:|:---|---:|---:|---:|---:|---:|---:|---:|
| y26n vanilla | 0.511 | **SVM** | 1.126 | **71.1%** | 93.7% | 71.6% | 53.7% | 65.3% | 1.1% |
| y26n vanilla | 0.511 | RF | 1.216 | 64.7% | 93.7% | 64.2% | 45.3% | 55.8% | 1.1% |
| y26n vanilla | 0.511 | M01 | 1.334 | 66.3% | 91.6% | 58.9% | 49.5% | 65.3% | 1.1% |
| y26s vanilla | 0.506 | SVM | 1.163 | 68.9% | 93.7% | 68.4% | 48.4% | 65.3% | 0.0% |
| y26s vanilla | 0.506 | RF | 1.216 | 66.6% | 96.8% | 68.4% | 48.4% | 52.6% | 1.1% |
| y26s vanilla | 0.506 | M01 | 1.403 | 65.5% | 89.5% | 66.3% | 38.9% | 67.4% | 2.1% |
| y26s scratch | 0.511 | SVM | 1.145 | 68.9% | 90.5% | 68.4% | 51.6% | 65.3% | 2.1% |
| y26s scratch | 0.511 | RF | 1.229 | 67.9% | 93.7% | 65.3% | 55.8% | 56.8% | 1.1% |
| y26s scratch | 0.511 | **M01** | 1.266 | **69.2%** | 91.6% | 63.2% | 52.6% | 69.5% | 2.1% |
| y26s no-aug | 0.465 | SVM | 1.126 | 70.5% | 91.6% | 69.5% | 56.8% | 64.2% | 1.1% |
| y26s no-aug | 0.465 | **RF** | 1.184 | 68.4% | 92.6% | 66.3% | 55.8% | 58.9% | 1.1% |
| y26s no-aug | 0.465 | M01 | 1.384 | 66.6% | 90.5% | 68.4% | 43.2% | 64.2% | 0.0% |
| y26m vanilla | 0.528 | SVM | ⏳ | ⏳ | — | — | — | — | — |
| y26m vanilla | 0.528 | RF | ⏳ | ⏳ | — | — | — | — | — |
| y26m vanilla | 0.528 | M01 | ⏳ | ⏳ | — | — | — | — | — |
| **M01 (fitur GT — target)** | — | — | **0.398** | **86.7%** | — | — | — | — | 26.3% |

**Bottleneck:** Semua 12 kombinasi detektor × counting menghasilkan Macro Acc±1 dalam rentang sempit **64–71%**, jauh di bawah M01 berbasis GT (86.7%). Algoritma penghitungan yang lebih baik (SVM vs RF vs M01) tidak mengubah kesimpulan secara signifikan — bottleneck sejati adalah propagasi galat YOLO ke fitur `naive_sum`, `max_per_side`, dan `mean_per_side`. Temuan pendukung: SVM dengan fitur GT mencapai 96.1% (Track C) menggunakan arsitektur fitur identik.

**Temuan tak terduga:** y26s-noaug (mAP50=0.465, detektor terlemah) menghasilkan SVM 70.5%, hampir setara dengan y26n (mAP50=0.511, SVM 71.1%). Ini mengindikasikan bahwa distribusi error detektor — bukan besarnya mAP — yang menentukan kualitas fitur 13-dim untuk counting.

```bash
# Jalankan E2E untuk satu detektor (inference + SVM + RF + M01):
python scripts/run_e2e_pipeline.py --name y26n_vanilla_local \
    --weights baseline-run/weights/y26n_vanilla_local.pt
```

---

## Kesimpulan

| Kasus Penggunaan | Rekomendasi |
|:---|:---|
| Penghitungan produksi | **M01_selector_b2b3** — Macro Acc±1 = 86.67%, valid per RULES.txt |
| Deteksi saja | **YOLO26m** — mAP50 = 0.528 |
| Baseline riset ML | **SVM pada fitur GT** — Macro Acc±1 = 96.1%, Macro MAE = 0.318 |
| End-to-end ML | ⚠️ Diperlukan representasi fitur yang robust terhadap FP/FN detektor |

---

## Quickstart

```bash
# 1. Install
pip install -r requirements.txt
pip install scikit-learn ultralytics huggingface_hub

# 2. Dataset (HuggingFace)
python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('ULM-DS-Lab/OilPalm-MultiView-BunchCount-YOLO', \
  repo_type='dataset', local_dir='./Tested-Brand-New-Dataset-YOLO')"

# 3. Run all tracks
python scripts/dedup_brand_new_953.py       # Track A
python scripts/run_counting_svm.py          # Track C
yolo detect train model=yolo26m.pt \
  data=local_data.yaml epochs=100           # Track B
python scripts/run_e2e_inference.py \
  --weights baseline-run/weights/y26s_vanilla_local.pt
python scripts/run_e2e_svm.py               # Track D
```

---

## Links

- [`RESEARCH.md`](RESEARCH.md) — Full research document
- [`exp_12 may 2026/REPORT.md`](exp_12%20may%202026/REPORT.md) — M60 deep dive
- [`baseline-run/SUMMARY.md`](baseline-run/SUMMARY.md) — ML results detail
- [`CLAUDE-TRAINING.md`](CLAUDE-TRAINING.md) — ML experiment orchestration
- [HuggingFace Dataset](https://huggingface.co/datasets/ULM-DS-Lab/OilPalm-MultiView-BunchCount-YOLO)

---

## Citation

```bibtex
@dataset{palm_bunch_2026,
  title={Multi-View Oil Palm Bunch Dataset},
  author={Muttaqin, M. Zainal},
  year={2026},
  publisher={HuggingFace},
  url={https://huggingface.co/datasets/ULM-DS-Lab/OilPalm-MultiView-BunchCount-YOLO}
}
```

License: **CC BY-NC 4.0**
