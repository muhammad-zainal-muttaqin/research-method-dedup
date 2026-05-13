# Multi-View Oil Palm Bunch Counting

Pipeline untuk menghitung jumlah tandan unik per pohon dari foto multi-sisi (4–8 sisi).
Dataset: **953 pohon** | Kelas: **B1→B2→B3→B4** (ordinal) | License: **CC BY-NC 4.0**

---

## Best Results Summary

### Per-Track Champion

| Track | Method | Macro MAE | Acc ±1 | Status |
|---|---:|---:|---:|:---|
| A. Heuristic Counting | M60_blind_strict | 0.302 | **90.24%** | ✅ Production |
| B. Detection | YOLO26m | — | mAP50=**0.528** | ✅ Best detector |
| C. ML Counting (GT features) | SVM (RBF) | **0.318** | — | ✅ Best ML counter |
| D. End-to-End | y26s→SVM | 1.163 | 38.9% | ⚠️ Lower bound |

### Absolute Best per Metric

| Metric | Value | Method |
|---|---:|---|
| Highest Acc ±1 | **90.24%** | M60_blind_strict |
| Lowest Macro MAE | **0.302** | M60_blind_strict |
| Best mAP50 | **0.528** | YOLO26m |
| Best ML Counting | **0.318 MAE** | SVM (GT features) |
| Fastest | **0.005 ms/tree** | M15_divide_global |

### Key Insight

Heuristic M60 (**90.24%**) >>> ML End-to-End (**38.9%**).
Bottleneck E2E bukan desain fitur 13-dim (terbukti: GT features → SVM = MAE 0.318 dengan fitur identik), melainkan **error propagation dari YOLO detector** — FP/FN merusak nilai naive_sum/max_per_side sebelum masuk SVM.

---

## Track A: Heuristic Counting (No Training)

| Rank | Method | Acc ±1 | Macro MAE | Exact Profile |
|---|---:|---|---:|---:|
| 1 | **M60_blind_strict** | **90.24%** | **0.302** | — |
| 2 | M53_three_band_override | 90.24% | 0.304 | — |
| 3 | M01_selector_b2b3 | 86.67% | 0.398 | 26.3% |
| 4 | M05_blend_vis_divide | 86.04% | 0.408 | 25.3% |
| 5 | M06_weight_visibility | 85.94% | 0.396 | 25.3% |
| 6 | M15_divide_global | 84.37% | 0.416 | 23.3% |

Full table (29 methods): `reports/dedup_brand_new_953/accuracy_953.csv`

**Novelty M60:** Side-aware median ratio divisor per `(n_sides, class)` + strict bilateral filter overrides (gain on train AND val). Evaluasi blind — test set dilihat sekali di akhir. Detail: [`exp_12 may 2026/REPORT.md`](exp_12%20may%202026/REPORT.md).

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

## Track C: ML Counting (from GT Features)

Feature 13-dim per tree: naive_sum(B1-B4), max_per_side(B1-B4), mean_per_side(B1-B4), n_sides.

| Model | Macro MAE | Exact Profile | Total ±1 | Acc±1 B1 | Acc±1 B2 | Acc±1 B3 | Acc±1 B4 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| **SVM (RBF, GridSearchCV)** | **0.318** | 27.4% | 72.6% | **100%** | 95.8% | 91.6% | 96.8% |
| RF (n=200, max_depth=10) | 0.353 | 27.4% | 70.5% | 96.8% | 96.8% | 90.5% | 96.8% |

Detail: `reports/counting_{svm,rf}/metrics.json`

```bash
python scripts/run_counting_svm.py
python scripts/run_counting_rf.py
```

---

## Track D: End-to-End (Detection → Counting)

Pipeline: YOLO26s → infer 4–8 sisi → 13-dim features → SVM/RF → 4-class count.

| Pipeline | Macro MAE | Total MAE | Total ±1 | Exact Profile |
|---:|---:|---:|---:|---:|
| y26s→SVM | 1.163 | 2.337 | 38.9% | 0.0% |
| y26s→RF | 1.216 | 2.337 | 40.0% | 1.1% |
| **M01 heuristic (target)** | **0.398** | **1.414** | **86.7%** | — |

**Bottleneck:** Error propagation dari YOLO detector (FP/FN merusak 13-dim features sebelum masuk SVM/RF). Fitur 13-dim sendiri **cukup** bila input sempurna — Track C (GT → SVM) mencapai MAE 0.318 dengan arsitektur fitur identik. Peningkatan mAP marginal (0.445→0.506) tidak cukup memperbaiki E2E — butuh representasi robust terhadap FP/FN.

```bash
python scripts/run_e2e_inference.py --weights baseline-run/weights/y26s_vanilla_local.pt
python scripts/run_e2e_svm.py
python scripts/run_e2e_rf.py
```

---

## Overall Verdict

| Use Case | Recommendation |
|---|---|
| Production counting | **M60_blind_strict** (90.24%, no training) |
| Detection only | **YOLO26m** (mAP50=0.528) |
| ML research baseline | **SVM on GT features** (0.318 MAE) |
| End-to-end ML | ⚠️ Butuh representasi robust terhadap FP/FN dari detector |

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
