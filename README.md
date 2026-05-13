# Multi-View Oil Palm Bunch Counting

Pipeline untuk menghitung jumlah tandan unik per pohon dari foto multi-sisi (4–8 sisi). Dua track: **(A) Heuristic dedup** — 100% aturan deterministik, tanpa training. **(B) ML Baseline** — YOLO26 detection + SVM/RF counting, sebagai *reference lower bound*.

**Verdict:** Heuristic `M60_blind_strict` menang telak (90.24% Acc±1). ML pipeline E2E (1.16 Macro MAE) ~3× lebih buruk karena feature 13-dim kehilangan informasi side-level.

---

## Dataset

| Sumber | Jumlah pohon | Status |
|---:|---:|---|
| `Tested-Brand-New-Dataset-YOLO/json/` | **953** | Kanonik, GT lengkap |
| `json_05 mei 2026/` | 882 | Legacy snapshot 5 Mei |
| `json/` | 228 | Legacy, dataset pengembangan v9 |

- **2 varietas:** DAMIMAS (854 pohon) + LONSUM (99 pohon)
- **4–8 sisi per pohon** (45 pohon 8-sisi), resolusi 960×1280
- **4 kelas ordinal:** B1 (matang merah) → B2 (transisi) → B3 (hitam) → B4 (kecil duri)
- **Ambiguitas irreducible:** B2↔B3 visual overlap — bukan derau label
- **Link HuggingFace:** `ULM-DS-Lab/OilPalm-MultiView-BunchCount-YOLO`
- **Lisensi:** CC BY-NC 4.0

---

## Repository Structure

```
├── Tested-Brand-New-Dataset-YOLO/   Dataset (images, labels, JSON GT)
├── algorithms/                       Heuristic methods (M01-M60)
├── scripts/                          Audit, counting, benchmark scripts
├── reports/                          All output CSVs
│   ├── dedup_brand_new_953/         Heuristic benchmark on 953 trees
│   ├── counting_{svm,rf}/           ML counting from GT features
│   └── e2e_{svm,rf}/                End-to-end detection→counting
├── baseline-run/                     ML experiment logs, weights, SUMMARY.md
│   └── weights/                     best.pt for each YOLO experiment
├── exp_12 may 2026/                  Latest dedup research (M60)
├── exp_13 May 2026/                  ML baseline experiment tracker
├── tools_sawit/                      Web annotator (vanilla JS)
├── dataset/                          Legacy YOLO config
├── json_05 mei 2026/                 Legacy GT snapshots
│   RESEARCH.md                       Main research document
│   CLAUDE-TRAINING.md                ML experiment orchestration doc
│   report_10Mei2026.md               Dedup final report (M01 era)
```

---

## A. Heuristic Counting (Track A)

### Latest Best: M60_blind_strict — 90.24% Acc±1

| Method | Full 953 Acc±1 | Macro MAE | n_fail | Test alone Acc±1 |
|---|---:|---:|---:|---:|
| M01_selector_b2b3 (prior champion) | 86.78% | 0.388 | 126 | 89.76% |
| **M60_blind_strict (new best)** | **90.24%** | **0.302** | **93** | **91.57%** |

Source: [`exp_12 may 2026/REPORT.md`](exp_12%20may%202026/REPORT.md)

### How M60 works

1. **M31 side-aware divisor** — Per `(n_sides, class)` median ratio computed from train split. Fixes M01's catastrophic failure on 8-side trees (M01 divisor clamp at 1.45 vs empirical 3-4).
2. **Regime overrides** — 11 override cuts selected via strict bilateral filter (gain on **train AND val** both). Test never inspected during selection.
3. **Strict blind protocol** — Test set evaluated once at the end.

### Top-7 on 953 trees

| Rank | Method | Acc±1 | Macro MAE |
|---:|---|---:|---:|
| 1 | **M60_blind_strict** | **90.24%** | **0.302** |
| 2 | M53_three_band_override | 90.24% | 0.304 |
| 3 | M01_selector_b2b3 | 86.67% | 0.398 |
| 4 | M02_selector_trifurc | 86.67% | 0.399 |
| 5 | M03_blend_geometric | 86.15% | 0.396 |
| 6 | M05_blend_vis_divide | 86.04% | 0.408 |
| 7 | M06_weight_visibility | 85.94% | 0.396 |

Full table of 29 methods: [`reports/dedup_brand_new_953/accuracy_953.csv`](reports/dedup_brand_new_953/accuracy_953.csv)

### Cross-dataset regression

| Method | 228 | 478 | 727 | 882 | **953** | Delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| M06_weight_visibility | 92.54% | 90.38% | 89.41% | 89.34% | 85.94% | −6.60 pp |
| M15_divide_global | 90.79% | 89.12% | 87.90% | 88.21% | 84.37% | **−6.42 pp** |
| M12_selector_overrides | **97.37%** | 92.68% | 89.27% | 88.78% | 84.68% | −12.69 pp |

### Run Heuristic Benchmark

```bash
pip install -r requirements.txt
python scripts/dedup_brand_new_953.py       # 953-tree canonical
python scripts/benchmark_multidim.py        # multi-snapshot regression
```

---

## B. Detection Baseline (Track B)

YOLO26 experiments on the oil palm dataset. All trained on RTX A5000 (24GB) or RunPod.

### Vanilla (pretrained, augmentation ON)

| Model | Best Epoch | mAP50 | mAP50-95 | Speed | Params | Source |
|---:|---:|---:|---:|---:|---:|---|
| YOLO26n | 30 | 0.511 | 0.237 | 0.2ms | 2.4M | RunPod |
| YOLO26s | 32 | 0.501 | 0.235 | 0.5ms | 9.5M | RunPod |
| **YOLO26m** | **20** | **0.528** | **0.240** | 0.8ms | 20.4M | RunPod |
| YOLO26s (retrain lokal) | 21 | 0.506 | 0.234 | 0.5ms | 9.5M | RTX A5000 |

### Ablations

| Experiment | Best Epoch | mAP50 | mAP50-95 | Note |
|---:|---:|---:|---:|---|
| y26s no-pretrained (scratch) | 57 | **0.511** | 0.231 | Scratch = pretrained! |
| y26s no-augmentation | 6 | 0.465 | 0.216 | Overfit cepat, early stop ep=56 |

### Per-class Detection (val mAP50)

| Model | B1 | B2 | B3 | B4 | B1 Recall | B2 Recall | B3 Recall | B4 Recall |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| y26n | **0.728** | 0.410 | 0.576 | 0.331 | **0.784** | 0.443 | 0.621 | 0.364 |
| y26s | 0.719 | 0.393 | 0.585 | 0.308 | 0.769 | **0.488** | **0.698** | 0.242 |
| y26m | 0.757 | 0.411 | **0.595** | **0.348** | 0.688 | 0.483 | 0.672 | **0.421** |

B4 is the weakest class (sample-starved, recall 24-42%).

### Run Detection

```bash
# Vanilla YOLO26s
yolo detect train model=yolo26s.pt data=local_data.yaml epochs=100 batch=16 imgsz=640

# Scratch (no pretrained)
yolo detect train model=yolo26s.yaml pretrained=False data=local_data.yaml epochs=100 batch=16 imgsz=640

# No augmentation
yolo detect train model=yolo26s.pt data=local_data.yaml epochs=100 batch=16 imgsz=640 \
  hsv_h=0 hsv_s=0 hsv_v=0 degrees=0 translate=0 scale=0 shear=0 \
  perspective=0 flipud=0 fliplr=0 mosaic=0 mixup=0 erasing=0 auto_augment=None
```

---

## C. ML Counting (Track C)

13-dim features per tree: naive_sum(B1-B4), max_per_side(B1-B4), mean_per_side(B1-B4), n_sides. Trained on GT JSON features, evaluated on test set (95 trees).

| Model | Macro MAE | Acc±1 B1 | Acc±1 B2 | Acc±1 B3 | Acc±1 B4 | Exact-profile | Total MAE | Total±1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SVM (GridSearchCV, RBF kernel) | **0.318** | 100% | 95.8% | 91.6% | 96.8% | 27.4% | 1.126 | 72.6% |
| RF (n=200, max_depth=10) | 0.353 | 96.8% | 96.8% | 90.5% | 96.8% | 27.4% | 1.200 | 70.5% |

### Run ML Counting

```bash
python scripts/build_counting_features.py    # Build 13-dim features
python scripts/run_counting_svm.py             # SVM training + eval
python scripts/run_counting_rf.py              # RF training + eval
```

---

## D. End-to-End Pipeline (Track D)

Pipeline: YOLO26s → infer on all 4-8 sides → extract 13-dim features → SVM/RF → 4-class count.

| Metrik | M01 heuristic | y26s→SVM | y26s→RF |
|---|---:|---:|---:|
| Macro class-MAE | **0.398** | 1.163 | 1.216 |
| Total-count MAE | **1.414** | 2.337 | 2.337 |
| Total ±1 | **86.7%** | 38.9% | 40.0% |
| Exact-profile | — | 0.0% | 1.1% |

**Key insight:** Detector quality is NOT the bottleneck (mAP50=0.506 vs 0.445 gave nearly identical E2E results). The 13-dim feature aggregation loses side-level visibility/overlap information.

### Run E2E

```bash
python scripts/run_e2e_inference.py --weights baseline-run/weights/y26s_vanilla_local.pt
python scripts/run_e2e_svm.py
python scripts/run_e2e_rf.py
```

---

## Overall Comparison

| Method | Macro MAE | Total MAE | Total ±1 | Exact Profile | Track |
|---|---:|---:|---:|---:|---|
| Naive sum (baseline) | ~2.28 | ~9.12 | ~2.8% | ~1.9% | — |
| M60_blind_strict | **0.302** | — | **90.24%** | — | A. Heuristic |
| M01_selector_b2b3 | 0.398 | **1.414** | 86.7% | 26.3% | A. Heuristic |
| SVM (GT features) | 0.318 | 1.126 | 72.6% | 27.4% | C. ML Counting |
| RF (GT features) | 0.353 | 1.200 | 70.5% | 27.4% | C. ML Counting |
| y26s→SVM (E2E) | 1.163 | 2.337 | 38.9% | 0.0% | D. End-to-End |
| y26s→RF (E2E) | 1.216 | 2.337 | 40.0% | 1.1% | D. End-to-End |

**Verdict:** Heuristic methods (M60/M01) are the production choice. ML pipeline needs side-aware feature engineering (not naive 13-dim aggregation) to be competitive.

---

## Quickstart

```bash
# 1. Install
pip install -r requirements.txt
pip install scikit-learn ultralytics huggingface_hub

# 2. Dataset (from HuggingFace)
export HUGGING_FACE_HUB_TOKEN=hf_xxx
python -c "from huggingface_hub import snapshot_download; snapshot_download('ULM-DS-Lab/OilPalm-MultiView-BunchCount-YOLO', repo_type='dataset', local_dir='./Tested-Brand-New-Dataset-YOLO')"

# 3. Heuristic benchmark (CPU, ~1 min)
python scripts/dedup_brand_new_953.py

# 4. ML counting (CPU, ~2 min)
python scripts/run_counting_svm.py
python scripts/run_counting_rf.py

# 5. Detection (GPU)
yolo detect train model=yolo26s.pt data=local_data.yaml epochs=100 batch=16 imgsz=640

# 6. End-to-end (GPU + CPU)
python scripts/run_e2e_inference.py --weights /workspace/runs/detect/y26s_vanilla_local/weights/best.pt
python scripts/run_e2e_svm.py
python scripts/run_e2e_rf.py

# 7. Generate report
python scripts/generate_training_summary.py
```

---

## Reproduce Checklist

- [ ] Clone repo + install dependencies
- [ ] Download dataset from HuggingFace
- [ ] Run heuristic benchmark (`scripts/dedup_brand_new_953.py`)
- [ ] Run ML counting (`scripts/run_counting_svm.py` + `scripts/run_counting_rf.py`)
- [ ] Run YOLO detection (vanilla + ablations)
- [ ] Run E2E pipeline
- [ ] Generate summary

Full logs and weights: `baseline-run/`. Experiment tracker: `exp_13 May 2026/PROGRESS.md`.

---

## Research Constraints

| Allowed | Forbidden |
|---|---|
| Rule-based heuristics | Neural network training (Siamese, CNN embedding, MLP) |
| Deterministic routing | Learned thresholds via backprop |
| Statistical correction (median ratios) | Strict matching (Hungarian, graph, cluster) on noisy TXT |
| YOLO detection (supervised) | Cross-view embedding / learned matcher |
| Classical ML (SVM, RF) | — |

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

---

## License

CC BY-NC 4.0

---

## Related Documents

- [`RESEARCH.md`](RESEARCH.md) — Full research document (start at Section 0)
- [`CLAUDE-TRAINING.md`](CLAUDE-TRAINING.md) — ML experiment orchestration
- [`exp_12 may 2026/REPORT.md`](exp_12%20may%202026/REPORT.md) — Latest dedup research (M60)
- [`exp_13 May 2026/PROGRESS.md`](exp_13%20May%202026/PROGRESS.md) — ML experiment tracker
- [`baseline-run/SUMMARY.md`](baseline-run/SUMMARY.md) — ML results summary
- [`report_10Mei2026.md`](report_10Mei2026.md) — Dedup final report (M01 era)
- [`CLAUDE.md`](CLAUDE.md) / [`AGENTS.md`](AGENTS.md) — AI assistant guides
