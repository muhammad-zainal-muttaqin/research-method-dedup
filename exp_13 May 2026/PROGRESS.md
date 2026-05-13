# Eksperimen 13 May 2026 — ML Training Pipeline

Session ini mengerjakan eksperimen #4–#9 dari CLAUDE-TRAINING.md.
GPU: NVIDIA RTX A5000 24GB | Workspace: /workspace/runs/detect/

---

## Status

| # | Eksperimen | Status | Best Result | Detail |
|---:|---|---|---:|---|
| 1 | y26n vanilla (RunPod) | ✅ DONE | mAP50=0.511 (ep=30) | mAP50-95=0.237, 0.2ms, 2.4M params |
| 2 | y26s vanilla (RunPod) | ✅ DONE | mAP50=0.501 (ep=32) | mAP50-95=0.235, 0.5ms, 9.5M params |
| 3 | **y26m vanilla (RunPod)** | ✅ DONE | **mAP50=0.528 (ep=20)** | mAP50-95=0.240, 0.8ms, 20.4M params |
| 4 | y26s no-pretrained (lokal) | ✅ DONE | mAP50=0.511 (ep=57) | scratch = pretrained! mAP50-95=0.231 |
| 5 | y26s no-augmentation (lokal) | ✅ DONE | mAP50=0.465 (ep=6) | overfit, early stop ep=56 |
| — | vanilla y26s retrain (lokal) | ✅ DONE | mAP50=0.506 (ep=21) | ≈ RunPod 0.501, dipakai E2E |
| 6 | SVM dari GT features | ✅ DONE | Macro class-MAE=0.318 | Acc±1 B1=100%, B3=91.6% |
| 7 | RF dari GT features | ✅ DONE | Macro class-MAE=0.353 | Acc±1 B1=96.8%, B3=90.5% |
| 8 | End-to-end y26s→SVM | ✅ DONE | Macro class-MAE=1.163 | Lebih buruk 3× dari heuristik |
| 9 | End-to-end y26s→RF | ✅ DONE | Macro class-MAE=1.216 | Lebih buruk 3× dari heuristik |

## Hasil #6 SVM (GT features, test set 95 trees)
| Class | MAE | Bias | Acc±1 |
|---|---:|---:|---:|
| B1 | 0.0842 | -0.042 | **100.0%** |
| B2 | 0.3368 | -0.021 | 95.8% |
| B3 | 0.5474 | +0.126 | 91.6% |
| B4 | 0.3053 | +0.011 | 96.8% |
| **Macro** | **0.3184** | — | — |
- Exact-profile acc: 27.4% | Total-count MAE: 1.1263 | Total±1: 72.6%

## Hasil #7 RF (GT features, test set 95 trees)
| Class | MAE | Bias | Acc±1 |
|---|---:|---:|---:|
| B1 | 0.1789 | -0.137 | 96.8% |
| B2 | 0.3368 | -0.021 | 96.8% |
| B3 | 0.5789 | +0.074 | 90.5% |
| B4 | 0.3158 | -0.021 | 96.8% |
| **Macro** | **0.3526** | — | — |
- Exact-profile acc: 27.4% | Total-count MAE: 1.2000 | Total±1: 70.5%
- **SVM menang vs RF** (macro MAE 0.318 vs 0.353)

---

## Hasil #8/#9 E2E (vanilla best.pt, mAP50=0.506)

| Metrik | E2E SVM | E2E RF | M01 heuristik |
|---|---:|---:|---:|
| Macro class-MAE | 1.163 | 1.216 | **0.398** |
| Total-count MAE | 2.337 | 2.337 | **1.414** |
| Total ±1 | 38.9% | 40.0% | **86.7%** |
| Exact-profile | 0.0% | 1.1% | — |

**Verdict:** Heuristik M01 tetap menang telak. Feature aggregation 13-dim kehilangan informasi side-level.

---

## Komparasi Semua Metode Counting

| Metode | Macro MAE | Total MAE | Total ±1 | Exact Profile |
|---|---:|---:|---:|---:|
| Naive sum (baseline buruk) | ~0.80 | ~3.2 | ~20% | ~0% |
| SVM (GT features) | **0.318** | 1.126 | 72.6% | 27.4% |
| RF (GT features) | 0.353 | 1.200 | 70.5% | 27.4% |
| y26s→SVM (E2E) | 1.163 | 2.337 | 38.9% | 0.0% |
| y26s→RF (E2E) | 1.216 | 2.337 | 40.0% | 1.1% |
| **M01_selector_b2b3 (target)** | **0.398** | **1.414** | **86.7%** | — |

---

## Per-class Detail — Detection (val, dari CLAUDE-TRAINING.md §0)

| Model | B1 mAP50 | B2 mAP50 | B3 mAP50 | B4 mAP50 | B1 Recall | B2 Recall | B3 Recall | B4 Recall |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| y26n vanilla | 0.728 | 0.410 | 0.576 | 0.331 | 0.784 | 0.443 | 0.621 | 0.364 |
| y26s vanilla | 0.719 | 0.393 | 0.585 | 0.308 | 0.769 | 0.488 | 0.698 | 0.242 |
| y26m vanilla | **0.757** | **0.411** | **0.595** | **0.348** | 0.688 | **0.483** | 0.672 | **0.421** |
| y26s vanilla retrain | — | — | — | — | — | — | — | — |

B4 weakest class (sample-starved, recall 24-42%). B2↔B3 irreducible visual ambiguity.

---

## Path Penting

- Dataset: `/home/claudeuser/research-method-dedup/Tested-Brand-New-Dataset-YOLO/`
- data.yaml (abs): `/home/claudeuser/research-method-dedup/local_data.yaml`
- YOLO runs: `/workspace/runs/detect/`
- Scripts: `scripts/build_counting_features.py`, `run_counting_svm.py`, `run_counting_rf.py`

---

## Catatan

- vanilla y26s retrain lokal: best ep=21, mAP50=0.506, mAP50-95=0.234 (≈ RunPod 0.501/0.235)
- Semua 9 eksperimen dari CLAUDE-TRAINING.md ✅ SELESAI
- SUMMARY.md di `baseline-run/SUMMARY.md`
- YOLO weights di `/workspace/runs/detect/{y26s_nopretrained,y26s_noaug,y26s_vanilla_local}/`
