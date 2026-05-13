# Eksperimen 13 May 2026 — ML Training Pipeline

Session ini mengerjakan eksperimen #4–#9 dari CLAUDE-TRAINING.md.
GPU: NVIDIA RTX A5000 24GB | Workspace: /workspace/runs/detect/

---

## Status

| # | Eksperimen | Status | Log | Notes |
|---|---|---|---|---|
| 4 | y26s no-pretrained | ✅ DONE | baseline-run/y26s_nopretrained.txt | Best ep=57, mAP50=0.511, mAP50-95=0.231 — mengungguli vanilla! |
| 5 | y26s no-augmentation | ✅ DONE | baseline-run/y26s_noaug.txt | Best ep=6, mAP50=0.465, early stop ep=56 — overfit cepat |
| 6 | SVM dari GT features | ✅ DONE | reports/counting_svm/ | Macro class-MAE=0.3184, Acc±1 B3=91.6% |
| 7 | RF dari GT features | ✅ DONE | reports/counting_rf/ | Macro class-MAE=0.3526, Acc±1 B3=90.5% |
| 8 | End-to-end y26s→SVM | ✅ DONE | reports/e2e_svm/ | Macro class-MAE=1.150 — heuristik menang |
| 9 | End-to-end y26s→RF | ✅ DONE | reports/e2e_rf/ | Macro class-MAE=1.253 — heuristik menang |
| - | vanilla y26s retrain | ✅ DONE | baseline-run/y26s_vanilla_local.txt | referensi untuk #8/#9 |

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
