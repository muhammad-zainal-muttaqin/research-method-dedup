# Eksperimen 13 May 2026 — ML Training Pipeline

Session ini mengerjakan eksperimen #4–#9 dari CLAUDE-TRAINING.md.
GPU: NVIDIA RTX A5000 24GB | Workspace: /workspace/runs/detect/

---

## Status

| # | Eksperimen | Status | Best Result | Detail |
|---:|---|---|---:|---|
| 1 | y26n vanilla (lokal, batch=16) | ✅ DONE | mAP50=0.521 (ep=38) | mAP50-95=0.237, 0.2ms, 2.4M params |
| 2 | y26s vanilla (lokal, batch=16) | ✅ DONE | mAP50=0.506 (ep=21) | mAP50-95=0.235, 0.5ms, 9.5M params |
| 3 | y26m vanilla (lokal, batch=16) | ✅ DONE | mAP50=0.509 (ep=33) | mAP50-95=0.231, 0.8ms, 20.4M params |
| 4 | y26s no-pretrained (lokal) | ✅ DONE | mAP50=0.511 (ep=57) | scratch = pretrained! mAP50-95=0.231 |
| 5 | y26s no-augmentation (lokal) | ✅ DONE | mAP50=0.465 (ep=6) | overfit, early stop ep=56 |
| 6 | SVM dari GT features | ✅ DONE | Macro Acc±1=96.1% | MAE=0.318, B1=100%, B3=91.6% |
| 7 | RF dari GT features | ✅ DONE | Macro Acc±1=95.3% | MAE=0.353, B1=96.8%, B3=90.5% |
| 8–22 | E2E: 5 detektor × {SVM, RF, M01} | ✅ DONE | y26m→SVM terbaik: 71.6% | Semua batch=16, konsisten |

Semua 15 kombinasi E2E selesai. Semua weights di `baseline-run/weights/`.

---

## Konfigurasi Training (Semua Konsisten)

| Parameter | Nilai |
|---|---|
| batch | **16** |
| imgsz | 640 |
| epochs | 100 |
| patience | 50 |
| seed | 42 |
| optimizer | auto |
| data | local_data.yaml |

---

## Hasil Detection (lokal, batch=16)

| Model | Best Epoch | mAP50 | mAP50-95 | Catatan |
|---|---:|---:|---:|---|
| **y26n vanilla** | 38 | **0.521** | 0.237 | Tercepat (0.2ms), mAP50 tertinggi lokal |
| y26s vanilla | 21 | 0.506 | 0.235 | Baseline E2E |
| y26m vanilla | 33 | 0.509 | 0.231 | Terbesar (20.4M params) |
| y26s no-pretrained | 57 | 0.511 | 0.231 | Scratch = pretrained! |
| y26s no-aug | 6 | 0.465 | 0.216 | Overfit, early stop ep=56 |

---

## Hasil #6 SVM (GT features, test set 95 trees)
| Class | MAE | Bias | Acc±1 |
|---|---:|---:|---:|
| B1 | 0.0842 | -0.042 | **100.0%** |
| B2 | 0.3368 | -0.021 | 95.8% |
| B3 | 0.5474 | +0.126 | 91.6% |
| B4 | 0.3053 | +0.011 | 96.8% |
| **Macro** | **0.318** | — | **96.1%** |
- Exact-profile: 27.4% | Total MAE: 1.126 | Total±1: 72.6%

## Hasil #7 RF (GT features, test set 95 trees)
| Class | MAE | Bias | Acc±1 |
|---|---:|---:|---:|
| B1 | 0.1789 | -0.137 | 96.8% |
| B2 | 0.3368 | -0.021 | 96.8% |
| B3 | 0.5789 | +0.074 | 90.5% |
| B4 | 0.3158 | -0.021 | 96.8% |
| **Macro** | **0.353** | — | **95.3%** |
- Exact-profile: 27.4% | Total MAE: 1.200 | Total±1: 70.5%

---

## Hasil E2E — Matrix Lengkap (test set, n=95)

| Detektor | mAP50 | Counting | Macro Acc±1 | Macro MAE |
|---|---:|---|---:|---:|
| y26n vanilla | 0.521 | SVM | 70.0% | 1.145 |
| y26n vanilla | 0.521 | RF | 68.2% | 1.218 |
| y26n vanilla | 0.521 | M01 | 67.1% | 1.337 |
| y26s vanilla | 0.506 | SVM | 68.9% | 1.163 |
| y26s vanilla | 0.506 | RF | 66.6% | 1.216 |
| y26s vanilla | 0.506 | M01 | 65.5% | 1.403 |
| y26s scratch | 0.511 | SVM | 68.9% | 1.145 |
| y26s scratch | 0.511 | RF | 67.9% | 1.229 |
| y26s scratch | 0.511 | M01 | 69.2% | 1.266 |
| y26s no-aug | 0.465 | SVM | 70.5% | 1.126 |
| y26s no-aug | 0.465 | RF | 68.4% | 1.184 |
| y26s no-aug | 0.465 | M01 | 66.6% | 1.384 |
| **y26m vanilla** | 0.509 | **SVM** | **71.6%** | **1.118** |
| y26m vanilla | 0.509 | RF | 67.9% | 1.211 |
| y26m vanilla | 0.509 | M01 | 64.5% | 1.400 |
| **M01 GT (target)** | — | — | **86.7%** | **0.398** |

**Pipeline terbaik:** y26m → SVM (71.6%), masih 15pp di bawah M01 GT.

---

## Path Penting

- Dataset: `/home/claudeuser/research-method-dedup/Tested-Brand-New-Dataset-YOLO/`
- data.yaml: `/home/claudeuser/research-method-dedup/local_data.yaml`
- Weights: `baseline-run/weights/{y26n,y26s,y26m}_vanilla_local.pt` + ablasi
- Inference: `predictions/{name}_inference/` (953 JSONs per detektor)
- Reports E2E: `reports/e2e_{name}_{method}/metrics.json`
- Script unified: `scripts/run_e2e_pipeline.py --name NAME --weights PATH`
