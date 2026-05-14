# CLAUDE-TRAINING.md

> Onboarding doc untuk Claude Code di RunPod/Vast.ai.
> Scope: detection + counting ML pipeline. Dedup heuristic scope di `CLAUDE.md` (root) — **jangan campur**.

---

## 0. State Sekarang (2026-05-13) — SEMUA SELESAI

Seluruh eksperimen deteksi, counting, dan E2E telah selesai. Semua model dilatih secara lokal dengan konfigurasi konsisten: `batch=16`, `imgsz=640`, `epochs=100`, `patience=50`, `seed=42`. Semua bobot tersimpan di `ml-track/baseline-run/weights/`.

### Hasil Deteksi (lokal, batch=16, test set)

| # | Model | Best Epoch | mAP50 | mAP50-95 | best.pt |
|---:|---|---:|---:|---:|---|
| 1 | **y26n vanilla** | 38 | **0.521** | 0.237 | `ml-track/baseline-run/weights/y26n_vanilla_local.pt` |
| 2 | y26s vanilla | 21 | 0.506 | 0.235 | `ml-track/baseline-run/weights/y26s_vanilla_local.pt` |
| 3 | y26m vanilla | 33 | 0.509 | 0.231 | `ml-track/baseline-run/weights/y26m_vanilla_local.pt` |
| 4 | y26s no-pretrained | 57 | 0.511 | 0.231 | `ml-track/baseline-run/weights/y26s_nopretrained.pt` |
| 5 | y26s no-augmentation | 6 | 0.465 | 0.216 | `ml-track/baseline-run/weights/y26s_noaug.pt` |

### Hasil Counting — Fitur GT (test set, n=95)

| # | Model | Macro MAE | Macro Acc ±1 | Acc±1 B1 | Acc±1 B3 | Profil Tepat |
|---:|---|---:|---:|---:|---:|---:|
| 6 | **SVM (GT feat)** | **0.318** | **96.1%** | 100.0% | 91.6% | 27.4% |
| 7 | RF (GT feat) | 0.353 | 95.3% | 96.8% | 90.5% | 27.4% |
| — | M01 heuristik (target) | 0.398 | 86.7% | — | — | 26.3% |

### Hasil E2E — Matrix Lengkap (test set, n=95)

| Detektor | mAP50 | Penghitung | Macro Acc ±1 | Macro MAE |
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
| M01 GT (target) | — | — | **86.7%** | **0.398** |

### Kesimpulan

- **Heuristik M01_selector_b2b3 tetap production choice** — Macro Acc±1 = 86.7%, valid per RULES.txt.
- **SVM dengan fitur GT (96.1%) mengungguli M01** — desain fitur 13-dim terbukti adequate; bottleneck E2E ada di detektor.
- **Pipeline E2E terbaik: y26m → SVM (71.6%)** — masih 15 pp di bawah heuristik GT.
- Seluruh 15 kombinasi E2E berada dalam rentang 64–72%: algoritma penghitung bukan penyebab utama kegagalan.
- Script unified: `python scripts/run_e2e_pipeline.py --name NAME --weights PATH`

---

## 0.1. Reproduksi Hasil (Tanpa Training Ulang)

Semua weights sudah ada di repo (`ml-track/baseline-run/weights/`). Dataset diunduh dari HF. Urutan reproduksi:

### Prasyarat

```bash
git clone https://github.com/muhammad-zainal-muttaqin/research-method-dedup.git
cd research-method-dedup
pip install -r requirements.txt
pip install scikit-learn huggingface_hub ultralytics

huggingface-cli login
huggingface-cli download ULM-DS-Lab/OilPalm-MultiView-BunchCount-YOLO \
  --repo-type dataset --local-dir ./Brand-New-Dataset-YOLO
```

### Step 1: Reproduksi Deteksi — mAP50 / mAP50-95 (butuh GPU)

```bash
for WEIGHT in \
  ml-track/baseline-run/weights/y26n_vanilla_local.pt \
  ml-track/baseline-run/weights/y26s_vanilla_local.pt \
  ml-track/baseline-run/weights/y26m_vanilla_local.pt \
  ml-track/baseline-run/weights/y26s_nopretrained.pt \
  ml-track/baseline-run/weights/y26s_noaug.pt; do
  yolo detect val model=$WEIGHT \
    data=Brand-New-Dataset-YOLO/data.yaml split=test
done
```

Expected output (test set, n=412):

| Weight | mAP50 | mAP50-95 |
|---|---:|---:|
| y26n_vanilla_local.pt | 0.521 | 0.237 |
| y26s_vanilla_local.pt | 0.506 | 0.235 |
| y26m_vanilla_local.pt | 0.509 | 0.231 |
| y26s_nopretrained.pt  | 0.511 | 0.231 |
| y26s_noaug.pt         | 0.465 | 0.216 |

Toleransi: ±0.003 (deterministic, `seed=42` baked ke weights).

### Step 2: Reproduksi Counting GT-Features — tanpa GPU

```bash
python scripts/build_counting_features.py   # ekstrak 13-dim features dari GT JSON
python scripts/run_counting_svm.py          # → reports/counting_svm/metrics.json
python scripts/run_counting_rf.py           # → reports/counting_rf/metrics.json
```

Verifikasi:

```bash
python -c "
import json
classes = ['B1', 'B2', 'B3', 'B4']
for name in ['counting_svm', 'counting_rf']:
    d = json.load(open(f'reports/{name}/metrics.json'))['test']
    acc = sum(d[f'acc_pm1_{c}'] for c in classes) / len(classes)
    print(f\"{name}: Acc±1={acc:.1%}, MAE={d['macro_class_mae']:.3f}\")
"
```

Expected: `counting_svm: Acc±1=96.1%, MAE=0.318` | `counting_rf: Acc±1=95.3%, MAE=0.353`

### Step 3: Reproduksi E2E — butuh GPU

```bash
# Jalankan semua 5 detektor (masing-masing run inference + SVM + RF + M01)
for NAME_WEIGHT in \
  "y26n_vanilla_local:ml-track/baseline-run/weights/y26n_vanilla_local.pt" \
  "y26s_vanilla_local:ml-track/baseline-run/weights/y26s_vanilla_local.pt" \
  "y26m_vanilla_local:ml-track/baseline-run/weights/y26m_vanilla_local.pt" \
  "y26s_nopretrained:ml-track/baseline-run/weights/y26s_nopretrained.pt" \
  "y26s_noaug:ml-track/baseline-run/weights/y26s_noaug.pt"; do
  NAME="${NAME_WEIGHT%%:*}"
  WEIGHT="${NAME_WEIGHT##*:}"
  python scripts/run_e2e_pipeline.py --name $NAME --weights $WEIGHT
done
```

Output per detektor: `reports/e2e_{name}_{svm,rf,m01}/metrics.json`

Verifikasi best E2E (y26m → SVM):

```bash
python -c "
import json
d = json.load(open('reports/e2e_y26m_vanilla_local_svm/metrics.json'))['test']
print(f\"E2E best: Acc±1={d['macro_acc_pm1']:.1%}, MAE={d['macro_class_mae']:.3f}\")
"
```

Expected: `E2E best: Acc±1=71.6%, MAE=1.118`

### Step 4: Generate SUMMARY

```bash
# Catatan: generate_training_summary.py menggunakan RUNS_DIR=/workspace/runs/detect
# (path RunPod). Di lokal tanpa runs/, hanya membaca reports/ dan ml-track/baseline-run/*.txt
python scripts/generate_training_summary.py   # → ml-track/baseline-run/SUMMARY.md
```

---

## 1. Environment Setup (RunPod / Vast.ai)

**GPU:** 24GB+ VRAM (RTX 4090 / A100 / RTX 6000 Blackwell).
**Software:** Python 3.10+, CUDA 12.1+.

```bash
# Clone repo
git clone https://github.com/muhammad-zainal-muttaqin/research-method-dedup.git
cd research-method-dedup

# Dependencies
pip install -r requirements.txt
pip install scikit-learn huggingface_hub

# Pull dataset dari HF (tidak commit ke git, terlalu besar)
huggingface-cli login                       # masukkan token, jangan hardcode
huggingface-cli download ULM-DS-Lab/OilPalm-MultiView-BunchCount-YOLO \
  --repo-type dataset --local-dir ./Brand-New-Dataset-YOLO
```

**Verifikasi:**
- `Brand-New-Dataset-YOLO/data.yaml` exist
- `Brand-New-Dataset-YOLO/json/` berisi 953 file
- `Brand-New-Dataset-YOLO/labels/` berisi 3992 TXT flat (split membership lewat `train.txt`/`val.txt`/`test.txt`)

---

## 2. Vanilla Hyperparameter (referensi konsistensi)

Salin dari `ml-track/baseline-run/vanilla_y26s.txt` baris 1-112. Highlight:

```yaml
optimizer: auto
patience: 50
batch: 16            # WAJIB sama untuk semua eksperimen — jangan ubah
imgsz: 640
cos_lr: False
seed: 42
deterministic: True

# Loss gains (default)
box: 7.5
cls: 0.5
dfl: 1.5

# Augmentation (default Ultralytics)
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
translate: 0.1
scale: 0.5
fliplr: 0.5
mosaic: 1.0
mixup: 0.0
erasing: 0.4
auto_augment: randaugment
close_mosaic: 10
```

Gunakan setting ini untuk eksperimen #4 dan #5, hanya ubah variabel ablasi.

---

## 3. Eksperimen #4 — Ablasi No-Pretrained

**Tujuan:** ukur kontribusi COCO pretrained weights vs scratch.

```bash
yolo detect train \
  model=yolo26s.yaml \
  pretrained=False \
  data=Brand-New-Dataset-YOLO/data.yaml \
  epochs=100 batch=16 imgsz=640 patience=50 \
  optimizer=auto seed=42 deterministic=True \
  project=ml-track/baseline-run name=y26s_nopretrained
```

Catatan: `model=yolo26s.yaml` (bukan `.pt`) → build from scratch.
**Ekspektasi:** mAP50 turun 5-15 pp ke 0.35-0.45. Buktikan pretrained penting di dataset kecil.

Simpan log: `ml-track/baseline-run/y26s_nopretrained.txt`.

---

## 4. Eksperimen #5 — Ablasi No-Augmentation

**Tujuan:** ukur kontribusi augmentasi.

```bash
yolo detect train \
  model=yolo26s.pt pretrained=True \
  data=Brand-New-Dataset-YOLO/data.yaml \
  epochs=100 batch=16 imgsz=640 patience=50 \
  hsv_h=0 hsv_s=0 hsv_v=0 \
  degrees=0 translate=0 scale=0 shear=0 perspective=0 \
  flipud=0 fliplr=0 \
  mosaic=0 mixup=0 cutmix=0 copy_paste=0 erasing=0 \
  auto_augment=None close_mosaic=0 \
  optimizer=auto seed=42 deterministic=True \
  project=ml-track/baseline-run name=y26s_noaug
```

**Ekspektasi:** overfit train, val mAP50 turun ke 0.40-0.48.

Simpan log: `ml-track/baseline-run/y26s_noaug.txt`.

---

## 5. Eksperimen #6, #7 — Counting dari GT Features

### Input

- Source: `Brand-New-Dataset-YOLO/json/*.json` (953 tree files)
- Schema: `images.sisi_N.annotations[]` — list of detections per side
- Target: `summary.by_class.{B1,B2,B3,B4}` — 4-dim integer vector
- Split: `Brand-New-Dataset-YOLO/split_manifest.csv`

### Feature Engineering (per tree, 13 dim)

| Group | Features | Count |
|---|---|---:|
| Naive sum | `naive_sum_B1..B4` | 4 |
| Max per side | `max_per_side_B1..B4` | 4 |
| Mean per side | `mean_per_side_B1..B4` | 4 |
| Meta | `n_sides` | 1 |

Lokasi script: `scripts/build_counting_features.py` (buat baru).

### Model #6 — SVM

```python
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svr", MultiOutputRegressor(SVR(kernel="rbf"))),
])
grid = {
    "svr__estimator__C": [0.1, 1, 10],
    "svr__estimator__gamma": ["scale", 0.01, 0.1],
}
gs = GridSearchCV(pipe, grid, cv=3, scoring="neg_mean_absolute_error", n_jobs=-1)
gs.fit(X_train, y_train)
y_pred_test = gs.predict(X_test)
```

### Model #7 — Random Forest

```python
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=200, max_depth=10,
    random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)
y_pred_test = rf.predict(X_test)
# Round to nearest integer for count
y_pred_test = np.clip(np.round(y_pred_test), 0, None).astype(int)
```

### Output

- `reports/counting_svm/metrics.json`, `predictions.csv`, `per_class_mae.csv`
- `reports/counting_rf/metrics.json`, `predictions.csv`, `feature_importance.csv`

### Metrik WAJIB (per CLAUDE.md §Decision Metric)

1. **Per-class MAE** (`MAE_B1`, `MAE_B2`, `MAE_B3`, `MAE_B4`)
2. **Macro class-MAE** (rata-rata 4 per-class MAE)
3. **Exact-profile accuracy** (% trees dengan prediksi `[B1,B2,B3,B4]` = GT exact)
4. **Total-count MAE** (MAE dari sum B1+B2+B3+B4 per tree)
5. **Total ±1 accuracy** (% trees dengan total prediksi dalam ±1 dari GT)
6. **Per-class mean error** (signed bias)
7. **Primary: Acc ±1 per class** (% trees within ±1 error tiap kelas)

---

## 6. Eksperimen #8, #9 — End-to-End

Pipeline: y26s `best.pt` → predict 4-8 sides per tree → ekstrak feature → SVM/RF → 4-class count.

### Step 1: Inference y26s pada semua tree

```python
from ultralytics import YOLO
import json, glob

model = YOLO("runs/detect/sawit-ulm/vanilla-train/y26s/weights/best.pt")
# Untuk setiap tree, run pada images/{split}/<tree_name>_<side>.jpg
# Output: JSON dengan schema mirror GT
```

Output: `ml-track/predictions/y26s_vanilla_local_inference/<tree_name>.json`.

### Step 2: Feature engineering dari prediksi detector

Sama persis seperti #6/#7, tapi source = inference JSON, bukan GT JSON.

### Step 3: Counting

**Rekomendasi: retrain SVM/RF dari scratch dengan detector features**, bukan reuse model dari #6/#7. GT-trained model akan bias ke distribusi GT yang lebih bersih.

### Step 4: Bandingkan

| Metrik | M01_selector_b2b3 (heuristik) | y26s→SVM | y26s→RF |
|---|---:|---:|---:|
| Acc ±1 | 86.67% | ? | ? |
| Macro class-MAE | 0.3982 | ? | ? |
| Total-count MAE | 1.4145 | ? | ? |

**Verdict:** kalau ML ≥ heuristik → ML pipeline winner. Kalau tidak → heuristik tetap menang, ML pipeline jadi reference baseline.

---

## 7. Output Convention

Setiap eksperimen WAJIB simpan:

| Artifact | Lokasi |
|---|---|
| Training log | `ml-track/baseline-run/<exp_name>.txt` |
| Weights | `runs/detect/.../<exp_name>/weights/best.pt` |
| Training curve | `runs/.../results.csv` |
| Confusion matrix | `runs/.../confusion_matrix_normalized.png` |
| Per-class metrics | `reports/<exp_name>/per_class_metrics.csv` |
| Counting eval | `reports/<exp_name>/metrics.json` |

---

## 8. Final Reporting

Setelah 9 eksperimen done, generate `ml-track/baseline-run/SUMMARY.md`:

1. Tabel komparasi 5 detection: n, s, m, no-pretrained, no-aug
2. Tabel komparasi 2 counting: SVM, RF
3. Tabel komparasi 2 end-to-end vs M01 heuristik
4. Naratif: winner per kategori + trade-off + recommendation

---

## 9. Constraint (Jangan Dilanggar)

- ❌ Jangan retrain di dataset dedup (953 trees JSON di `json/` legacy) — scope CLAUDE.md
- ❌ Jangan modifikasi `algorithms/M*.py` (dedup heuristik)
- ❌ Jangan hardcode HF token — gunakan `huggingface-cli login` atau env var `HUGGING_FACE_HUB_TOKEN`
- ❌ Jangan commit `runs/` ke git (heavy artifacts)
- ✅ Commit hanya: `best.pt` per eksperimen, `metrics.json`, log `.txt`
- ✅ Push branch terpisah `training-pipeline` agar tidak konflik dengan dedup `main`

---

## 10. Reference Cepat

| Item | Value |
|---|---|
| Class distribution (train) | B1=11%, B2=18%, B3=52% (dominan), B4=18% |
| Dataset split | 3,164 train / 416 val / 412 test |
| Trees total | 953 (multi-view 4-8 sides/tree) |
| HF dataset repo | `ULM-DS-Lab/OilPalm-MultiView-BunchCount-YOLO` |
| GitHub repo | `muhammad-zainal-muttaqin/research-method-dedup` |
| Branch usulan | `training-pipeline` |
| Best dedup heuristik (target ML) | `M01_selector_b2b3` — 86.67% Acc±1 |
| Hard ambiguity | B2↔B3 visual overlap (irreducible per JSON-01) |
| Weak class | B4 (sample-starved, recall 24-42% vanilla) |
