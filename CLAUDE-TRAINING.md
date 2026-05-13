# CLAUDE-TRAINING.md

> Onboarding doc untuk Claude Code di RunPod/Vast.ai.
> Scope: detection + counting ML pipeline. Dedup heuristic scope di `CLAUDE.md` (root) — **jangan campur**.

---

## 0. State Sekarang (2026-05-13)

### Eksperimen DONE (9/9) — semua selesai 2026-05-13

| # | Model | Best epoch | mAP50 | mAP50-95 | best.pt path |
|---|---|---:|---:|---:|---|
| 1 | YOLO26n vanilla | 30 | 0.511 | 0.237 | `runs/detect/sawit-ulm/vanilla-train/y26n/weights/best.pt` |
| 2 | YOLO26s vanilla | 32 | 0.501 | 0.235 | `runs/detect/sawit-ulm/vanilla-train/y26s/weights/best.pt` |
| 3 | **YOLO26m vanilla** | 20 | **0.528** | **0.240** | `runs/detect/sawit-ulm/vanilla-train/y26m/weights/best.pt` |

Log mentah: `baseline-run/vanilla_y26{n,s,m}.txt`.

| # | Model | Best epoch | mAP50 | mAP50-95 | Notes |
|---:|---|---:|---:|---:|---|
| 4 | y26s no-pretrained | 57 | **0.511** | 0.231 | scratch = pretrained! |
| 5 | y26s no-aug | 6 | **0.465** | 0.216 | overfit cepat, early stop ep=56 |
| 6 | SVM (GT feat) | — | — | — | Macro class-MAE=0.318 |
| 7 | RF (GT feat) | — | — | — | Macro class-MAE=0.353 |
| 8 | y26s→SVM E2E | — | — | — | Macro MAE=1.163 |
| 9 | y26s→RF E2E | — | — | — | Macro MAE=1.216 |
| — | vanilla y26s retrain | 21 | **0.506** | 0.234 | lokal, ≈ RunPod 0.501 |

### Per-class detail (val, mAP50)

| Model | B1 | B2 | B3 | B4 | Speed (ms) | Params |
|---|---:|---:|---:|---:|---:|---:|
| y26n | 0.728 | 0.410 | 0.576 | 0.331 | 0.2 | 2.4M |
| y26s | 0.719 | 0.393 | 0.585 | 0.308 | 0.5 | 9.5M |
| y26m | 0.757 | 0.411 | 0.595 | 0.348 | 0.8 | 20.4M |

### Per-class recall (val)

| Model | B1 R | B2 R | B3 R | B4 R |
|---|---:|---:|---:|---:|
| y26n | 0.784 | 0.443 | 0.621 | 0.364 |
| y26s | 0.769 | 0.488 | 0.698 | 0.242 |
| y26m | 0.688 | 0.483 | 0.672 | 0.421 |

**Observasi:**
- y26m menang tipis, tapi y26n hampir setara dengan **4× lebih cepat** (0.2ms vs 0.8ms)
- B4 lemah di semua model (mAP50 0.31-0.35) — sample-starved + visual overlap
- B2↔B3 konsisten lemah (irreducible — lihat CLAUDE.md JSON-01 audit)
- y26m converge cepat (best epoch 20) → model besar overfit cepat di dataset kecil

### Eksperimen DONE (9/9)

- [x] **#4** Ablasi: y26s tanpa pretrained → mAP50=0.511 (ep=57, scratch = pretrained!)
- [x] **#5** Ablasi: y26s tanpa augmentasi → mAP50=0.465 (ep=6, overfit cepat)
- [x] **#6** SVM dari GT features → Macro class-MAE=0.318
- [x] **#7** Random Forest dari GT features → Macro class-MAE=0.353
- [x] **#8** End-to-end: y26s → SVM → Macro class-MAE=1.163 (heuristik menang)
- [x] **#9** End-to-end: y26s → RF → Macro class-MAE=1.216 (heuristik menang)

### Kesimpulan

- **Heuristik M01_selector_b2b3 tetap production choice** — tidak ada ML pipeline yang bisa menyaingi.
- ML counting butuh feature yang side-aware, bukan naive 13-dim aggregation.

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
  --repo-type dataset --local-dir ./Tested-Brand-New-Dataset-YOLO
```

**Verifikasi:**
- `Tested-Brand-New-Dataset-YOLO/data.yaml` exist
- `Tested-Brand-New-Dataset-YOLO/json/` berisi 953 file
- `Tested-Brand-New-Dataset-YOLO/labels/{train,val,test}/` berisi 3164 / 416 / 412 TXT

---

## 2. Vanilla Hyperparameter (referensi konsistensi)

Salin dari `baseline-run/vanilla_y26s.txt` baris 1-112. Highlight:

```yaml
optimizer: auto
patience: 50
batch: 16            # y26s default; y26n bisa 64, y26m bisa 16-32
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
  data=Tested-Brand-New-Dataset-YOLO/data.yaml \
  epochs=100 batch=16 imgsz=640 patience=50 \
  optimizer=auto seed=42 deterministic=True \
  project=baseline-run name=y26s_nopretrained
```

Catatan: `model=yolo26s.yaml` (bukan `.pt`) → build from scratch.
**Ekspektasi:** mAP50 turun 5-15 pp ke 0.35-0.45. Buktikan pretrained penting di dataset kecil.

Simpan log: `baseline-run/y26s_nopretrained.txt`.

---

## 4. Eksperimen #5 — Ablasi No-Augmentation

**Tujuan:** ukur kontribusi augmentasi.

```bash
yolo detect train \
  model=yolo26s.pt pretrained=True \
  data=Tested-Brand-New-Dataset-YOLO/data.yaml \
  epochs=100 batch=16 imgsz=640 patience=50 \
  hsv_h=0 hsv_s=0 hsv_v=0 \
  degrees=0 translate=0 scale=0 shear=0 perspective=0 \
  flipud=0 fliplr=0 \
  mosaic=0 mixup=0 cutmix=0 copy_paste=0 erasing=0 \
  auto_augment=None close_mosaic=0 \
  optimizer=auto seed=42 deterministic=True \
  project=baseline-run name=y26s_noaug
```

**Ekspektasi:** overfit train, val mAP50 turun ke 0.40-0.48.

Simpan log: `baseline-run/y26s_noaug.txt`.

---

## 5. Eksperimen #6, #7 — Counting dari GT Features

### Input

- Source: `Tested-Brand-New-Dataset-YOLO/json/*.json` (953 tree files)
- Schema: `images.sisi_N.annotations[]` — list of detections per side
- Target: `summary.by_class.{B1,B2,B3,B4}` — 4-dim integer vector
- Split: `Tested-Brand-New-Dataset-YOLO/split_manifest.csv`

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

Output: `predictions/y26s_inference/<tree_name>.json`.

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
| Training log | `baseline-run/<exp_name>.txt` |
| Weights | `runs/detect/.../<exp_name>/weights/best.pt` |
| Training curve | `runs/.../results.csv` |
| Confusion matrix | `runs/.../confusion_matrix_normalized.png` |
| Per-class metrics | `reports/<exp_name>/per_class_metrics.csv` |
| Counting eval | `reports/<exp_name>/metrics.json` |

---

## 8. Final Reporting

Setelah 9 eksperimen done, generate `baseline-run/SUMMARY.md`:

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
