#!/bin/bash
# Sequential YOLO training queue: #4 → #5 → vanilla_local (untuk #8/#9)
set -e
export PATH="$PATH:/home/claudeuser/.local/bin"
REPO="/home/claudeuser/research-method-dedup"
DATA="$REPO/local_data.yaml"
RUNS="/workspace/runs/detect"
LOG="$REPO/baseline-run"

echo "=============================="
echo "EXP #4: y26s no-pretrained"
echo "=============================="
yolo detect train \
  model=yolo26s.yaml \
  pretrained=False \
  data="$DATA" \
  epochs=100 batch=16 imgsz=640 patience=50 \
  optimizer=auto seed=42 deterministic=True \
  project="$RUNS" name=y26s_nopretrained \
  2>&1 | tee "$LOG/y26s_nopretrained.txt"

echo "=============================="
echo "EXP #5: y26s no-augmentation"
echo "=============================="
yolo detect train \
  model=yolo26s.pt pretrained=True \
  data="$DATA" \
  epochs=100 batch=16 imgsz=640 patience=50 \
  hsv_h=0 hsv_s=0 hsv_v=0 \
  degrees=0 translate=0 scale=0 shear=0 perspective=0 \
  flipud=0 fliplr=0 \
  mosaic=0 mixup=0 cutmix=0 copy_paste=0 erasing=0 \
  auto_augment=None close_mosaic=0 \
  optimizer=auto seed=42 deterministic=True \
  project="$RUNS" name=y26s_noaug \
  2>&1 | tee "$LOG/y26s_noaug.txt"

echo "=============================="
echo "Vanilla y26s retrain (for #8/#9)"
echo "=============================="
yolo detect train \
  model=yolo26s.pt pretrained=True \
  data="$DATA" \
  epochs=100 batch=16 imgsz=640 patience=50 \
  optimizer=auto seed=42 deterministic=True \
  project="$RUNS" name=y26s_vanilla_local \
  2>&1 | tee "$LOG/y26s_vanilla_local.txt"

echo "=============================="
echo "All YOLO training DONE"
echo "=============================="
