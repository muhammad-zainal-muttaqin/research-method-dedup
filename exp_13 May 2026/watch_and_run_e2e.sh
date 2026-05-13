#!/bin/bash
# Tunggu y26s_nopretrained selesai, lalu langsung jalankan E2E inference + #8/#9
set -e
export PATH="$PATH:/home/claudeuser/.local/bin"
REPO="/home/claudeuser/research-method-dedup"
WEIGHTS="/workspace/runs/detect/y26s_nopretrained/weights/best.pt"
LOG="$REPO/baseline-run"

echo "[WATCHER] Tunggu y26s_nopretrained training selesai... $(date)"

# Tunggu sampai log punya "Training complete" atau "Results saved to"
while true; do
    if grep -q "Results saved to\|Training complete\|EarlyStopping\|model saved\|Speed:" "$LOG/y26s_nopretrained.txt" 2>/dev/null; then
        # Cek juga bahwa best.pt ada
        if [ -f "$WEIGHTS" ]; then
            echo "[WATCHER] Training #4 selesai! $(date)"
            break
        fi
    fi
    echo "[WATCHER] Masih training... $(date) ($(wc -l < "$REPO"/baseline-run/y26s_nopretrained.txt 2>/dev/null || echo 0) lines in log)"
    sleep 60
done

# E2E inference (GPU, bisa jalan paralel dengan training #5 yang sudah mulai)
echo "[E2E INFERENCE] Start: $(date)"
cd "$REPO"
python3 scripts/run_e2e_inference.py --weights "$WEIGHTS" 2>&1 | tee "$LOG/e2e_inference.txt"
echo "[E2E INFERENCE] Done: $(date)"

# Exp #8
echo "[EXP #8 E2E-SVM] Start: $(date)"
python3 scripts/run_e2e_svm.py 2>&1 | tee "$LOG/e2e_svm.txt"
echo "[EXP #8] Done: $(date)"

# Exp #9
echo "[EXP #9 E2E-RF] Start: $(date)"
python3 scripts/run_e2e_rf.py 2>&1 | tee "$LOG/e2e_rf.txt"
echo "[EXP #9] Done: $(date)"

# Update SUMMARY
python3 scripts/generate_training_summary.py 2>&1 | tee -a "$LOG/e2e_rf.txt"
echo "[WATCHER] SEMUA E2E SELESAI. $(date)"
