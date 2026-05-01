"""
Algoritma: v13_selector  (STACKING v9 + Correction Layer)
Generasi: v13 - Gunakan v9 sebagai base, tambahkan correction layer

Ide:
----
1. Start dengan v9 prediction
2. Deteksi kondisi dimana v9 cenderung salah (dianalisis dari 725 files)
3. Apply correction untuk kondisi tersebut

Correction rules (dari analisis failure cases):
- B3_ratio > 3.5 & B3_naive > 10 -> undercount, tambah 1 B3
- B4_naive > 5 & B4_yrange < 0.05 -> overcount, kurangi 1 B4  
- B2_ratio > 3.0 & B2_naive > 6 -> undercount, tambah 1 B2
"""

from collections import Counter
import numpy as np

NAMES = ["B1", "B2", "B3", "B4"]

# Import v9 sebagai base
from algorithms.v9_selector import predict as v9_predict, load_params as v9_load_params


def _should_correct_b3(feat: dict) -> bool:
    """Detect B3 undercount condition."""
    return feat.get("B3_ratio", 0) > 3.3 and feat.get("B3_naive", 0) > 8


def _should_correct_b2(feat: dict) -> bool:
    """Detect B2 undercount condition."""
    return feat.get("B2_ratio", 0) > 3.0 and feat.get("B2_naive", 0) > 6


def _should_correct_b4(feat: dict) -> bool:
    """Detect B4 overcount condition."""
    return feat.get("B4_naive", 0) > 5 and feat.get("B4_yrange", 0) < 0.08


def predict(detections: list) -> dict:
    """v13: v9 + correction layer."""
    if not detections:
        return {c: 0 for c in NAMES}
    
    # Get v9 prediction
    params = v9_load_params()
    pred = v9_predict(detections, params).copy()
    
    # Extract features untuk correction detection
    by_class = {c: [d for d in detections if d["class"] == c] for c in NAMES}
    naive = {c: len(by_class[c]) for c in NAMES}
    side_counts = {c: Counter(d["side_index"] for d in by_class[c]) for c in NAMES}
    
    feat = {}
    for c in NAMES:
        ys = [d["y_norm"] for d in by_class[c]]
        max_side = max(side_counts[c].values(), default=0)
        feat[f"{c}_naive"] = naive[c]
        feat[f"{c}_maxside"] = max_side
        feat[f"{c}_ratio"] = naive[c] / max(max_side, 1) if naive[c] else 0.0
        feat[f"{c}_yrange"] = (max(ys) - min(ys)) if ys else 0.0
    
    feat["total_det"] = len(detections)
    
    # Apply corrections
    if _should_correct_b3(feat):
        pred["B3"] = pred.get("B3", 0) + 1
    
    if _should_correct_b2(feat):
        pred["B2"] = pred.get("B2", 0) + 1
    
    if _should_correct_b4(feat):
        pred["B4"] = max(0, pred.get("B4", 0) - 1)
    
    return pred
