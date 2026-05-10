"""
M06_weight_visibility
Family: weight  | Old name: visibility / v2_visibility / visibility_count
Benchmark 953-tree: Acc±1 85.94%, MAE 0.3960.

Gauss visibility weighting on x_norm. Detection at frame center
(x_norm ≈ 0.5) → weight ~1 (likely unique). Detection at edge →
lower weight (likely seen from neighboring side).
"""

import numpy as np

NAMES = ["B1", "B2", "B3", "B4"]


def predict(detections: list, alpha: float = 1.0, sigma: float = 0.3) -> dict:
    out = {}
    for c in NAMES:
        cd = [d for d in detections if d["class"] == c]
        if not cd:
            out[c] = 0
            continue
        total = sum(
            1.0 / (1.0 + alpha * np.exp(-((d["x_norm"] - 0.5) ** 2) / (2.0 * sigma ** 2)))
            for d in cd
        )
        out[c] = max(0, int(round(total)))
    return out
