"""
M27_weight_visibility_adaptive
Family: weight  | Old name: adaptive_visibility
Benchmark 953-tree: not in primary ranking (experimental).

Visibility with alpha and sigma adapted to detection density and
y-range. Dense or vertically-spread trees get tighter sigma + larger
alpha; sparse trees get the opposite.
"""

import numpy as np

from algorithms.M06_weight_visibility import predict as _visibility

NAMES = ["B1", "B2", "B3", "B4"]


def predict(detections: list) -> dict:
    n_total = len(detections)
    y_vals = [d["y_norm"] for d in detections]
    y_span = (max(y_vals) - min(y_vals)) if y_vals else 0.5
    density = n_total / 12.0
    alpha = 1.0 * (1.35 - 0.35 * min(density, 1.6))
    sigma = 0.3 * (0.55 + 0.45 * min(density, 1.6))
    if y_span > 0.7:
        sigma *= 0.88
        alpha *= 1.08
    elif y_span < 0.3:
        sigma *= 1.18
        alpha *= 0.92
    return _visibility(
        detections,
        alpha=float(np.clip(alpha, 0.5, 1.6)),
        sigma=float(np.clip(sigma, 0.12, 0.55)),
    )
