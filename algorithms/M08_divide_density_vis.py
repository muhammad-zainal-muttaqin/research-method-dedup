"""
M08_divide_density_vis
Family: divide  | Old name: density_scaled_vis
Benchmark 953-tree: Acc±1 85.94%, MAE 0.4020.

Visibility output multiplied by a density-aware boost. Dense trees
get a small upward correction (more bunches missed by visibility);
sparse trees get a slight downward correction.
"""

import numpy as np

from algorithms.M06_weight_visibility import predict as _visibility

NAMES = ["B1", "B2", "B3", "B4"]


def predict(detections: list) -> dict:
    n_total = len(detections)
    vis = _visibility(detections)
    boost = float(np.clip(1.0 + 0.025 * (n_total - 12) / 12.0, 0.92, 1.15))
    return {c: max(0, int(round(vis[c] * boost))) for c in NAMES}
