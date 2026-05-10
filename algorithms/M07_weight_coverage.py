"""
M07_weight_coverage
Family: weight  | Old name: side_coverage
Benchmark 953-tree: Acc±1 85.94%, MAE 0.3930.

Visibility result clamped between max_per_side (physical floor) and
naive count (ceiling). Each side cannot show more than the true
number of unique bunches, and total cannot exceed naive sum.
"""

from collections import Counter

from algorithms.M06_weight_visibility import predict as _visibility

NAMES = ["B1", "B2", "B3", "B4"]


def predict(detections: list) -> dict:
    vis = _visibility(detections)
    n = Counter(d["class"] for d in detections)
    out = {}
    for c in NAMES:
        cd = [d for d in detections if d["class"] == c]
        if not cd:
            out[c] = 0
            continue
        max_per_side = max(Counter(d["side_index"] for d in cd).values())
        out[c] = min(max(vis[c], max_per_side), n.get(c, 0))
    return out
