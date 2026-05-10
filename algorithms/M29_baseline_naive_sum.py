"""
M29_baseline_naive_sum
Family: baseline  | Old name: naive
Benchmark 953-tree: Acc±1 3.99%, MAE 2.2800.

Raw sum of detections per class across all sides. Ignores duplicates
across viewpoints. Worst possible reference baseline. Overcounts
~83.4% on average (multi-side overlap factor 1.834).
"""

from collections import Counter

NAMES = ["B1", "B2", "B3", "B4"]


def predict(detections: list) -> dict:
    n = Counter(d["class"] for d in detections)
    return {c: int(n.get(c, 0)) for c in NAMES}
