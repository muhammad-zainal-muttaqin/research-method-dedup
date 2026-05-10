"""
M15_divide_global
Family: divide  | Old name: corrected / v1_corrected / corrected_naive
Benchmark 953-tree: Acc±1 84.37%, MAE 0.4160.

First-generation method. Divides naive count by per-class fixed
duplication factor. Factors are median ratio naive/GT computed on
the 228-tree JSON snapshot.
"""

from collections import Counter

NAMES = ["B1", "B2", "B3", "B4"]
FACTORS = {"B1": 1.986, "B2": 1.786, "B3": 1.795, "B4": 1.655}


def predict(detections: list) -> dict:
    n = Counter(d["class"] for d in detections)
    return {c: max(0, round(n.get(c, 0) / FACTORS[c])) for c in NAMES}
