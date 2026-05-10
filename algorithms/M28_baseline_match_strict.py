"""
M28_baseline_match_strict
Family: baseline  | Old name: relaxed_match
Benchmark 953-tree: Acc±1 5.98%, MAE 1.8110.

Strict pair matching across sides via union-find. Two detections
of the same class on different sides are considered the same bunch
if y, area, and cx are all within thresholds.

Misleading old name "relaxed_match" referred only to the internal
threshold tolerance (relatively loose). The algorithm itself is
strict matching and fails catastrophically on noisy YOLO labels.

Kept as a baseline floor — DO NOT use in production.
"""

import numpy as np

NAMES = ["B1", "B2", "B3", "B4"]


class _UF:
    def __init__(self, n):
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.r[rx] < self.r[ry]:
            rx, ry = ry, rx
        self.p[ry] = rx
        if self.r[rx] == self.r[ry]:
            self.r[rx] += 1


def predict(
    detections: list,
    y_thresh: float = 0.15,
    area_thresh: float = 0.12,
    cx_thresh: float = 0.35,
) -> dict:
    out = {}
    for c in NAMES:
        cd = [d for d in detections if d["class"] == c]
        n = len(cd)
        if n == 0:
            out[c] = 0
            continue
        if n == 1:
            out[c] = 1
            continue
        uf = _UF(n)
        for i in range(n):
            for j in range(i + 1, n):
                if cd[i]["side_index"] == cd[j]["side_index"]:
                    continue
                if abs(cd[i]["y_norm"] - cd[j]["y_norm"]) > y_thresh:
                    continue
                if abs(np.sqrt(cd[i]["area_norm"]) - np.sqrt(cd[j]["area_norm"])) > area_thresh:
                    continue
                if abs(cd[i]["x_norm"] - cd[j]["x_norm"]) > cx_thresh:
                    continue
                uf.union(i, j)
        out[c] = len({uf.find(i) for i in range(n)})
    return out
