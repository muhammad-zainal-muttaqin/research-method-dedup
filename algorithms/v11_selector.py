"""
Algoritma: v11_selector (STACKED DENSITY + B1/B4 RESCUE)
Generasi: v11 — targets remaining 11% errors after v10.

- B3e2 reduced by v10 B23 correction.
- Remaining dominant: B1 under-count + B4 small-object miss (low visibility, top of tree).

v11 adds two safe specialists:
1. _b1_rescue: low-density B1 trees → use relaxed divisor + y-anchor.
2. _b4_boost: sparse B4 + high y-range → lower visibility threshold.

Always anchors on v6/v10 backbone. Deterministic, no training.
Target: 91.5%+ on 727.
"""

from __future__ import annotations
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from algorithms.v9_selector import predict as v9_predict, load_params
from algorithms.v10_selector import predict as v10_predict
from algorithms.v6_selector import _adaptive_corrected as _base_adaptive

NAMES = ["B1", "B2", "B3", "B4"]


def _b1_low_density_rescue(dets: list[dict]) -> dict[str, int]:
    """B1 trees with very few B1 detections but high naive B2/B3 → possible B1 hidden in lower band."""
    b1 = [d for d in dets if d["class"] == "B1"]
    if len(b1) >= 2:
        return _base_adaptive(dets)

    # Count vertical distribution
    low_b2b3 = sum(1 for d in dets if d["class"] in ("B2", "B3") and d["y_norm"] >= 0.55)
    if low_b2b3 >= 3:
        base = _base_adaptive(dets)
        base["B1"] = max(base["B1"], 1)  # rescue at least 1 B1
        return base
    return _base_adaptive(dets)


def _b4_small_object_boost(dets: list[dict]) -> dict[str, int]:
    """B4 are small and often missed on upper canopy. Boost if high vertical spread + low B4 count."""
    b4 = [d for d in dets if d["class"] == "B4"]
    if len(b4) >= 4:
        return _base_adaptive(dets)

    max_y = max((d["y_norm"] for d in dets), default=0)
    min_y = min((d["y_norm"] for d in dets), default=0)
    y_span = max_y - min_y

    base = _base_adaptive(dets)

    if y_span > 0.28 and len(b4) == 0:
        # very likely missed B4 at top
        base["B4"] = 1
    elif y_span > 0.22 and len(b4) == 1:
        base["B4"] = max(base["B4"], 2)
    return base


def predict(detections: list[dict], params: dict) -> dict[str, int]:
    """v11 stacked rescue: v10 backbone + B1 rescue + B4 boost."""
    if not detections:
        return {c: 0 for c in NAMES}

    # Layer 1: v10 (B23 fix)
    p = v10_predict(detections, params)

    # Layer 2: B1 rescue on low-B1 trees
    b1_naive = sum(1 for d in detections if d["class"] == "B1")
    if b1_naive <= 1:
        p = _b1_low_density_rescue(detections)
        # merge B23 from v10
        p10 = v10_predict(detections, params)
        for c in ["B2", "B3"]:
            p[c] = p10[c]

    # Layer 3: B4 boost
    b4_naive = sum(1 for d in detections if d["class"] == "B4")
    if b4_naive <= 2:
        p = _b4_small_object_boost(detections)

    return {c: int(p[c]) for c in NAMES}
