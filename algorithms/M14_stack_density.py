"""
Algoritma: stacking_density  (v7_stacking_density)
Generasi: v7
Benchmark JSON 228 pohon: Acc ±1 = 94.30%, MAE = 0.2708

Ide utama
---------
Versi tanpa bracket constraint dari stacking_bracketed. Hitung per-class
vertical stacking density lalu gunakan sebagai modulator divisor.

    stacking_density(c) = n_c / max(y_span_c, 0.05)

Jika density di atas referensi median dataset, divisor dinaikkan:
    extra(c) = 1 + stack_coeff * max(0, density(c) - ref(c))

Referensi median: B1≈42, B2≈56, B3≈72, B4≈50.
stack_coeff = 0.0008.

Perbedaan dari stacking_bracketed: tidak ada floor/ceiling constraint.
Hasil bisa lebih under- atau over-predict pada kasus ekstrem.

Acc sama dengan stacking_bracketed (94.30%) tapi MAE lebih tinggi (0.2708
vs 0.2643), sehingga stacking_bracketed lebih direkomendasikan.

Input
-----
detections : list[dict]
    - "class": str
    - "y_norm": float  (cy YOLO)

Output
------
dict[str, int]
"""

from collections import Counter

import numpy as np

NAMES = ["B1", "B2", "B3", "B4"]
BASE_FACTORS = {"B1": 1.986, "B2": 1.786, "B3": 1.795, "B4": 1.655}
STACK_REF = {"B1": 42.0, "B2": 56.0, "B3": 72.0, "B4": 50.0}
STACK_COEFF = 0.0008


def _density_scale(n_total: int) -> float:
    return float(np.clip(2.05 - 0.014 * n_total, 1.45, 2.10) / 1.79)


def _per_class_stack_density(detections: list) -> dict:
    density = {}
    for c in NAMES:
        cd = [d for d in detections if d["class"] == c]
        if not cd:
            density[c] = 0.0
            continue
        y_vals = [d["y_norm"] for d in cd]
        y_span = max(y_vals) - min(y_vals) if len(y_vals) > 1 else 0.1
        density[c] = len(cd) / max(y_span, 0.05)
    return density


def predict(detections: list) -> dict:
    """
    Hitung count unik per kelas dengan stacking density correction (tanpa bracket).

    Parameters
    ----------
    detections : list[dict]
        Field wajib: "class", "y_norm".

    Returns
    -------
    dict[str, int]
    """
    naive = Counter(d["class"] for d in detections)
    n_total = len(detections)
    sc = _density_scale(n_total)
    stack = _per_class_stack_density(detections)

    result = {}
    for c in NAMES:
        nc = naive.get(c, 0)
        if nc == 0:
            result[c] = 0
            continue
        extra = 1.0 + STACK_COEFF * max(0.0, stack[c] - STACK_REF[c])
        result[c] = max(0, round(nc / (BASE_FACTORS[c] * sc * extra)))
    return result
