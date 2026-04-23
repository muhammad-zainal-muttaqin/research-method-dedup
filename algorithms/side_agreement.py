"""
Algoritma: side_agreement  (v8_side_agreement)
Generasi: v8
Benchmark JSON 228 pohon: Acc ±1 = 83.33%, MAE = 0.3618

Ide utama
---------
Estimasi dup-rate per kelas per pohon secara langsung dari distribusi
deteksi antar sisi, tanpa menggunakan BASE_FACTORS global.

Ukuran yang dipakai adalah agreement_ratio:
    agreement_ratio(c) = max_side_count(c) / mean_side_count(c)

Interpretasi fisika:
    - Jika satu sisi mendominasi (agreement tinggi), tandan-tandan satu
      kelas banyak yang hanya muncul dari satu sudut → lebih sedikit
      duplikasi total → divisor lebih besar.
    - Jika tiap sisi melihat jumlah yang hampir sama (agreement rendah),
      tiap tandan muncul di banyak sisi → duplikasi tinggi.

Rumus:
    agreement_scale(c) = 1 + agreement_coeff * max(0, ratio - 1)
    agreement_scale di-cap pada 1.5 untuk mencegah over-koreksi.
    agreement_coeff = 0.40

Performa global 83.33% — metode ini terlalu konservatif (undercount) pada
pohon padat karena agreement_ratio-nya tinggi mengakibatkan divisor terlalu
besar.

Input
-----
detections : list[dict]
    - "class": str
    - "y_norm": float
    - "side_index": int

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
MIN_DUP_RATIO = 1.10
AGREEMENT_COEFF = 0.40


def _density_scale(n_total: int) -> float:
    return float(np.clip(2.05 - 0.014 * n_total, 1.45, 2.10) / 1.79)


def _n_sides(detections: list) -> int:
    return len(set(d["side_index"] for d in detections)) if detections else 1


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


def _bracket(pred: dict, detections: list) -> dict:
    naive = Counter(d["class"] for d in detections)
    result = {}
    for c in NAMES:
        cd = [d for d in detections if d["class"] == c]
        if not cd:
            result[c] = 0
            continue
        per_side = Counter(d["side_index"] for d in cd)
        floor = max(per_side.values()) if per_side else 0
        ceiling = max(floor, round(naive.get(c, 0) / MIN_DUP_RATIO))
        result[c] = int(np.clip(pred[c], floor, ceiling))
    return result


def predict(detections: list) -> dict:
    """
    Hitung count unik per kelas dengan side agreement ratio modulation.

    Parameters
    ----------
    detections : list[dict]
        Field wajib: "class", "y_norm", "side_index".

    Returns
    -------
    dict[str, int]
    """
    naive = Counter(d["class"] for d in detections)
    n_total = len(detections)
    sc = _density_scale(n_total)
    n_sides_total = max(_n_sides(detections), 1)
    stack = _per_class_stack_density(detections)

    raw = {}
    for c in NAMES:
        nc = naive.get(c, 0)
        if nc == 0:
            raw[c] = 0
            continue
        psc = Counter(d["side_index"] for d in detections if d["class"] == c)
        if not psc:
            raw[c] = 0
            continue
        max_side = max(psc.values())
        mean_side = nc / n_sides_total
        agreement_ratio = max_side / max(mean_side, 0.5)
        agreement_scale = 1.0 + AGREEMENT_COEFF * max(0.0, agreement_ratio - 1.0)
        agreement_scale = min(agreement_scale, 1.5)
        stack_extra = 1.0 + STACK_COEFF * max(0.0, stack[c] - STACK_REF[c])
        divisor = BASE_FACTORS[c] * sc * agreement_scale * stack_extra
        raw[c] = max(0, round(nc / divisor))

    return _bracket(raw, detections)
