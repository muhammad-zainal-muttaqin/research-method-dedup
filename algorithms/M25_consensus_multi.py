"""
Algoritma: multi_consensus  (v8_multi_consensus)
Generasi: v8
Benchmark JSON 228 pohon: Acc ±1 = 18.86%, MAE = 0.9583

Catatan penting: performa global sama buruknya dengan per_side_median karena
per_side_median adalah salah satu komponen utamanya. Tidak direkomendasikan
dipakai secara mandiri.

Ide utama
---------
Gabungkan tiga estimator berbeda dan ambil median-nya per kelas:

    E1 = stacking_density estimate (corrected naive berbasis divisor)
    E2 = per_side_median estimate
    E3 = floor = max deteksi dari sisi manapun

Intuisi: jika ketiga estimator setuju, hasilnya lebih terpercaya. Jika
ada outlier (misalnya E3 jauh di bawah E1), median lebih robust daripada
mean.

Dalam praktik, E2 (per_side_median) cenderung sangat rendah sehingga
menarik median ke bawah, menyebabkan undercount sistematis.

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
    Hitung count unik per kelas dengan median dari 3 estimator.

    Peringatan: performa global 18.86% — undercount ekstrem karena
    per_side_median menarik median ke bawah. Tidak disarankan.

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
    n_sides_total = _n_sides(detections)
    stack = _per_class_stack_density(detections)

    raw = {}
    for c in NAMES:
        nc = naive.get(c, 0)
        if nc == 0:
            raw[c] = 0
            continue

        # E1: stacking density estimate
        extra = 1.0 + STACK_COEFF * max(0.0, stack[c] - STACK_REF[c])
        e1 = nc / (BASE_FACTORS[c] * sc * extra)

        # E2: per-side median (termasuk padding sisi kosong)
        psc = Counter(d["side_index"] for d in detections if d["class"] == c)
        counts_padded = [psc.get(s, 0) for s in range(n_sides_total)]
        med = float(np.median(counts_padded))
        top2 = sorted(psc.values(), reverse=True)[:2]
        top2_mean = np.mean(top2) if top2 else 0.0
        e2 = max(med, top2_mean * 0.65)

        # E3: floor (max per sisi)
        e3 = float(max(psc.values())) if psc else 0.0

        raw[c] = max(0, round(float(np.median([e1, e2, e3]))))

    return _bracket(raw, detections)
