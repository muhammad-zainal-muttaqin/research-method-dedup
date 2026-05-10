"""
Algoritma: entropy_modulated  (v8_entropy_modulated)
Generasi: v8
Benchmark JSON 228 pohon: Acc ±1 = 94.30%, MAE = 0.2763

Ide utama
---------
Jika deteksi kelas C pada sebuah pohon terkonsentrasi di satu sisi (entropi
rendah), itu artinya tandan-tandan tersebut hanya terlihat dari satu sudut
pandang → kemungkinan besar unik, bukan duplikat → dup-rate lebih rendah →
divisor harus lebih kecil.

Sebaliknya, jika deteksi tersebar merata ke semua sisi (entropi tinggi),
setiap tandan terlihat dari banyak sisi → dup-rate lebih tinggi → divisor
penuh dipakai.

Rumus:
    H_norm(c) = Shannon entropy dari distribusi per-sisi, dinormalisasi ke [0, 1]
    ent_scale(c) = 1 - entropy_coeff * (1 - H_norm(c))

Jadi:
    - H_norm = 1 (sempurna merata)  → ent_scale = 1.0 (divisor normal)
    - H_norm = 0 (satu sisi saja)   → ent_scale = 1 - entropy_coeff (divisor lebih kecil)

entropy_coeff = 0.25, dikombinasikan dengan stacking density correction dari v7.

Batasan
-------
- Tidak memberi breakthrough beyond v7; menunjukkan bahwa entropy tidak
  menambah sinyal yang cukup di atas stacking density.
- Tidak ada override per-regime; digantikan oleh v9_selector.

Input
-----
detections : list[dict]
    Setiap elemen adalah bounding box dengan field:
      - "class": str          → "B1", "B2", "B3", atau "B4"
      - "y_norm": float       → koordinat pusat vertikal (YOLO cy), range [0, 1]
      - "side_index": int     → indeks sisi foto (0-based)

Output
------
dict[str, int]
    Count unik per kelas: {"B1": int, "B2": int, "B3": int, "B4": int}
"""

from collections import Counter

import numpy as np

NAMES = ["B1", "B2", "B3", "B4"]
BASE_FACTORS = {"B1": 1.986, "B2": 1.786, "B3": 1.795, "B4": 1.655}
STACK_REF = {"B1": 42.0, "B2": 56.0, "B3": 72.0, "B4": 50.0}
STACK_COEFF = 0.0008
MIN_DUP_RATIO = 1.10
ENTROPY_COEFF = 0.25


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


def _per_side_counts(detections: list) -> dict:
    result = {}
    for c in NAMES:
        cd = [d for d in detections if d["class"] == c]
        result[c] = Counter(d["side_index"] for d in cd)
    return result


def _class_entropy(side_counts: Counter, n_sides_total: int) -> float:
    """Normalized Shannon entropy dari distribusi per-sisi kelas."""
    if not side_counts:
        return 1.0
    counts = np.array([side_counts.get(s, 0) for s in range(n_sides_total)], dtype=float)
    total = counts.sum()
    if total == 0:
        return 1.0
    p = counts / total
    p = p[p > 0]
    H = -np.sum(p * np.log(p))
    H_max = np.log(n_sides_total) if n_sides_total > 1 else 1.0
    return float(H / H_max) if H_max > 0 else 1.0


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
    Hitung count unik per kelas dengan entropy modulation + stacking density.

    Parameters
    ----------
    detections : list[dict]
        Daftar bounding box dari semua sisi pohon.
        Setiap elemen harus memiliki field "class", "y_norm", "side_index".

    Returns
    -------
    dict[str, int]
        Count unik per kelas B1–B4.
    """
    naive = Counter(d["class"] for d in detections)
    n_total = len(detections)
    sc = _density_scale(n_total)
    n_sides_total = max(_n_sides(detections), 1)
    psc = _per_side_counts(detections)
    stack = _per_class_stack_density(detections)

    raw = {}
    for c in NAMES:
        nc = naive.get(c, 0)
        if nc == 0:
            raw[c] = 0
            continue
        H = _class_entropy(psc[c], n_sides_total)
        ent_scale = 1.0 - ENTROPY_COEFF * (1.0 - H)
        stack_extra = 1.0 + STACK_COEFF * max(0.0, stack[c] - STACK_REF[c])
        divisor = BASE_FACTORS[c] * sc * ent_scale * stack_extra
        raw[c] = max(0, round(nc / divisor))

    return _bracket(raw, detections)
