"""
Algoritma: floor_anchor_50  (v8_floor_anchor_50)
Generasi: v8
Benchmark JSON 228 pohon: Acc ±1 = 69.74%, MAE = 0.4211

Ide utama
---------
Metode ini bukan untuk dipakai secara global — performa globalnya rendah.
Ia dirancang sebagai spesialis untuk regime pohon tertentu, dan memang
dipakai oleh v9_selector hanya pada kondisi "b3b4_only_lowtotal".

Ide dasarnya: pada pohon dengan sedikit deteksi dan semua deteksi terlihat
dari semua sisi, estimasi stacking density biasanya terlalu tinggi. Solusi:
"angker" estimasi ke arah floor (nilai minimum yang mungkin), bukan ke arah
divisor penuh.

Untuk tiap kelas C:
1. Hitung estimasi stacking_density (E_stack)
2. Hitung floor = max deteksi per sisi untuk kelas C
3. Jika E_stack ≤ floor + 1: pakai E_stack (sudah konservatif)
4. Jika E_stack > floor + 1:
       estimate = floor + anchor * (E_stack - floor)
       anchor = 0.50  → tepat di tengah antara floor dan stacking estimate

Dengan anchor=0.50, metode ini cenderung under-predict pada pohon padat
(karena floor-nya sudah tinggi), tapi lebih akurat pada pohon dengan
deteksi terbatas.

Batasan
-------
- Secara global menghasilkan 69.74% — di bawah baseline. Hanya efektif
  pada subset pohon dengan total deteksi ≤ 13 dan pola B3/B4-only.
- Tidak boleh dipakai tanpa regime filter (v9_selector).

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
ANCHOR = 0.50


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


def _per_side_counts(detections: list) -> dict:
    result = {}
    for c in NAMES:
        cd = [d for d in detections if d["class"] == c]
        result[c] = Counter(d["side_index"] for d in cd)
    return result


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
    Hitung count unik per kelas dengan floor-anchored estimator (anchor=0.50).

    Catatan: Metode ini dirancang sebagai spesialis untuk pohon dengan
    deteksi sangat sedikit dan pola B3/B4-only. Tidak disarankan dipakai
    secara global — gunakan v9_selector yang sudah menyertakan regime filter.

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
    stack = _per_class_stack_density(detections)
    psc = _per_side_counts(detections)

    raw = {}
    for c in NAMES:
        nc = naive.get(c, 0)
        if nc == 0:
            raw[c] = 0
            continue
        extra = 1.0 + STACK_COEFF * max(0.0, stack[c] - STACK_REF[c])
        sd_est = nc / (BASE_FACTORS[c] * sc * extra)

        side_counts = psc[c]
        floor = float(max(side_counts.values())) if side_counts else 0.0

        if sd_est <= floor + 1.0:
            raw[c] = max(0, round(sd_est))
        else:
            compromise = floor + ANCHOR * (sd_est - floor)
            raw[c] = max(0, round(compromise))

    return _bracket(raw, detections)
