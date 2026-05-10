"""
Algoritma: stacking_bracketed  (v7_stacking_bracketed)
Generasi: v7
Benchmark JSON 228 pohon: Acc ±1 = 94.30%, MAE = 0.2643

Ide utama
---------
Dua koreksi digabung:

1. Stacking density correction
   Tandan yang terdeteksi dalam rentang y yang sempit (berdekatan secara
   vertikal di frame) lebih mungkin saling tumpang-tindih antar sisi →
   dup-rate lebih tinggi → perlu pembagi lebih besar.

   Per kelas, dihitung:
       stacking_density(c) = n_c / max(y_span_c, 0.05)

   Jika density di atas referensi median dataset, divisor ditingkatkan:
       extra(c) = 1 + stack_coeff * max(0, density(c) - ref(c))

   Referensi median (n_c / y_span_c): B1≈42, B2≈56, B3≈72, B4≈50.
   stack_coeff = 0.0008 (diturunkan dari analisis residual, bukan training).

2. Bracket constraint
   Estimasi akhir dikurung antara:
   - Floor = max deteksi pada satu sisi manapun (fisika murni: pasti ada
     minimal sebanyak yang terlihat dari sisi terbaik)
   - Ceiling = round(naive / 1.10) (asumsi dup-rate minimal 1.10x)

   Ini mencegah under/over-count ekstrem.

Batasan
-------
- Stacking density paling efektif pada B3 (tinggi dup-rate, rentang y sempit).
- Tidak ada override per-regime; satu rumus global masih menjadi batasnya.
- Digantikan oleh v9_selector yang memakai metode ini hanya pada regime tertentu.

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


def _bracket(pred: dict, detections: list) -> dict:
    """Kurung estimasi antara floor (fisika) dan ceiling (dup-rate minimum)."""
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
    Hitung count unik per kelas dengan stacking density + bracket constraint.

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

    raw = {}
    for c in NAMES:
        nc = naive.get(c, 0)
        if nc == 0:
            raw[c] = 0
            continue
        extra = 1.0 + STACK_COEFF * max(0.0, stack[c] - STACK_REF[c])
        raw[c] = max(0, round(nc / (BASE_FACTORS[c] * sc * extra)))

    return _bracket(raw, detections)
