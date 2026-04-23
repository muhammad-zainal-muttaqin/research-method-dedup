"""
Algoritma: per_side_median  (v8_per_side_median)
Generasi: v8
Benchmark JSON 228 pohon: Acc ±1 = 18.86%, MAE = 0.9583

Catatan penting: performa global sangat buruk (undercount ekstrem).
Metode ini TIDAK direkomendasikan dipakai secara mandiri. Dicantumkan
untuk kelengkapan audit.

Ide utama
---------
Bukan berbasis divisor — sepenuhnya berbasis distribusi per sisi.

Untuk tiap kelas C:
    1. Hitung count per sisi (termasuk sisi yang melihat 0 untuk kelas ini).
    2. Ambil MEDIAN dari semua sisi (termasuk yang bernilai 0).
    3. Bandingkan dengan rata-rata 2 sisi terbaik × 0.65.
    4. Ambil yang lebih besar dari keduanya sebagai estimasi.

Intuisinya: jika suatu kelas hanya terlihat dari 1-2 sisi, median yang
menyertakan sisi kosong akan sangat rendah. Maka diambil top-2 mean × 0.65
sebagai lower bound.

Bracket constraint tetap diterapkan.

Mengapa gagal secara global: pada pohon dengan 4 sisi dan kelas yang hanya
muncul di 1-2 sisi, sisi kosong mendominasi median → estimasi terlalu rendah.
Ini adalah kasus yang sangat umum di dataset ini.

Input
-----
detections : list[dict]
    - "class": str
    - "side_index": int

Output
------
dict[str, int]
"""

from collections import Counter

import numpy as np

NAMES = ["B1", "B2", "B3", "B4"]
MIN_DUP_RATIO = 1.10


def _n_sides(detections: list) -> int:
    return len(set(d["side_index"] for d in detections)) if detections else 1


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
    Hitung count unik per kelas dengan median per-sisi.

    Peringatan: performa global 18.86% — undercount ekstrem. Tidak
    disarankan dipakai tanpa filtering regime.

    Parameters
    ----------
    detections : list[dict]
        Field wajib: "class", "side_index".

    Returns
    -------
    dict[str, int]
    """
    n_sides_total = _n_sides(detections)
    raw = {}
    for c in NAMES:
        cd = [d for d in detections if d["class"] == c]
        if not cd:
            raw[c] = 0
            continue
        psc = Counter(d["side_index"] for d in cd)
        # Pad sisi kosong dengan 0
        counts_padded = [psc.get(s, 0) for s in range(n_sides_total)]
        med = float(np.median(counts_padded))
        top2 = sorted(psc.values(), reverse=True)[:2]
        top2_mean = np.mean(top2) if top2 else 0.0
        raw[c] = max(0, round(max(med, top2_mean * 0.65)))

    return _bracket(raw, detections)
