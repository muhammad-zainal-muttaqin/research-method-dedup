"""
Algoritma: adaptive_corrected
Generasi: v5
Benchmark JSON 228 pohon: Acc ±1 = 93.86%, MAE = 0.2774

Ide utama
---------
Naive sum langsung dibagi faktor duplikasi per kelas. Faktor ini tidak
tetap — ia disesuaikan (adaptive) berdasarkan jumlah total deteksi pohon.

Mengapa dup-rate perlu diadaptasi?
Pohon yang padat (banyak tandan, banyak deteksi) cenderung memiliki
dup-rate lebih rendah karena tandan-tandan saling menghalangi antar sisi,
sehingga tidak semuanya terlihat dari setiap sisi. Sebaliknya, pohon jarang
memiliki dup-rate lebih tinggi. Hubungan ini dimodelkan sebagai:

    dup_rate(n) = clip(2.05 - 0.014 * n, 1.45, 2.10)

lalu diubah menjadi scale factor relatif terhadap rata-rata 1.79:

    scale = dup_rate(n) / 1.79

Faktor per kelas (B1/B2/B3/B4) dikalikan scale ini sebelum dipakai
sebagai pembagi. Faktor dasar diturunkan dari median rasio naive/GT
pada 228 pohon JSON.

Batasan
-------
- Satu rumus global; tidak cukup optimal untuk semua pola pohon.
- Digantikan oleh v6_selector dan v9_selector untuk akurasi lebih tinggi.

Input
-----
detections : list[dict]
    Setiap elemen adalah satu bounding box dengan field:
      - "class": str  → "B1", "B2", "B3", atau "B4"
    Field lain (bbox_yolo, side_index, dll.) tidak dipakai algoritma ini.

Output
------
dict[str, int]
    Count unik per kelas: {"B1": int, "B2": int, "B3": int, "B4": int}
"""

from collections import Counter

import numpy as np

NAMES = ["B1", "B2", "B3", "B4"]

# Median rasio naive/GT per kelas, dihitung dari 228 pohon JSON.
BASE_FACTORS = {"B1": 1.986, "B2": 1.786, "B3": 1.795, "B4": 1.655}


def _density_scale(n_total: int) -> float:
    """
    Hitung scale factor adaptif berdasarkan total deteksi pohon.
    Semakin padat pohon, semakin kecil dup-rate → scale < 1.
    """
    dup_rate = np.clip(2.05 - 0.014 * n_total, 1.45, 2.10)
    return float(dup_rate / 1.79)


def predict(detections: list) -> dict:
    """
    Hitung count unik per kelas dengan adaptive_corrected.

    Parameters
    ----------
    detections : list[dict]
        Daftar bounding box dari semua sisi pohon.

    Returns
    -------
    dict[str, int]
        Count unik per kelas B1–B4.
    """
    n_total = len(detections)
    scale = _density_scale(n_total)
    naive = Counter(d["class"] for d in detections)
    return {
        c: max(0, round(naive.get(c, 0) / (BASE_FACTORS[c] * scale)))
        for c in NAMES
    }
