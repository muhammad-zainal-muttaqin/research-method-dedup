"""
Algoritma: ordinal_b3  (v7_ordinal_b3)
Generasi: v7
Benchmark JSON 228 pohon: Acc ±1 = 91.23%, MAE = 0.2939

Catatan penting: metode ini BROKEN pada dataset non-JSON (bisa menghasilkan
count negatif). Tidak direkomendasikan untuk produksi. Tercantum di sini
hanya untuk kelengkapan audit.

Ide utama
---------
Secara ordinal, B1 berada paling bawah (y_norm tinggi), B4 paling atas
(y_norm rendah). B3 yang posisinya berada di rentang y tipikal B2 diduga
merupakan deteksi yang lebih jarang (satu sisi saja) → dup-rate lebih
rendah. B3 di rentang y tipikal B3 sendiri lebih sering ter-duplikasi
antar sisi.

Strategi: pisahkan deteksi B3 menjadi dua grup berdasarkan y_norm:
    boundary = (median_y_B2 + median_y_B3) / 2

    low_B3  = deteksi B3 dengan y_norm < boundary  (dekat zona B2)
    high_B3 = deteksi B3 dengan y_norm ≥ boundary  (zona B3 normal)

    pred_low  = round(n_low  / (base_div * low_factor_boost))   # boost 1.12
    pred_high = round(n_high / base_div)
    pred_B3   = pred_low + pred_high

Ini bukan reklasifikasi — label tidak diubah. Hanya divisor yang berbeda.
Bracket constraint diterapkan setelah prediksi.

Y_MEDIANS harus dihitung dari dataset sebelum memanggil predict().
Gunakan compute_y_medians() yang tersedia di modul ini.

Keterbatasan
------------
- Performa global 91.23% — lebih rendah dari adaptive_corrected (93.86%).
- Broken pada non-JSON: bisa output negatif.
- Dikecualikan dari rekomendasi metode.

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
LOW_FACTOR_BOOST = 1.12

# Diisi oleh compute_y_medians() sebelum predict() dipanggil.
Y_MEDIANS: dict = {}


def compute_y_medians(all_detections_list: list) -> dict:
    """
    Hitung median y_norm per kelas dari seluruh dataset.

    Parameters
    ----------
    all_detections_list : list[list[dict]]
        List of detections per tree (outer list = per tree).

    Returns
    -------
    dict[str, float]
        Median y_norm per kelas.
    """
    global Y_MEDIANS
    y_vals = {c: [] for c in NAMES}
    for dets in all_detections_list:
        for d in dets:
            if d["class"] in y_vals:
                y_vals[d["class"]].append(d["y_norm"])
    Y_MEDIANS = {
        c: float(np.median(y_vals[c])) if y_vals[c] else 0.5
        for c in NAMES
    }
    return Y_MEDIANS


def _density_scale(n_total: int) -> float:
    return float(np.clip(2.05 - 0.014 * n_total, 1.45, 2.10) / 1.79)


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
    Hitung count unik per kelas dengan ordinal B3 split modulation.

    Catatan: compute_y_medians() harus dipanggil terlebih dahulu.
    Metode ini tidak disarankan untuk produksi — gunakan v9_selector.

    Parameters
    ----------
    detections : list[dict]
        Field wajib: "class", "y_norm", "side_index".

    Returns
    -------
    dict[str, int]
    """
    if not Y_MEDIANS:
        raise RuntimeError("Y_MEDIANS belum dihitung. Panggil compute_y_medians() dulu.")

    naive = Counter(d["class"] for d in detections)
    n_total = len(detections)
    sc = _density_scale(n_total)

    result = {}
    for c in NAMES:
        nc = naive.get(c, 0)
        if nc == 0:
            result[c] = 0
            continue
        if c != "B3":
            result[c] = max(0, round(nc / (BASE_FACTORS[c] * sc)))
        else:
            cd = [d for d in detections if d["class"] == "B3"]
            boundary = (Y_MEDIANS.get("B2", 0.5) + Y_MEDIANS.get("B3", 0.5)) / 2.0
            low_b3 = [d for d in cd if d["y_norm"] < boundary]
            high_b3 = [d for d in cd if d["y_norm"] >= boundary]
            base_div = BASE_FACTORS["B3"] * sc
            pred_low = round(len(low_b3) / (base_div * LOW_FACTOR_BOOST)) if low_b3 else 0
            pred_high = round(len(high_b3) / base_div) if high_b3 else 0
            result["B3"] = max(0, pred_low + pred_high)

    return _bracket(result, detections)
