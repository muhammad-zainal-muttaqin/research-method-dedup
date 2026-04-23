"""
Algoritma: median_strong5  (v9_median_strong5)
Generasi: v9
Benchmark JSON 228 pohon: Acc ±1 = 95.18%, MAE = 0.2390

Ide utama
---------
Ensemble dari 5 metode terkuat — ambil median per kelas dari keluaran
kelimanya:

    1. v6_selector
    2. stacking_bracketed  (v7)
    3. b2_b4_boosted  (v8)
    4. floor_anchor_50  (v8)
    5. per_side_median  (v8)

Intuisi: jika mayoritas dari 5 estimator setuju pada suatu nilai, median
lebih robust daripada memilih satu metode saja.

Keuntungan: MAE-nya adalah yang terbaik kedua (0.2390) setelah v9_selector
(0.2533). Ini karena per_side_median menarik estimasi ke bawah untuk pohon
yang over-predicted, mengurangi MAE rata-rata meski Acc-nya tidak optimal.

Kelemahan: Acc hanya 95.18% — di bawah v6_selector (96.49%) dan
v9_selector (98.68%). Median bisa mengompromikan kasus di mana satu
metode benar dan empat salah.

Ketergantungan
--------------
Memerlukan params dari v6_selector.load_params().

Input
-----
detections : list[dict]
    - "class": str
    - "x_norm": float
    - "y_norm": float
    - "side_index": int

params : dict
    Dari v6_selector.load_params().

Output
------
dict[str, int]
"""

import numpy as np

NAMES = ["B1", "B2", "B3", "B4"]


def predict(detections: list, params: dict) -> dict:
    """
    Hitung count unik per kelas dengan median dari 5 estimator kuat.

    Parameters
    ----------
    detections : list[dict]
        Field wajib: "class", "x_norm", "y_norm", "side_index".
    params : dict
        Dari v6_selector.load_params().

    Returns
    -------
    dict[str, int]
    """
    from algorithms.v6_selector import predict as v6_predict
    from algorithms.stacking_bracketed import predict as stacking_pred
    from algorithms.b2_b4_boosted import predict as b2b4_pred
    from algorithms.floor_anchor_50 import predict as floor_pred
    from algorithms.per_side_median import predict as median_pred

    preds = [
        v6_predict(detections, params),
        stacking_pred(detections),
        b2b4_pred(detections),
        floor_pred(detections),
        median_pred(detections),
    ]
    return {
        c: int(round(float(np.median([p[c] for p in preds]))))
        for c in NAMES
    }
