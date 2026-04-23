"""
Algoritma: b2_median_v6  (v9_b2_median_v6)
Generasi: v9
Benchmark JSON 228 pohon: Acc ±1 = 96.49%, MAE = 0.2588

Ide utama
---------
v6_selector sangat baik untuk semua kelas kecuali B2 yang kadang
over-predicted. Solusi targeted: pakai v6_selector untuk B1, B3, B4 —
tapi khusus untuk B2, ambil median dari 5 estimator.

    output["B1"] = v6_selector["B1"]
    output["B2"] = median([v6, stacking_bracketed, b2_b4_boosted, floor_anchor_50, per_side_median]["B2"])
    output["B3"] = v6_selector["B3"]
    output["B4"] = v6_selector["B4"]

Ini adalah intervensi surgical — tidak mengubah kelas yang sudah benar
di v6, hanya memperbaiki B2 yang paling sering bermasalah.

Performa: Acc sama dengan v6_selector (96.49%) tapi MAE sedikit lebih baik
(0.2588 vs 0.2632), menunjukkan bahwa median B2 mengurangi magnitude error
meskipun tidak selalu mengubah biner pass/fail.

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
    Hitung count unik per kelas: v6 untuk semua kelas, median untuk B2.

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

    all_preds = {
        "v6": v6_predict(detections, params),
        "stacking": stacking_pred(detections),
        "b2b4": b2b4_pred(detections),
        "floor": floor_pred(detections),
        "median": median_pred(detections),
    }

    out = dict(all_preds["v6"])
    out["B2"] = int(round(float(np.median([p["B2"] for p in all_preds.values()]))))
    return out
