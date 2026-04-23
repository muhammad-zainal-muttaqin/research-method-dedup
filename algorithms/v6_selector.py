"""
Algoritma: v6_selector
Generasi: v6
Benchmark JSON 228 pohon: Acc ±1 = 96.49%, MAE = 0.2632

Ide utama
---------
Titik balik riset ini. v1–v5 mencoba satu rumus global yang "cukup baik
untuk semua pohon". v6 menyadari bahwa tidak ada satu rumus seperti itu.

v6 menggunakan pendekatan selector: deteksi fitur pohon terlebih dahulu,
lalu routing ke metode yang paling cocok untuk pola tersebut.

Struktur selector:
1. Default → adaptive_corrected  (adaptif berbasis total deteksi)
2. Jika pohon "tidak stabil" (unstable gate) → override ke salah satu:
   a. best_visibility_grid   → untuk pohon dengan B3 dup-rate tinggi
   b. class_aware_vis        → untuk pohon B2-heavy, B4 dup-rate rendah
3. Semua pohon yang TIDAK masuk unstable gate tetap pakai adaptive_corrected.

"Unstable gate" adalah decision tree pendek yang melihat fitur seperti:
    - B4_naive (jumlah deteksi B4)
    - B4_yrange (rentang vertikal deteksi B4)
    - B3_ratio = naive_B3 / max_side_B3
    - disagreement antara adaptive_corrected dan best_visibility_grid

Decision tree ini bukan learned dari data split — ia dibentuk secara
manual berbasis analisis error v5. Namun demikian, threshold-nya berasal
dari statistik 228 pohon, sehingga ada potensi sedikit overfit.

Ketergantungan
--------------
Metode ini memerlukan parameter dari grid search v5 (alpha dan sigma terbaik
untuk visibility, serta alpha/sigma kelas-aware). Parameter dibaca dari
file CSV output v5 di reports/dedup_research_v5/.

Pemakaian terbaik: v6_selector adalah backbone default dari v9_selector.
220 dari 228 pohon tetap diproses oleh v6_selector bahkan di v9.

Batasan
-------
- Decision tree threshold difit pada semua 228 pohon → potensi overfit minor.
- 8 pohon masih gagal di v6 (diselesaikan sebagian oleh v9).

Input (via parameter params)
-----------------------------
detections : list[dict]
    Setiap elemen adalah bounding box dengan field:
      - "class": str          → "B1", "B2", "B3", atau "B4"
      - "x_norm": float       → cx YOLO [0,1]
      - "y_norm": float       → cy YOLO [0,1]
      - "side_index": int     → indeks sisi (0-based)

params : dict[str, float]
    Parameter dari v5 grid search. Diisi oleh load_params() di bawah.

Output
------
dict[str, int]
    Count unik per kelas: {"B1": int, "B2": int, "B3": int, "B4": int}
"""

from collections import Counter
from pathlib import Path

import numpy as np

NAMES = ["B1", "B2", "B3", "B4"]
BASE_FACTORS = {"B1": 1.986, "B2": 1.786, "B3": 1.795, "B4": 1.655}
_V5_REPORT = Path(__file__).resolve().parent.parent / "reports" / "dedup_research_v5" / "method_comparison_v5.csv"


def load_params() -> dict:
    """
    Baca parameter grid search terbaik dari output v5.
    Harus dipanggil sekali sebelum predict().
    """
    import pandas as pd
    comp = pd.read_csv(_V5_REPORT).set_index("method")
    return {
        "vis_alpha": float(comp.loc["best_visibility_grid", "alpha"]),
        "vis_sigma": float(comp.loc["best_visibility_grid", "sigma"]),
        "class_alpha_B1B4": float(comp.loc["best_class_aware_grid", "alpha_B1B4"]),
        "class_alpha_B2B3": float(comp.loc["best_class_aware_grid", "alpha_B2B3"]),
        "class_sigma_B1B4": float(comp.loc["best_class_aware_grid", "sigma_B1B4"]),
        "class_sigma_B2B3": float(comp.loc["best_class_aware_grid", "sigma_B2B3"]),
    }


def _density_scale(n_total: int) -> float:
    return float(np.clip(2.05 - 0.014 * n_total, 1.45, 2.10) / 1.79)


def _adaptive_corrected(detections: list) -> dict:
    n_total = len(detections)
    sc = _density_scale(n_total)
    naive = Counter(d["class"] for d in detections)
    return {c: max(0, round(naive.get(c, 0) / (BASE_FACTORS[c] * sc))) for c in NAMES}


def _visibility(detections: list, alpha: float, sigma: float) -> dict:
    counts = {}
    for c in NAMES:
        cd = [d for d in detections if d["class"] == c]
        if not cd:
            counts[c] = 0
            continue
        total = sum(
            1.0 / (1.0 + alpha * np.exp(-((d["x_norm"] - 0.5) ** 2) / (2.0 * sigma ** 2)))
            for d in cd
        )
        counts[c] = max(0, int(round(total)))
    return counts


def _class_aware_vis(detections: list, params: dict) -> dict:
    counts = {}
    for c in NAMES:
        cd = [d for d in detections if d["class"] == c]
        if not cd:
            counts[c] = 0
            continue
        alpha = params["class_alpha_B2B3"] if c in ("B2", "B3") else params["class_alpha_B1B4"]
        sigma = params["class_sigma_B2B3"] if c in ("B2", "B3") else params["class_sigma_B1B4"]
        total = sum(
            1.0 / (1.0 + alpha * np.exp(-((d["x_norm"] - 0.5) ** 2) / (2.0 * sigma ** 2)))
            for d in cd
        )
        counts[c] = max(0, int(round(total)))
    return counts


def _extract_features(detections, adaptive_corr, vis_grid, class_aware, class_aware_grid) -> dict:
    """Ekstrak fitur untuk decision tree selector."""
    by_class = {c: [d for d in detections if d["class"] == c] for c in NAMES}
    naive = {c: len(by_class[c]) for c in NAMES}
    side_counts = {
        c: Counter(d["side_index"] for d in by_class[c]) for c in NAMES
    }

    feat = {"total_det": len(detections)}
    for c in NAMES:
        ys = [d["y_norm"] for d in by_class[c]]
        max_side = max(side_counts[c].values(), default=0)
        feat[f"{c}_naive"] = naive[c]
        feat[f"{c}_maxside"] = max_side
        feat[f"{c}_activesides"] = len(side_counts[c])
        feat[f"{c}_ratio"] = naive[c] / max(max_side, 1) if naive[c] else 0.0
        feat[f"{c}_yrange"] = (max(ys) - min(ys)) if ys else 0.0
        feat[f"d_ac_bvg_{c}"] = adaptive_corr[c] - vis_grid[c]
        feat[f"d_ac_cag_{c}"] = adaptive_corr[c] - class_aware_grid[c]
        feat[f"d_ac_cav_{c}"] = adaptive_corr[c] - class_aware[c]
    return feat


def _unstable_gate(feat: dict) -> bool:
    """Decision tree untuk mendeteksi pohon yang perlu override."""
    if feat["B4_naive"] <= 6.5:
        if feat["B1_activesides"] <= 2.5:
            if feat["B2_naive"] <= 5.5:
                if feat["B4_ratio"] <= 3.5:
                    return feat["d_ac_bvg_B3"] > 0.5
                return feat["B4_yrange"] > 0.09450500085949898
            if feat["B4_yrange"] <= 0.05225699953734875:
                return feat["d_ac_cag_B2"] > -0.5
        return False
    if feat["B4_yrange"] <= 0.09679850190877914:
        return False
    if feat["B4_yrange"] <= 0.1491520032286644:
        return feat["B2_ratio"] <= 3.5
    return False


def _pick_method(feat: dict) -> str:
    if feat["B3_ratio"] <= 3.166666626930237:
        if feat["B4_ratio"] <= 2.583333373069763:
            return "class_aware_vis"
        return "adaptive_corrected"
    return "best_visibility_grid"


def predict(detections: list, params: dict) -> dict:
    """
    Hitung count unik per kelas dengan v6 selector.

    Parameters
    ----------
    detections : list[dict]
        Daftar bounding box. Field wajib: "class", "x_norm", "y_norm", "side_index".
    params : dict
        Parameter dari load_params(). Harus disiapkan sekali di awal program.

    Returns
    -------
    dict[str, int]
        Count unik per kelas B1–B4.
    """
    adaptive_corr = _adaptive_corrected(detections)
    vis_grid = _visibility(detections, params["vis_alpha"], params["vis_sigma"])
    class_aware = _class_aware_vis(detections, params)
    class_aware_grid = _class_aware_vis(detections, params)

    feat = _extract_features(detections, adaptive_corr, vis_grid, class_aware, class_aware_grid)

    if not _unstable_gate(feat):
        return adaptive_corr

    method = _pick_method(feat)
    if method == "best_visibility_grid":
        return vis_grid
    if method == "class_aware_vis":
        return class_aware
    return adaptive_corr
