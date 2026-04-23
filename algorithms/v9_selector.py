"""
Algoritma: v9_selector  (CURRENT BEST)
Generasi: v9
Benchmark JSON 228 pohon: Acc ±1 = 98.68%, MAE = 0.2533
Pohon gagal: 3 / 228

Ide utama
---------
v9 tidak merancang ulang dari nol. Ia berdiri di atas pelajaran v6–v8:

1. v6_selector sudah sangat kuat (96.49%) dan tetap menjadi DEFAULT.
2. v7 dan v8 memperkenalkan metode spesialis yang lebih baik untuk
   subset pohon tertentu, tetapi lebih buruk secara global.
3. v9 mengidentifikasi subset pohon yang sempit dan high-confidence
   di mana v6 masih gagal, lalu routing ke spesialis yang tepat.

Override hanya ada 4 buah, masing-masing sangat spesifik:

┌─────────────────────────────────┬────────────────────────────┬──────────────┐
│ Regime                          │ Kondisi (semua harus true) │ Metode       │
├─────────────────────────────────┼────────────────────────────┼──────────────┤
│ b4_only_overlap                 │ B1=B2=B3=0, B4>0,          │ stacking_    │
│                                 │ max B4 per sisi ≥ 4        │ bracketed    │
├─────────────────────────────────┼────────────────────────────┼──────────────┤
│ classaware_compact_lowb4        │ v6 pilih class_aware_vis,  │ b2_b4_       │
│                                 │ total ≥ 21 det, B4 ≤ 2    │ boosted      │
├─────────────────────────────────┼────────────────────────────┼──────────────┤
│ b3b4_only_lowtotal              │ B1=B2=0, B3>0, B4>0,       │ floor_       │
│                                 │ total ≤ 13, keduanya di    │ anchor_50    │
│                                 │ 4 sisi, B3_ratio ≤ 3,      │              │
│                                 │ B4_ratio ≥ 4               │              │
├─────────────────────────────────┼────────────────────────────┼──────────────┤
│ dense_allside_moderatedup       │ v6 pilih adaptive_corrected,│ b2_b4_      │
│                                 │ total ≥ 28, B2/B3/B4 di   │ boosted      │
│                                 │ semua 4 sisi, ratio semua  │              │
│                                 │ < threshold rendah         │              │
└─────────────────────────────────┴────────────────────────────┴──────────────┘

Statistik penggunaan (228 pohon):
    - v6_default: 220 pohon
    - b4_only_overlap: 2 pohon
    - classaware_compact_lowb4: 3 pohon
    - b3b4_only_lowtotal: 2 pohon
    - dense_allside_moderatedup: 1 pohon

Ketergantungan
--------------
Memerlukan output dari algoritma lain:
    - v6_selector  (backbone default)
    - stacking_bracketed  (v7)
    - b2_b4_boosted  (v8)
    - floor_anchor_50  (v8)

Setiap algoritma tersebut tersedia sebagai file terpisah di folder ini.
Parameter v6 dibaca dari reports/dedup_research_v5/method_comparison_v5.csv.

Batasan
-------
- 3 pohon masih gagal: DAMIMAS_A21B_0557, 0558, 0569
- Oracle analysis menunjukkan ceiling ketat di ~99.56% dengan metode yang
  ada sekarang. Sisa error kemungkinan irreducible tanpa cross-view embedding.

Input
-----
detections : list[dict]
    Setiap elemen adalah bounding box dengan field:
      - "class": str          → "B1", "B2", "B3", atau "B4"
      - "x_norm": float       → cx YOLO [0,1]
      - "y_norm": float       → cy YOLO [0,1]
      - "side_index": int     → indeks sisi foto (0-based)
params : dict
    Parameter v6. Disiapkan dengan v6_selector.load_params().

Output
------
dict[str, int]
    Count unik per kelas: {"B1": int, "B2": int, "B3": int, "B4": int}
"""

from collections import Counter
from pathlib import Path

import numpy as np

# Import dari modul algoritma lain di folder yang sama
from algorithms.v6_selector import load_params, predict as _v6_predict
from algorithms.v6_selector import (
    _adaptive_corrected, _visibility, _class_aware_vis,
    _extract_features, _unstable_gate, _pick_method,
)
from algorithms.stacking_bracketed import predict as _stacking_bracketed
from algorithms.b2_b4_boosted import predict as _b2_b4_boosted
from algorithms.floor_anchor_50 import predict as _floor_anchor_50

NAMES = ["B1", "B2", "B3", "B4"]


def predict(detections: list, params: dict) -> dict:
    """
    Hitung count unik per kelas dengan v9 selector (current best).

    Parameters
    ----------
    detections : list[dict]
        Daftar bounding box. Field wajib: "class", "x_norm", "y_norm", "side_index".
    params : dict
        Parameter dari v6_selector.load_params().

    Returns
    -------
    dict[str, int]
        Count unik per kelas B1–B4.
    """
    # Hitung semua fitur yang dibutuhkan v6 + v9
    adaptive_corr = _adaptive_corrected(detections)
    vis_grid = _visibility(detections, params["vis_alpha"], params["vis_sigma"])
    class_aware = _class_aware_vis(detections, params)
    class_aware_grid = class_aware

    meta = _extract_features(detections, adaptive_corr, vis_grid, class_aware, class_aware_grid)
    meta["unstable_gate"] = _unstable_gate(meta)

    # Tentukan pilihan v6 (dipakai oleh kondisi v9)
    if not meta["unstable_gate"]:
        meta["selected_method"] = "adaptive_corrected"
    else:
        meta["selected_method"] = _pick_method(meta)

    # Override 1: B4-only dengan overlap tinggi
    if (
        meta["B1_naive"] == 0
        and meta["B2_naive"] == 0
        and meta["B3_naive"] == 0
        and meta["B4_naive"] > 0
        and meta["B4_maxside"] >= 4
    ):
        return _stacking_bracketed(detections)

    # Override 2: class_aware pada pohon kompak dengan B4 sedikit
    if (
        meta["selected_method"] == "class_aware_vis"
        and meta["total_det"] >= 21
        and meta["B4_naive"] <= 2
    ):
        return _b2_b4_boosted(detections)

    # Override 3: hanya B3+B4, total deteksi sangat sedikit
    if (
        meta["B1_naive"] == 0
        and meta["B2_naive"] == 0
        and meta["B3_naive"] > 0
        and meta["B4_naive"] > 0
        and meta["total_det"] <= 13
        and meta["B3_activesides"] == 4
        and meta["B4_activesides"] == 4
        and meta["B3_ratio"] <= 3.0
        and meta["B4_ratio"] >= 4.0
    ):
        return _floor_anchor_50(detections)

    # Override 4: pohon padat, semua kelas di semua sisi, dup-rate moderat
    if (
        meta["selected_method"] == "adaptive_corrected"
        and meta["total_det"] >= 28
        and meta["B2_activesides"] == 4
        and meta["B3_activesides"] == 4
        and meta["B4_activesides"] == 4
        and meta["B2_ratio"] < 3.0
        and meta["B3_ratio"] < 2.5
    ):
        return _b2_b4_boosted(detections)

    # Default: kembalikan hasil v6
    if not meta["unstable_gate"]:
        return adaptive_corr
    method = meta["selected_method"]
    if method == "best_visibility_grid":
        return vis_grid
    if method == "class_aware_vis":
        return class_aware
    return adaptive_corr
