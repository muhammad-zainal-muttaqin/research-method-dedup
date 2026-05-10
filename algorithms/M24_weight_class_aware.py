"""
Algoritma: class_aware_vis  (v5_class_aware_vis)
Generasi: v5
Benchmark JSON 228 pohon: Acc ±1 = 85.09%, MAE = 0.3805

Ide utama
---------
Modifikasi dari visibility weighting dengan parameter berbeda per kelompok
kelas. B2 dan B3 secara visual lebih ambigu dan cenderung muncul di
posisi tengah frame, sehingga mereka mendapat alpha dan sigma yang berbeda
dari B1 dan B4.

    w(x, c) = 1 / (1 + alpha_c * exp(-((x - 0.5)^2) / (2 * sigma_c^2)))

Dua set parameter:
    - B1, B4: alpha=1.0, sigma=0.35  (penalti tepi lebih ketat)
    - B2, B3: alpha=0.65, sigma=0.45 (penalti tepi lebih longgar)

Nilai-nilai ini diperoleh dari grid search pada 228 pohon JSON (v5).

Dalam konteks v6_selector, metode ini dipakai sebagai salah satu opsi
override untuk pohon B2-heavy dengan B4 dup-rate rendah.

Secara global performa hanya 85.09% karena parameter B2/B3 terlalu longgar
untuk sebagian pohon. Sebagai komponen selector (v6/v9) ia berguna karena
dipilih hanya untuk pohon yang cocok.

Input
-----
detections : list[dict]
    - "class": str
    - "x_norm": float  (cx YOLO)

Output
------
dict[str, int]
"""

import numpy as np

NAMES = ["B1", "B2", "B3", "B4"]

ALPHA_B1B4 = 1.0
ALPHA_B2B3 = 0.65
SIGMA_B1B4 = 0.35
SIGMA_B2B3 = 0.45


def predict(detections: list) -> dict:
    """
    Hitung count unik per kelas dengan class-aware visibility weighting.

    Parameters
    ----------
    detections : list[dict]
        Field wajib: "class", "x_norm".

    Returns
    -------
    dict[str, int]
    """
    counts = {}
    for c in NAMES:
        cd = [d for d in detections if d["class"] == c]
        if not cd:
            counts[c] = 0
            continue
        alpha = ALPHA_B2B3 if c in ("B2", "B3") else ALPHA_B1B4
        sigma = SIGMA_B2B3 if c in ("B2", "B3") else SIGMA_B1B4
        total = sum(
            1.0 / (1.0 + alpha * np.exp(-((d["x_norm"] - 0.5) ** 2) / (2.0 * sigma ** 2)))
            for d in cd
        )
        counts[c] = max(0, int(round(total)))
    return counts
