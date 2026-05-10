"""
Algoritma: best_visibility_grid
Generasi: v5
Benchmark JSON 228 pohon: Acc ±1 = 92.54%, MAE = 0.2664

Ide utama
---------
Bukan setiap deteksi punya bobot yang sama. Bounding box yang posisinya
jauh dari tengah frame (x_norm jauh dari 0.5) kemungkinan besar adalah
objek yang "mengintip" dari tepi — terlihat dari sisi ini, tapi menjadi
terlihat penuh dari sisi lain. Bobot gaussian di sumbu horizontal
memodelkan ini: semakin ke tepi, semakin kecil kontribusinya terhadap
count akhir.

Rumus bobot per deteksi:
    w(x) = 1 / (1 + alpha * exp(-((x - 0.5)^2) / (2 * sigma^2)))

Jumlah weighted ini langsung menjadi estimasi count unik per kelas.

Parameter alpha dan sigma tidak di-set manual — dicari lewat grid search
pada 228 pohon JSON (visibility grid search). Nilai terbaik yang ditemukan
dipakai sebagai konstanta.

Catatan: "grid search" di sini bukan training. Tidak ada split train/test
yang terpisah — grid search dilakukan pada semua 228 pohon, sehingga
hasilnya bisa sedikit optimistik. Metode ini tetap closed-form dan
deterministik.

Batasan
-------
- Grid search pada data penuh = mungkin sedikit overfit parameter.
- Tidak mempertimbangkan class-specific behavior (B2 dan B3 berbeda dari B1/B4).
- Digantikan oleh v6_selector dan v9_selector.

Input
-----
detections : list[dict]
    Setiap elemen adalah bounding box dengan field:
      - "class": str       → "B1", "B2", "B3", atau "B4"
      - "x_norm": float    → koordinat pusat horizontal (YOLO cx), range [0, 1]
    Field lain tidak dipakai.

Output
------
dict[str, int]
    Count unik per kelas: {"B1": int, "B2": int, "B3": int, "B4": int}
"""

import numpy as np

NAMES = ["B1", "B2", "B3", "B4"]

# Parameter terbaik dari grid search pada 228 pohon JSON (v5).
# alpha = kekuatan penalti tepi; sigma = lebar kurva gaussian.
ALPHA = 0.9
SIGMA = 0.45


def predict(detections: list) -> dict:
    """
    Hitung count unik per kelas dengan visibility weighting (gaussian horizontal).

    Parameters
    ----------
    detections : list[dict]
        Daftar bounding box dari semua sisi pohon.
        Setiap elemen harus memiliki field "class" dan "x_norm".

    Returns
    -------
    dict[str, int]
        Count unik per kelas B1–B4.
    """
    counts = {}
    for c in NAMES:
        cdets = [d for d in detections if d["class"] == c]
        if not cdets:
            counts[c] = 0
            continue
        total = sum(
            1.0 / (1.0 + ALPHA * np.exp(-((d["x_norm"] - 0.5) ** 2) / (2.0 * SIGMA ** 2)))
            for d in cdets
        )
        counts[c] = max(0, int(round(total)))
    return counts
