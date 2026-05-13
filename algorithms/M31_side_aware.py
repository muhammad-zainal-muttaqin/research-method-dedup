"""
M31_side_aware_selector
=======================
Acc±1 = 89.30% pada 953-tree Brand-New-Dataset-YOLO (full canonical GT).
Basis dari M60_blind_strict — komponen terpercaya, tidak overfit.

Tidak butuh file CSV, tidak import file lain.
Semua nilai divisor sudah dihardcode dari hasil kalkulasi train-split saja.

Cara pakai:
    from algorithms.M31_side_aware import predict
    result = predict(detections)   # {"B1": int, "B2": int, "B3": int, "B4": int}

Format detections: list[dict], tiap dict punya:
    "class"      : str   → "B1", "B2", "B3", atau "B4"
    "x_norm"     : float → koordinat pusat horizontal YOLO, range [0, 1]
    "y_norm"     : float → koordinat pusat vertikal YOLO, range [0, 1]
    "side_index" : int   → indeks sisi foto (0-based)
"""

from collections import Counter
import numpy as np

NAMES = ("B1", "B2", "B3", "B4")

# =============================================================================
# DIVISOR TABLE — 2D median (n_sides, class)
# =============================================================================
#
# Sumber: exp_12 may 2026/out/divisor_2d.csv
# Dihitung oleh step04_side_factor.py dari 599-pohon TRAIN split saja.
#
# Artinya: rata-rata 1 bunch yang sama terlihat sebanyak N kali lintas sisi.
# Contoh: pohon 4-sisi, kelas B3 → divisor 1.857 → tiap bunch B3 rata-rata
# terlihat 1.857× dari sisi yang berbeda.
#
# Aturan fallback untuk sel yang kurang data (support < 20 pohon train):
#   - B4 selalu pakai median ns=4 (1.600) — B4 paling unik, duplikasi rendah
#   - B1/B2/B3 di ns bukan 4: median_ns4 × max(1.0, ns/4)
#   - Logika ini sudah dikalkulasi, hasilnya langsung di dict ini.
#
# Baris dari tabel asli (hanya yang support >= 20):
#   ns=4  B1 median=2.000  count=333
#   ns=4  B2 median=2.000  count=421
#   ns=4  B3 median=1.857  count=544
#   ns=4  B4 median=1.600  count=428
#   ns=8  B2 median=3.875  count=22
#   ns=8  B3 median=3.143  count=23
#   (ns=8 B1 count=17 dan B4 count=15 → di bawah threshold, pakai fallback)
#
# Hasil setelah fallback dikalkulasi:
_DIVISOR = {
    # ns  : { B1    , B2    , B3    , B4    }
    1     : {"B1": 2.000, "B2": 2.000, "B3": 1.857, "B4": 1.600},  # ns<4, max(1, ns/4)=1 → sama dgn ns=4
    2     : {"B1": 2.000, "B2": 2.000, "B3": 1.857, "B4": 1.600},  # sama
    3     : {"B1": 2.000, "B2": 2.000, "B3": 1.857, "B4": 1.600},  # sama
    4     : {"B1": 2.000, "B2": 2.000, "B3": 1.857, "B4": 1.600},  # dari tabel langsung
    5     : {"B1": 2.500, "B2": 2.500, "B3": 2.321, "B4": 1.600},  # 4-side × (5/4)=1.25
    6     : {"B1": 3.000, "B2": 3.000, "B3": 2.786, "B4": 1.600},  # 4-side × (6/4)=1.50
    7     : {"B1": 3.500, "B2": 3.500, "B3": 3.250, "B4": 1.600},  # 4-side × (7/4)=1.75
    8     : {"B1": 4.000, "B2": 3.875, "B3": 3.143, "B4": 1.600},  # B2/B3 tabel, B1 fallback ×2, B4 tetap
}

def _get_divisor(ns: int, cl: str) -> float:
    """Ambil divisor dari tabel; kalau ns > 8 gunakan interpolasi linear dari ns=4."""
    if ns in _DIVISOR:
        return _DIVISOR[ns][cl]
    # ns > 8: sangat jarang, tapi tangani gracefully
    if cl == "B4":
        return 1.600
    return _DIVISOR[4][cl] * (ns / 4.0)


# =============================================================================
# UTILITAS DASAR
# =============================================================================

def _naive(dets: list) -> dict:
    """Hitung mentah per kelas (belum didedup)."""
    c = Counter(d["class"] for d in dets)
    return {cl: int(c.get(cl, 0)) for cl in NAMES}


def _n_sides(dets: list) -> int:
    """Jumlah sisi unik yang ada di deteksi pohon ini."""
    return len({d["side_index"] for d in dets}) if dets else 0


def _max_per_side(dets: list, cl: str) -> int:
    """
    Batas bawah fisik: satu sisi saja sudah melihat sebanyak ini.
    Prediksi tidak boleh lebih rendah dari angka ini.
    """
    cd = [d for d in dets if d["class"] == cl]
    if not cd:
        return 0
    return int(max(Counter(d["side_index"] for d in cd).values()))


# =============================================================================
# ESTIMATOR 1 — side_aware_divide  (dipakai untuk ns >= 5)
# =============================================================================
#
# Formula: unique[c] = round(naive[c] / divisor[ns][c])
# Floor   : max_per_side — tidak mungkin lebih sedikit dari yang 1 sisi lihat.
#
# Ini adalah fix utama untuk pohon 8-sisi. M01 lama pakai dup_rate max 1.45,
# padahal empiris 8-sisi butuh ~4× → M01 overcounting 2× pada pohon dense.

def _side_aware_divide(dets: list) -> dict:
    ns    = _n_sides(dets)
    naive = _naive(dets)
    out   = {}
    for c in NAMES:
        divisor = _get_divisor(ns, c)
        est     = int(round(naive[c] / divisor)) if divisor > 0 else naive[c]
        out[c]  = max(est, _max_per_side(dets, c))
    return out


# =============================================================================
# ESTIMATOR 2 — adaptive_corrected  (dipakai untuk pohon B1-berat)
# =============================================================================
#
# Formula:
#   dup_rate = clip(2.05 - 0.014 × n_total, 1.45, 2.10)
#   scale    = dup_rate / 1.79                         ← relatif terhadap rata-rata
#   unique[c] = round(naive[c] / (BASE[c] × scale))
#
# Logika: pohon padat (banyak bbox) cenderung saling menghalangi antar sisi,
# sehingga dup_rate lebih rendah. Sebaliknya pohon jarang dup_rate lebih tinggi.
#
# BASE_FACTORS = median rasio naive/GT per kelas dari 228-pohon dev set.

_BASE = {"B1": 1.986, "B2": 1.786, "B3": 1.795, "B4": 1.655}

def _adaptive_corrected(dets: list) -> dict:
    n_total  = len(dets)
    dup_rate = float(np.clip(2.05 - 0.014 * n_total, 1.45, 2.10))
    scale    = dup_rate / 1.79
    naive    = _naive(dets)
    return {c: max(0, int(round(naive[c] / (_BASE[c] * scale)))) for c in NAMES}


# =============================================================================
# ESTIMATOR 3 — visibility_count  (komponen blending)
# =============================================================================
#
# Ide: bbox yang posisi x-nya di tengah gambar (x_norm ≈ 0.5) lebih "visible"
# karena tidak terpotong tepi. Tiap bbox dikontribusikan dengan bobot:
#
#   w = 1 / (1 + alpha × exp(-(x_norm - 0.5)² / (2σ²)))
#
# Bobot tinggi (≈1.0) kalau x_norm ≈ 0.5 (tengah).
# Bobot rendah (≈0.5) kalau x_norm dekat 0 atau 1 (tepi terpotong).
# unique[c] ≈ sum bobot semua bbox kelas c.

def _visibility_count(dets: list, alpha: float = 1.0, sigma: float = 0.3) -> dict:
    out = {}
    for c in NAMES:
        cd = [d for d in dets if d["class"] == c]
        if not cd:
            out[c] = 0
            continue
        total = sum(
            1.0 / (1.0 + alpha * np.exp(-((d["x_norm"] - 0.5) ** 2) / (2.0 * sigma ** 2)))
            for d in cd
        )
        out[c] = max(0, int(round(total)))
    return out


# =============================================================================
# ESTIMATOR 4 — side_coverage  (komponen blending)
# =============================================================================
#
# Ambil visibility_count lalu clamp ke rentang fisik yang valid:
#   floor   = max_per_side (tidak mungkin lebih sedikit)
#   ceiling = naive count  (tidak mungkin lebih banyak dari total yang terdeteksi)

def _side_coverage(dets: list) -> dict:
    vis   = _visibility_count(dets)
    naive = _naive(dets)
    out   = {}
    for c in NAMES:
        cd = [d for d in dets if d["class"] == c]
        if not cd:
            out[c] = 0
            continue
        floor   = _max_per_side(dets, c)
        ceiling = naive[c]
        out[c]  = min(max(vis[c], floor), ceiling)
    return out


# =============================================================================
# ESTIMATOR 5 — geometric_mean_blend  (default catch-all)
# =============================================================================
#
# Blend dua estimator: visibility dan adaptive_corrected.
# Pakai rata-rata geometri (sqrt(a × b)) karena keduanya cenderung over- dan
# under-estimate di kasus yang berbeda — rata-rata geometri lebih konservatif
# dari rata-rata aritmetik.
#
# Kalau salah satu = 0: pakai rata-rata aritmetik (hindari sqrt(0) = 0 palsu).
# Floor: max_per_side di akhir.

def _geometric_mean_blend(dets: list) -> dict:
    vis    = _visibility_count(dets)
    adap   = _adaptive_corrected(dets)
    out    = {}
    for c in NAMES:
        v, a = vis[c], adap[c]
        if v == 0 or a == 0:
            out[c] = (v + a) // 2
        else:
            out[c] = int(round(np.sqrt(v * a)))
    return {c: max(out[c], _max_per_side(dets, c)) for c in NAMES}


# =============================================================================
# ESTIMATOR 6 — median3_floor  (untuk pohon B3-dominan + padat)
# =============================================================================
#
# Ambil median dari tiga estimator: visibility, adaptive_corrected, side_coverage.
# Median lebih robust dari blend — outlier satu estimator tidak mendominasi.
# Floor: max_per_side di akhir.

def _median3_floor(dets: list) -> dict:
    vis  = _visibility_count(dets)
    adap = _adaptive_corrected(dets)
    cov  = _side_coverage(dets)
    out  = {c: sorted([vis[c], adap[c], cov[c]])[1] for c in NAMES}
    return {c: max(out[c], _max_per_side(dets, c)) for c in NAMES}


# =============================================================================
# M31 — SELECTOR UTAMA
# =============================================================================
#
# Trifurkasi berdasarkan profil pohon:
#
#   JALUR A (ns >= 5, pohon 8-sisi):
#       → side_aware_divide
#         Satu-satunya estimator yang dikalibrasi untuk multiplisitas tinggi.
#         M01 lama gagal 100% di bucket >40 dets karena dup_rate clamp 1.45 << 4.
#
#   JALUR B (ns <= 4, B3-dominan + padat: b3frac >= 0.60 DAN n_total >= 25):
#       → median3_floor
#         Pohon dengan banyak B3 (matang sedang). Dup-rate B3 lebih tinggi dari
#         B1/B4 di kondisi padat. Median 3 estimator lebih stabil dari blend tunggal.
#
#   JALUR C (ns <= 4, B1-berat: B1 >= 3 DAN b3frac < 0.45 DAN B4 < 10):
#       → adaptive_corrected
#         Pohon muda dengan banyak B1 (belum matang). Pola dup-rate lebih
#         seragam lintas kelas → global divisor + density correction cukup.
#
#   JALUR D (ns <= 4, semua kasus lain):
#       → geometric_mean_blend
#         Default. Blend geometri paling stabil untuk pohon campuran tanpa
#         karakter ekstrem.

def predict(detections: list) -> dict:
    """
    Hitung jumlah bunch unik per kelas untuk satu pohon.

    Parameters
    ----------
    detections : list[dict]
        Semua bbox dari semua sisi pohon. Tiap dict wajib punya:
        "class" (B1-B4), "x_norm", "y_norm", "side_index".

    Returns
    -------
    dict[str, int]
        {"B1": int, "B2": int, "B3": int, "B4": int}
    """
    if not detections:
        return {c: 0 for c in NAMES}

    ns      = _n_sides(detections)
    n_total = len(detections)
    naive   = _naive(detections)
    b3frac  = naive["B3"] / max(n_total, 1)

    # ── JALUR A: pohon 8-sisi (atau 5-7 dari capture parsial) ─────────────────
    if ns >= 5:
        return _side_aware_divide(detections)

    # ── JALUR B: B3-dominan + padat ───────────────────────────────────────────
    if b3frac >= 0.60 and n_total >= 25:
        return _median3_floor(detections)

    # ── JALUR C: B1-berat, B4 langka ──────────────────────────────────────────
    if naive["B1"] >= 3 and b3frac < 0.45 and naive["B4"] < 10:
        return _adaptive_corrected(detections)

    # ── JALUR D: default ───────────────────────────────────────────────────────
    return _geometric_mean_blend(detections)
