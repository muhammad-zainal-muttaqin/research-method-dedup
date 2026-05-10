# Iterasi 6 — Analisis Disagreement Antar Metode

Tanggal: 10 Mei 2026.
Skrip: `iter6_disagreement.py`. Hasil: `iter6_disagreement.csv`, `iter6_recoverable.csv`.

## Tujuan

Tentukan apakah plateau 86,15% adalah ceiling struktural dataset atau ceiling per-metode (yang dapat dipecah dengan ensemble routing).

## Metode yang Diuji (7 estimator)

| Metode | Acc±1 | n_pass |
|---|---:|---:|
| `geometric_mean_blend` (winner) | 86,15% | 821 |
| `floor_clamped_hybrid` | 86,04% | 820 |
| `visibility` | 85,94% | 819 |
| `side_coverage` | 85,94% | 819 |
| `median3_floor` | 85,94% | 819 |
| `density_scaled_vis` | 85,94% | 819 |
| `adaptive_corrected` | 82,58% | 787 |

## Distribusi n_methods_pass per Pohon

| n metode benar | n pohon | % | Interpretasi |
|---:|---:|---:|---|
| 7 (semua) | 752 | 78,91% | Easy — semua metode setuju |
| 6 | 63 | 6,61% | Mostly easy |
| 4 | 4 | 0,42% | Disagreement |
| 3 | 5 | 0,52% | Strong disagreement |
| 2 | 1 | 0,10% | Strong disagreement |
| 1 | 29 | 3,04% | Recoverable via single specialist |
| 0 | 99 | 10,39% | **Structural hard** |

## Temuan Utama

### Oracle ceiling = 89,61%
Jika selector sempurna mampu memilih metode yang benar untuk setiap pohon: 854/953 = 89,61% Acc±1. **Peluang headroom 3,46pp** (86,15 → 89,61).

### 99 pohon adalah struktural keras
Tidak satu pun dari 7 metode benar — kemungkinan besar B2↔B3 ambiguity yang tidak dapat dipecahkan tanpa cross-view embedding (dilarang oleh constraint). Ini adalah **lower bound iredusibel** dataset.

### `adaptive_corrected` adalah specialist tersembunyi
Dari 132 kegagalan winner, **29 di antaranya benar oleh `adaptive_corrected`** (87,9% dari pool recoverable). Padahal `adaptive_corrected` overall hanya 82,58% — **lebih buruk 3,57pp dari winner**. Artinya `adaptive_corrected` menang/kalah pada subset berbeda.

### Metode lain merecover sedikit
- `visibility`, `side_coverage`, `median3_floor`, `density_scaled_vis`: masing-masing 4 recovery.
- `floor_clamped_hybrid`: 0 recovery (sangat dekat dengan winner).

### 102 pohon disagreement zone (1 ≤ n_pass ≤ 6)
Subset di mana selector dapat memberikan dampak nyata.

## Rencana Iter7

Bangun **selector deterministik** yang memilih antara `geometric_mean_blend` (default winner) dan `adaptive_corrected` (specialist) berdasarkan fitur pohon:

- `n_dets` (kepadatan)
- `n_sides_active` (sebaran sisi)
- `naive_count` per kelas
- `max_per_side` per kelas

Selector harus:
1. **Deterministik** — sama input sama output, tidak ada training/learning.
2. **Berbasis fitur tree-level**, bukan tree_id memorisasi.
3. **Tervalidasi held-out** — kondisi selector dirumuskan pada train, dievaluasi pada val + test.
4. **Tidak merugikan winner pada split mana pun** — gate yang sama dengan iter4.

Headroom realistik: jika selector dapat menyelamatkan separuh dari 33 recoverable (≈16 pohon), Acc±1 naik ke 87,7%. Lebih realistis: 5–10 recovery, Acc±1 naik ke 86,7–87,2%.

## Catatan Kejujuran (RULES.txt)

- 99 pohon struktural keras adalah lantai. Tidak ada heuristik dapat menyentuhnya tanpa melanggar batasan riset.
- 33 trees recoverable BUKAN jaminan +33 pohon perbaikan — selector akan kehilangan sebagian winners pada subset di mana adaptive lebih buruk.
- Selector berbasis fitur dapat overfit. Iter7 harus uji kondisi sederhana (1–2 fitur, ambang cukup tegas), bukan rule kompleks per-pohon.
- **Honest range yang dapat dijanjikan**: peningkatan 0,5–1,5pp di Acc±1 jika selector well-designed; nol jika sinyal fitur tidak konklusif.
