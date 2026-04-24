# Multi-View Oil Palm Bunch Counting

Riset deduplikasi multi-view untuk menghitung jumlah tandan sawit unik per pohon dari 4–8 sisi foto. Fokus: **counting berbasis label ground truth JSON** dan **dedup algorithmic-only** — tanpa training model baru.

## Inti Masalah

Satu tandan yang sama bisa muncul di beberapa sisi foto. Tanpa dedup, tandan dihitung berulang kali. Naive sum overcount ~78.8%. Problem utama bukan sekadar menghitung bbox, tetapi **mengubah multi-view detection menjadi unique bunch count per kelas**.

## Dataset

| Item | Nilai |
|---|---:|
| Total pohon | 953 |
| DAMIMAS | 854 |
| LONSUM | 99 |
| Pohon dengan JSON | 228 |
| Pohon non-JSON | 725 |
| Sisi per pohon | 4 (mayoritas), 8 (±45 pohon terbaru) |

### Kelas Kematangan

| Class | Deskripsi |
|---|---|
| `B1` | merah, paling matang, posisi paling bawah |
| `B2` | transisi merah-hitam |
| `B3` | hitam penuh, di atas B2 |
| `B4` | paling kecil, berduri, paling atas |

Kelas bersifat ordinal B1→B4. Ambiguitas utama: **B2 ↔ B3** (irreducible, bukan label noise).

---

## Metrik yang Dilaporkan

Benchmark utama pada 228 pohon JSON. Sebuah pohon **pass** jika prediksi setiap kelas meleset maksimal ±1 dari GT.

| Metrik | Definisi |
|---|---|
| **Per class MAE** | rata-rata \|pred − GT\| per kelas (B1/B2/B3/B4), dirata-rata lintas pohon |
| **Macro class-MAE** | rata-rata dari 4 per-class MAE (bobot sama tiap kelas) |
| **Exact accuracy** | % pohon dengan prediksi **tepat sama** dengan GT untuk semua kelas |
| **Acc ±1** | % pohon dengan semua kelas dalam selisih ≤1 dari GT (primer) |
| **Total count MAE** | rata-rata \|total_pred − total_GT\| per pohon (jumlah B1+B2+B3+B4) |
| **Total ±1 accuracy** | % pohon dengan total count dalam ±1 dari GT total |
| **Per class mean error** | rata-rata (pred − GT) per kelas — mengukur bias arah (positif = overcount, negatif = undercount) |

---

## Hasil Terkini (2026-04-24)

Dataset: 228 pohon JSON. Pohon **pass** jika semua 4 kelas dalam ±1 dari GT.

> Catatan: `dedup_research_v9.py` melaporkan 98.68% (225/228) karena perbedaan minor data loading pipeline — logika selector sama.

### Akurasi Keseluruhan (Acc ±1 per kelas)

| Rank | Method | Gen | Acc ±1 | MAE | MTE | Gagal |
|---:|---|---|---:|---:|---:|---:|
| 1 | [`v9_selector`](algorithms/v9_selector.py) | v9 | **97.37%** | **0.2533** | 1.0132 | 6 |
| 2 | [`v9_b2_median_v6`](algorithms/b2_median_v6.py) | v9 | 96.05% | 0.2577 | 1.0307 | 9 |
| 3 | [`v6_selector`](algorithms/v6_selector.py) | v6 | 96.05% | 0.2599 | 1.0395 | 9 |
| 4 | [`v7_stacking_bracketed`](algorithms/stacking_bracketed.py) | v7 | 94.30% | 0.2643 | 1.0570 | 13 |
| 5 | [`v7_stacking_density`](algorithms/stacking_density.py) | v7 | 94.30% | 0.2708 | 1.0833 | 13 |
| 6 | [`v8_entropy_modulated`](algorithms/entropy_modulated.py) | v8 | 94.30% | 0.2763 | 1.1053 | 13 |
| 7 | [`v5_adaptive_corrected`](algorithms/adaptive_corrected.py) | v5 | 93.86% | 0.2774 | 1.1096 | 14 |
| 8 | [`v8_b2_b4_boosted`](algorithms/b2_b4_boosted.py) | v8 | 92.54% | 0.2632 | 1.0526 | 17 |
| 9 | `v2_visibility` | v2 | 92.54% | 0.2664 | 1.0658 | 17 |
| 10 | [`v5_best_visibility_grid`](algorithms/best_visibility_grid.py) | v5 | 92.54% | 0.2664 | 1.0658 | 17 |
| 11 | `v1_corrected` | v1 | 90.79% | 0.2851 | 1.1404 | 21 |

> MTE = Mean Total Error (jumlah absolut error semua kelas, rata-rata per pohon). Tie-break: MAE ascending.

### Akurasi Per Kelas (% pohon dalam ±1)

B1 termudah — 100% semua metode. B2 dan B3 bottleneck utama.

| Method | B1 | B2 | B3 | B4 |
|---|---:|---:|---:|---:|
| `v9_selector` | 100.0% | 98.7% | 99.1% | 99.6% |
| `v9_b2_median_v6` | 100.0% | 98.2% | 98.7% | 99.1% |
| `v6_selector` | 100.0% | 98.2% | 98.7% | 99.1% |
| `v7_stacking_bracketed` | 100.0% | 98.7% | 96.5% | 98.7% |
| `v7_stacking_density` | 100.0% | 98.7% | 96.5% | 98.7% |
| `v8_entropy_modulated` | 100.0% | 98.7% | 96.5% | 98.7% |
| `v5_adaptive_corrected` | 100.0% | 98.2% | 96.9% | 98.2% |
| `v8_b2_b4_boosted` | 100.0% | 97.8% | 96.5% | 97.8% |
| `v2_visibility` | 100.0% | 97.8% | 96.0% | 98.2% |
| `v5_best_visibility` | 100.0% | 97.8% | 96.0% | 98.2% |
| `v1_corrected` | 100.0% | 97.8% | 94.3% | 97.8% |

### Pola Error Per Kelas (jumlah pohon dengan error >±1)

↑ = overcount ekstrem (pred − GT > 1), ↓ = undercount ekstrem (pred − GT < −1)

| Method | B1↑ | B1↓ | B2↑ | B2↓ | B3↑ | B3↓ | B4↑ | B4↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `v9_selector` | 0 | 0 | 1 | 2 | 1 | 1 | 0 | 1 |
| `v9_b2_median_v6` | 0 | 0 | 0 | 4 | 2 | 1 | 1 | 1 |
| `v6_selector` | 0 | 0 | 2 | 2 | 2 | 1 | 1 | 1 |
| `v7_stacking_bracketed` | 0 | 0 | 1 | 2 | 6 | 2 | 2 | 1 |
| `v7_stacking_density` | 0 | 0 | 1 | 2 | 6 | 2 | 2 | 1 |
| `v8_entropy_modulated` | 0 | 0 | 1 | 2 | 6 | 2 | 2 | 1 |
| `v5_adaptive_corrected` | 0 | 0 | 2 | 2 | 6 | 1 | 3 | 1 |
| `v8_b2_b4_boosted` | 0 | 0 | 0 | 5 | 6 | 2 | 2 | 3 |
| `v2_visibility` | 0 | 0 | 0 | 5 | 2 | 7 | 0 | 4 |
| `v5_best_visibility` | 0 | 0 | 0 | 5 | 2 | 7 | 0 | 4 |
| `v1_corrected` | 0 | 0 | 0 | 5 | 7 | 6 | 4 | 1 |

### Kecepatan (ms/pohon, 30 repetisi × 228 pohon)

| Rank | Method | Mean ms | Median ms | Std ms | pohon/detik |
|---:|---|---:|---:|---:|---:|
| 1 | `v1_corrected` | 0.0036 | 0.0034 | 0.0005 | 279,830 |
| 2 | `v5_adaptive_corrected` | 0.0073 | 0.0073 | 0.0003 | 136,242 |
| 3 | `v7_stacking_density` | 0.0142 | 0.0138 | 0.0013 | 70,585 |
| 4 | `v2_visibility` | 0.0222 | 0.0218 | 0.0009 | 45,146 |
| 5 | `v5_best_visibility` | 0.0228 | 0.0227 | 0.0015 | 43,833 |
| 6 | `v8_b2_b4_boosted` | 0.0477 | 0.0480 | 0.0021 | 20,951 |
| 7 | `v7_stacking_bracketed` | 0.0480 | 0.0481 | 0.0018 | 20,830 |
| 8 | `v9_selector` | **0.0792** | 0.0794 | 0.0023 | **12,619** |
| 9 | `v6_selector` | 0.0993 | 0.0986 | 0.0035 | 10,074 |
| 10 | `v8_entropy_modulated` | 0.1046 | 0.1044 | 0.0023 | 9,558 |
| 11 | `v9_b2_median_v6` | 0.4291 | 0.4250 | 0.0140 | 2,330 |

### Robustness terhadap Noise Koordinat

Simulasi: Gaussian noise σ ke `x_norm`/`y_norm` tiap bbox. Drop@20% = selisih Acc antara σ=0% dan σ=20%.

| Method | σ=0% | σ=5% | σ=10% | σ=20% | Drop@20% |
|---|---:|---:|---:|---:|---:|
| `v9_selector` | 97.37% | 95.18% | 95.18% | 94.74% | 2.63% |
| `v9_b2_median_v6` | 96.05% | 94.30% | 94.30% | 93.86% | 2.19% |
| `v6_selector` | 96.05% | 94.30% | 94.30% | 93.86% | 2.19% |
| `v7_stacking_bracketed` | 94.30% | 93.86% | 93.86% | 93.86% | 0.44% |
| `v7_stacking_density` | 94.30% | 93.86% | 93.86% | 93.86% | 0.44% |
| `v8_entropy_modulated` | 94.30% | 93.86% | 92.98% | 92.98% | 1.32% |
| `v5_adaptive_corrected` | 93.86% | 93.86% | 93.86% | 93.86% | **0.00%** |
| `v8_b2_b4_boosted` | 92.54% | 93.42% | 93.42% | 93.42% | **−0.88%** |
| `v2_visibility` | 92.54% | 92.11% | 91.67% | 91.67% | 0.87% |
| `v5_best_visibility` | 92.54% | 92.11% | 91.67% | 91.67% | 0.87% |
| `v1_corrected` | 90.79% | 90.79% | 90.79% | 90.79% | **0.00%** |

`v5_adaptive_corrected` dan `v1_corrected` drop 0% karena tidak memakai koordinat bbox — hanya menghitung kelas. `v8_b2_b4_boosted` bahkan naik karena noise mengurangi over-prediction-nya.

### Breakdown Per Split (train=196, test=31, val=1)

| Method | test Acc | train Acc | val Acc |
|---|---:|---:|---:|
| `v9_selector` | **90.32%** | **98.47%** | 100.00% |
| `v9_b2_median_v6` | 87.10% | 97.45% | 100.00% |
| `v6_selector` | 83.87% | 97.96% | 100.00% |
| `v7_stacking_bracketed` | 83.87% | 95.92% | 100.00% |
| `v7_stacking_density` | 83.87% | 95.92% | 100.00% |
| `v8_entropy_modulated` | 83.87% | 95.92% | 100.00% |
| `v5_adaptive_corrected` | 80.65% | 95.92% | 100.00% |
| `v8_b2_b4_boosted` | 83.87% | 93.88% | 100.00% |
| `v2_visibility` | 80.65% | 94.39% | 100.00% |
| `v5_best_visibility` | 80.65% | 94.39% | 100.00% |
| `v1_corrected` | 80.65% | 92.35% | 100.00% |

`v9_selector` memimpin di test set (90.32%) dengan selisih 6+ poin — perbedaan paling bermakna ada di split paling sulit.

### Ringkasan Tradeoff Antar Dimensi

| Method | Acc ±1 | Rank Acc | ms/pohon | Rank Speed | Drop@20% | Rank Robust |
|---|---:|---:|---:|---:|---:|---:|
| `v9_selector` | 97.37% | #1 | 0.079 | #8 | 2.63% | #11 |
| `v9_b2_median_v6` | 96.05% | #2 | 0.429 | #11 | 2.19% | #10 |
| `v6_selector` | 96.05% | #3 | 0.099 | #9 | 2.19% | #9 |
| `v7_stacking_bracketed` | 94.30% | #4 | 0.048 | #7 | 0.44% | #5 |
| `v7_stacking_density` | 94.30% | #5 | 0.014 | #3 | 0.44% | #4 |
| `v8_entropy_modulated` | 94.30% | #6 | 0.105 | #10 | 1.32% | #8 |
| `v5_adaptive_corrected` | 93.86% | #7 | 0.007 | #2 | 0.00% | #3 |
| `v8_b2_b4_boosted` | 92.54% | #8 | 0.048 | #6 | −0.88% | #1 |
| `v2_visibility` | 92.54% | #9 | 0.022 | #4 | 0.87% | #7 |
| `v5_best_visibility` | 92.54% | #10 | 0.023 | #5 | 0.87% | #6 |
| `v1_corrected` | 90.79% | #11 | 0.004 | #1 | 0.00% | #2 |

| Kebutuhan | Rekomendasi | Alasan |
|---|---|---|
| Akurasi maksimal (JSON GT) | `v9_selector` | #1 Acc, test 90.32% |
| Throughput tinggi, Acc >93% | `v5_adaptive_corrected` | 136k pohon/det, 0% noise drop |
| Balance Acc + Speed | `v7_stacking_density` | Acc 94.30%, 70k pohon/det |
| Pipeline noisy (TXT prediksi) | `v5_adaptive_corrected` | paling robust terhadap noise koordinat |
| Tidak butuh koordinat bbox | `v1_corrected` | hanya butuh label kelas, 280k pohon/det |

---

## Evolusi Metode

| Gen | Method | Acc ±1 | Catatan |
|---|---|---:|---|
| naive | — | ~2% | overcount ~78.8% |
| v1 | `corrected` | 90.79% | divisor global |
| v2 | `visibility` | 92.54% | geometri sederhana |
| v3 | `per_class_ridge` | 90.79% | threshold dari links — tidak breakthrough |
| v4 | `visibility` | 92.54% | HSV + Hungarian tidak menembus v2 |
| v5 | `adaptive_corrected` | 93.86% | adaptive divisor, >93% stabil |
| v6 | `v6_selector` | 96.05% | **titik balik** — routing per regime |
| v7 | `stacking_bracketed` | 94.30% | stacking family, unggul di throughput |
| v8 | `stacking_bracketed_v7` | 94.30% | entropy/per-side tidak breakthrough |
| v9 | `v9_selector` | **97.37%** | narrow overrides di atas v6 |

**Logika override v9:**
1. default → `v6_selector`
2. `b4_only_overlap` → `v7_stacking_bracketed`
3. `classaware_compact_lowb4` → `v8_b2_b4_boosted`
4. `b3b4_only_lowtotal` → `v8_floor_anchor_50`
5. `dense_allside_moderatedup` → `v8_b2_b4_boosted`

---

## Total Tandan Seluruh Dataset (945 Pohon)

Target rasio empiris **0.5594** (dari 228 pohon ber-GT: unique/naive = 2466/4408).

| Rank | Metode | Total | Rasio | Jarak ke 0.56 |
|---:|---|---:|---:|---:|
| 1 | `v8_b2_b4_boosted` | 10,129 | 0.5604 | 0.0010 |
| 2 | `v9_median_strong5` | 10,130 | 0.5605 | 0.0011 |
| 3 | `hybrid_vis_corr` | 9,988 | 0.5526 | 0.0068 |
| 11 | `v9_selector` | 10,449 | 0.5782 | 0.0188 |
| 12 | `v6_selector` | 10,467 | 0.5792 | 0.0198 |
| — | `naive` | 18,073 | 1.0000 | 0.4406 ← overcount |
| — | `relaxed_match` | 3,345 | 0.1851 | 0.3743 ← undercount |

Detail per pohon: `reports/dedup_all_953/all_953_per_tree.csv`

---

## Menjalankan Script

```bash
python scripts/count_all_trees.py          # GT counting semua 953 pohon
python scripts/count_gt_vs_naive.py        # JSON-05 + JSON-01 audit
python scripts/benchmark_multidim.py       # benchmark 4 dimensi (11 algoritma)
python scripts/dedup_research_v9.py        # v9 research script
python scripts/dedup_all_953.py            # semua metode pada 945 pohon
python scripts/dedup_nonjson_compare.py    # non-JSON + report
```

---

## Struktur Repo

```
json/                    228 file JSON dengan bunch-link antar-view
dataset/                 image dan label YOLO
scripts/                 audit, counting, dan dedup research
algorithms/              implementasi bersih tiap algoritma
reports/
  benchmark_multidim/    akurasi + kecepatan + robustness + per-split
  dedup_research_v9/     benchmark script riset terbaru
  dedup_all_953/         semua metode pada 945 pohon
RESEARCH.md              dokumen riset panjang — baca Section 0 dulu
AGENTS.md                instruksi operasional repo
```

---

## Schema JSON Ringkas

```json
{
  "tree_id": "20260422-DAMIMAS-001",
  "images": {
    "sisi_1": {
      "annotations": [{"class_name": "B3", "bbox_yolo": [0.5, 0.5, 0.1, 0.2], "box_index": 0}]
    }
  },
  "bunches": [{"bunch_id": 1, "class": "B3", "appearance_count": 2}],
  "summary": {"by_class": {"B1": 1, "B2": 2, "B3": 5, "B4": 0}}
}
```

`summary.by_class` = ground truth count unik per kelas.

---

## Batasan Riset

**100% algorithmic / heuristic.** Tidak boleh:
- Siamese / CNN embedding
- Learned thresholds via backprop
- MLP pada fitur bbox
- Strict matching (Hungarian/graph/cluster) pada TXT labels — broken oleh coordinate noise

Laporan lengkap: [`reports/benchmark_multidim/REPORT.md`](reports/benchmark_multidim/REPORT.md)
