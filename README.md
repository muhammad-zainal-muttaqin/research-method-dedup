# Multi-View Oil Palm Bunch Counting

Dedup multi-view untuk menghitung tandan sawit unik per pohon dari 4–8 sisi foto. **Tanpa training.** Hanya heuristik + routing algoritmik.

## Masalah

Satu tandan muncul di beberapa sisi. Penjumlahan naif overcount ~78.8%. Target: ubah deteksi per-sisi menjadi **unique bunch count per kelas kematangan**.

## Dataset

| Item | Jumlah |
|---|---:|
| Total pohon | 953 (DAMIMAS 854 + LONSUM 99) |
| Dengan JSON GT | 228 |
| Hanya TXT prediksi | 725 |
| Sisi per pohon | 4 (mayoritas), 8 (~45 terbaru) |

**Kelas ordinal B1→B4** — B1 merah paling matang (bawah), B2 transisi, B3 hitam, B4 kecil berduri (atas). Ambiguitas utama: **B2↔B3** (irreducible, bukan label noise).

---

## Metrik Primer

Seluruh hasil di bawah dihitung pada 228 pohon ber-GT JSON.

| Metrik | Arah | Definisi |
|---|:---:|---|
| **Per-class MAE** | ↓ | rata-rata \|pred − GT\| tiap kelas (B1/B2/B3/B4), dirata-rata lintas pohon |
| **Macro class-MAE** | ↓ | rata-rata dari 4 per-class MAE (bobot sama antar kelas) |
| **Exact accuracy** | ↑ | % pohon dengan prediksi tepat sama GT di **semua** kelas |
| **Total count MAE** | ↓ | rata-rata \|Σpred − ΣGT\| per pohon |
| **Total ±1 accuracy** | ↑ | % pohon dengan total count dalam ±1 dari total GT |
| **Per-class mean error** | →0 | rata-rata (pred − GT) per kelas — mengukur **bias arah** (+ overcount, − undercount) |

> Legenda: **↓** makin kecil makin baik, **↑** makin besar makin baik, **→0** ideal mendekati nol.

**Traceability.** Setiap nama metode di seluruh tabel README bisa diklik → halaman [`reports/methods/<method>.md`](reports/methods) yang berisi nilai semua metrik primer untuk metode itu, derivasi perhitungan, CSV sumber ([`accuracy_per_tree.csv`](reports/benchmark_multidim/accuracy_per_tree.csv), [`accuracy_per_class.csv`](reports/benchmark_multidim/accuracy_per_class.csv), [`accuracy_summary.csv`](reports/benchmark_multidim/accuracy_summary.csv), [`speed_summary.csv`](reports/benchmark_multidim/speed_summary.csv), [`robustness_summary.csv`](reports/benchmark_multidim/robustness_summary.csv)), daftar pohon yang gagal, dan sample per-tree rows. Reproduce:

```bash
python scripts/benchmark_multidim.py         # regenerate CSV mentah (benchmark)
python scripts/generate_method_reports.py    # regenerate per-method breakdown
```

---

## Hasil Utama — 11 Algoritma

Urut berdasarkan **macro class-MAE**. Nama metode link ke breakdown data mentah; kolom *Impl* link ke file algoritma.

| Rank | Method | Impl | Macro MAE ↓ | Exact % ↑ | Total MAE ↓ | Total ±1 % ↑ |
|---:|---|---|---:|---:|---:|---:|
| 1 | [`v9_selector`](reports/methods/v9_selector.md) | [py](algorithms/v9_selector.py) | **0.2533** | 29.39% | **0.8553** | **83.77%** |
| 2 | [`v9_b2_median_v6`](reports/methods/v9_b2_median_v6.md) | [py](algorithms/b2_median_v6.py) | 0.2577 | 29.82% | 0.8640 | 82.02% |
| 3 | [`v6_selector`](reports/methods/v6_selector.md) | [py](algorithms/v6_selector.py) | 0.2599 | 28.07% | 0.8816 | 82.46% |
| 4 | [`v8_b2_b4_boosted`](reports/methods/v8_b2_b4_boosted.md) | [py](algorithms/b2_b4_boosted.py) | 0.2632 | **31.14%** | 0.9035 | 80.26% |
| 5 | [`v7_stacking_bracketed`](reports/methods/v7_stacking_bracketed.md) | [py](algorithms/stacking_bracketed.py) | 0.2643 | 31.14% | 0.8904 | 79.82% |
| 6 | [`v2_visibility`](reports/methods/v2_visibility.md) | — | 0.2664 | **31.58%** | 0.8728 | 82.02% |
| 7 | [`v5_best_visibility`](reports/methods/v5_best_visibility.md) | [py](algorithms/best_visibility_grid.py) | 0.2664 | 31.58% | 0.8728 | 82.02% |
| 8 | [`v7_stacking_density`](reports/methods/v7_stacking_density.md) | [py](algorithms/stacking_density.py) | 0.2708 | 29.39% | 0.9079 | 79.39% |
| 9 | [`v8_entropy_modulated`](reports/methods/v8_entropy_modulated.md) | [py](algorithms/entropy_modulated.py) | 0.2763 | 30.70% | 0.8772 | 78.95% |
| 10 | [`v5_adaptive_corrected`](reports/methods/v5_adaptive_corrected.md) | [py](algorithms/adaptive_corrected.py) | 0.2774 | 26.32% | 0.9342 | 79.82% |
| 11 | [`v1_corrected`](reports/methods/v1_corrected.md) | — | 0.2851 | 30.26% | 0.9035 | 78.51% |

Sumber: [`accuracy_per_tree.csv`](reports/benchmark_multidim/accuracy_per_tree.csv) (228 pohon × 11 metode = 2508 baris). Derivasi tiap angka ada di breakdown per-metode.

### Per-Class MAE (↓ lebih kecil lebih baik)

Sumber: kolom `err_B*` (sudah absolute) di [`accuracy_per_tree.csv`](reports/benchmark_multidim/accuracy_per_tree.csv); cross-check di [`accuracy_per_class.csv`](reports/benchmark_multidim/accuracy_per_class.csv).

| Method | B1 ↓ | B2 ↓ | B3 ↓ | B4 ↓ |
|---|---:|---:|---:|---:|
| [`v9_selector`](reports/methods/v9_selector.md) | **0.105** | **0.219** | **0.386** | 0.303 |
| [`v9_b2_median_v6`](reports/methods/v9_b2_median_v6.md) | 0.105 | 0.224 | 0.395 | 0.307 |
| [`v6_selector`](reports/methods/v6_selector.md) | 0.105 | 0.232 | 0.395 | 0.307 |
| [`v8_b2_b4_boosted`](reports/methods/v8_b2_b4_boosted.md) | **0.079** | 0.268 | 0.425 | **0.281** |
| [`v7_stacking_bracketed`](reports/methods/v7_stacking_bracketed.md) | 0.079 | 0.241 | 0.425 | 0.311 |
| [`v2_visibility`](reports/methods/v2_visibility.md) | 0.101 | 0.224 | 0.447 | 0.294 |
| [`v5_best_visibility`](reports/methods/v5_best_visibility.md) | 0.101 | 0.224 | 0.447 | 0.294 |
| [`v7_stacking_density`](reports/methods/v7_stacking_density.md) | 0.096 | 0.241 | 0.425 | 0.320 |
| [`v8_entropy_modulated`](reports/methods/v8_entropy_modulated.md) | 0.096 | 0.254 | 0.434 | 0.320 |
| [`v5_adaptive_corrected`](reports/methods/v5_adaptive_corrected.md) | 0.110 | 0.237 | 0.434 | 0.329 |
| [`v1_corrected`](reports/methods/v1_corrected.md) | 0.101 | 0.224 | 0.487 | 0.329 |

B1 termudah untuk semua metode. **B3 adalah bottleneck** universal.

### Per-Class Mean Error (Bias) (→0 ideal)

Nilai positif = overcount, negatif = undercount, nol = tidak bias. Sumber: `mean(pred_B* − gt_B*)` di [`accuracy_per_tree.csv`](reports/benchmark_multidim/accuracy_per_tree.csv).

| Method | B1 →0 | B2 →0 | B3 →0 | B4 →0 |
|---|---:|---:|---:|---:|
| [`v9_selector`](reports/methods/v9_selector.md) | +0.044 | +0.044 | **0.000** | +0.039 |
| [`v9_b2_median_v6`](reports/methods/v9_b2_median_v6.md) | +0.044 | −0.101 | +0.009 | +0.044 |
| [`v6_selector`](reports/methods/v6_selector.md) | +0.044 | +0.057 | +0.009 | +0.044 |
| [`v8_b2_b4_boosted`](reports/methods/v8_b2_b4_boosted.md) | +0.044 | −0.154 | +0.004 | −0.114 |
| [`v7_stacking_bracketed`](reports/methods/v7_stacking_bracketed.md) | +0.044 | +0.039 | +0.004 | +0.048 |
| [`v2_visibility`](reports/methods/v2_visibility.md) | +0.092 | −0.066 | −0.175 | −0.215 |
| [`v5_best_visibility`](reports/methods/v5_best_visibility.md) | +0.092 | −0.066 | −0.175 | −0.215 |
| [`v7_stacking_density`](reports/methods/v7_stacking_density.md) | +0.026 | +0.039 | +0.004 | +0.039 |
| [`v8_entropy_modulated`](reports/methods/v8_entropy_modulated.md) | +0.070 | +0.070 | +0.048 | +0.110 |
| [`v5_adaptive_corrected`](reports/methods/v5_adaptive_corrected.md) | +0.039 | +0.061 | +0.057 | +0.066 |
| [`v1_corrected`](reports/methods/v1_corrected.md) | +0.092 | −0.048 | −0.039 | +0.022 |

[`v9_selector`](reports/methods/v9_selector.md) punya profil bias paling seimbang: magnitude ≤0.044 di semua kelas, **tepat 0 di B3** (kelas tersulit).

---

## Metrik Pelengkap

### Acc ±1 per kelas per pohon (pohon lulus jika semua 4 kelas meleset ≤1)

Sumber: kolom `acc_pct` dan `n_fail` di [`accuracy_summary.csv`](reports/benchmark_multidim/accuracy_summary.csv).

| Rank | Method | Acc ±1 ↑ | Gagal ↓ |
|---:|---|---:|---:|
| 1 | [`v9_selector`](reports/methods/v9_selector.md) | **97.37%** | 6 |
| 2 | [`v9_b2_median_v6`](reports/methods/v9_b2_median_v6.md) | 96.05% | 9 |
| 3 | [`v6_selector`](reports/methods/v6_selector.md) | 96.05% | 9 |
| 4 | [`v7_stacking_bracketed`](reports/methods/v7_stacking_bracketed.md) | 94.30% | 13 |
| 5 | [`v7_stacking_density`](reports/methods/v7_stacking_density.md) | 94.30% | 13 |
| 6 | [`v8_entropy_modulated`](reports/methods/v8_entropy_modulated.md) | 94.30% | 13 |
| 7 | [`v5_adaptive_corrected`](reports/methods/v5_adaptive_corrected.md) | 93.86% | 14 |
| 8 | [`v8_b2_b4_boosted`](reports/methods/v8_b2_b4_boosted.md) | 92.54% | 17 |
| 9 | [`v2_visibility`](reports/methods/v2_visibility.md) | 92.54% | 17 |
| 10 | [`v5_best_visibility`](reports/methods/v5_best_visibility.md) | 92.54% | 17 |
| 11 | [`v1_corrected`](reports/methods/v1_corrected.md) | 90.79% | 21 |

### Kecepatan (ms/pohon, 30 repetisi × 228 pohon)

Sumber: [`speed_summary.csv`](reports/benchmark_multidim/speed_summary.csv).

| Method | ms ↓ | pohon/detik ↑ |
|---|---:|---:|
| [`v1_corrected`](reports/methods/v1_corrected.md) | 0.004 | 279,830 |
| [`v5_adaptive_corrected`](reports/methods/v5_adaptive_corrected.md) | 0.007 | 136,242 |
| [`v7_stacking_density`](reports/methods/v7_stacking_density.md) | 0.014 | 70,585 |
| [`v2_visibility`](reports/methods/v2_visibility.md) | 0.022 | 45,146 |
| [`v9_selector`](reports/methods/v9_selector.md) | 0.079 | 12,619 |
| [`v6_selector`](reports/methods/v6_selector.md) | 0.099 | 10,074 |
| [`v9_b2_median_v6`](reports/methods/v9_b2_median_v6.md) | 0.429 | 2,330 |

### Robustness (Gaussian noise σ=20% pada koordinat)

Sumber: [`robustness_summary.csv`](reports/benchmark_multidim/robustness_summary.csv).

| Method | Drop Acc @ σ=20% ↓ |
|---|---:|
| [`v1_corrected`](reports/methods/v1_corrected.md), [`v5_adaptive_corrected`](reports/methods/v5_adaptive_corrected.md) | 0.00% (tidak pakai koordinat) |
| [`v8_b2_b4_boosted`](reports/methods/v8_b2_b4_boosted.md) | −0.88% (malah naik) |
| [`v7_stacking_bracketed`](reports/methods/v7_stacking_bracketed.md), [`v7_stacking_density`](reports/methods/v7_stacking_density.md) | 0.44% |
| [`v9_selector`](reports/methods/v9_selector.md) | 2.63% (paling sensitif) |

### Per Split (train=196, test=31, val=1)

Sumber: [`domain_breakdown.csv`](reports/benchmark_multidim/domain_breakdown.csv).

| Method | test ↑ | train ↑ | val ↑ |
|---|---:|---:|---:|
| [`v9_selector`](reports/methods/v9_selector.md) | **90.32%** | **98.47%** | 100% |
| [`v9_b2_median_v6`](reports/methods/v9_b2_median_v6.md) | 87.10% | 97.45% | 100% |
| [`v6_selector`](reports/methods/v6_selector.md) | 83.87% | 97.96% | 100% |

[`v9_selector`](reports/methods/v9_selector.md) unggul 6+ poin di test set — gap terbesar justru di split tersulit.

---

## Rekomendasi

| Kebutuhan | Pilihan |
|---|---|
| Akurasi maksimal pada JSON GT | [`v9_selector`](reports/methods/v9_selector.md) |
| Throughput tinggi + Acc ±1 >93% | [`v5_adaptive_corrected`](reports/methods/v5_adaptive_corrected.md) |
| Tradeoff akurasi/kecepatan | [`v7_stacking_density`](reports/methods/v7_stacking_density.md) |
| Pipeline TXT noisy (tanpa GT) | [`v5_adaptive_corrected`](reports/methods/v5_adaptive_corrected.md) atau [`v1_corrected`](reports/methods/v1_corrected.md) |
| Tidak butuh koordinat bbox | [`v1_corrected`](reports/methods/v1_corrected.md) |

---

## Evolusi Metode

| Gen | Method | Macro MAE ↓ | Catatan |
|---|---|---:|---|
| naive | — | — | overcount ~78.8% |
| v1 | [`v1_corrected`](reports/methods/v1_corrected.md) | 0.2851 | divisor global |
| v2 | [`v2_visibility`](reports/methods/v2_visibility.md) | 0.2664 | geometri sederhana |
| v5 | [`v5_adaptive_corrected`](reports/methods/v5_adaptive_corrected.md) | 0.2774 | adaptive divisor, >93% Acc±1 |
| v6 | [`v6_selector`](reports/methods/v6_selector.md) | 0.2599 | **titik balik** — routing per regime |
| v7 | [`v7_stacking_bracketed`](reports/methods/v7_stacking_bracketed.md) | 0.2643 | stacking density family |
| v8 | [`v8_b2_b4_boosted`](reports/methods/v8_b2_b4_boosted.md) | 0.2632 | per-kelas boosting |
| v9 | [`v9_selector`](reports/methods/v9_selector.md) | **0.2533** | narrow overrides di atas v6 |

**Logika [`v9_selector`](reports/methods/v9_selector.md):**
1. default → [`v6_selector`](reports/methods/v6_selector.md)
2. `b4_only_overlap` → [`v7_stacking_bracketed`](reports/methods/v7_stacking_bracketed.md)
3. `classaware_compact_lowb4` → [`v8_b2_b4_boosted`](reports/methods/v8_b2_b4_boosted.md)
4. `b3b4_only_lowtotal` → `v8_floor_anchor_50`
5. `dense_allside_moderatedup` → [`v8_b2_b4_boosted`](reports/methods/v8_b2_b4_boosted.md)

---

## Total Tandan 945 Pohon

Target rasio empiris **0.5594** (unique/naive dari 228 JSON = 2466/4408). Sumber: [`reports/dedup_all_953/all_953_totals.csv`](reports/dedup_all_953/all_953_totals.csv), detail per-pohon di [`all_953_per_tree.csv`](reports/dedup_all_953/all_953_per_tree.csv).

| Metode | Total | Rasio →0.5594 | Jarak ↓ |
|---|---:|---:|---:|
| `v8_b2_b4_boosted` | 10,129 | 0.5604 | **0.0010** |
| `v9_median_strong5` | 10,130 | 0.5605 | 0.0011 |
| `hybrid_vis_corr` | 9,988 | 0.5526 | 0.0068 |
| `v9_selector` | 10,449 | 0.5782 | 0.0188 |
| `v6_selector` | 10,467 | 0.5792 | 0.0198 |
| naive | 18,073 | 1.0000 | 0.4406 |

> Breakdown per-metode di section ini memakai CSV terpisah (`dedup_all_953/`) — bukan subset 228. Link breakdown per metode tidak mencakup 945 ini.

---

## Menjalankan

```bash
pip install -r requirements.txt

python scripts/count_all_trees.py            # GT counting 953 pohon
python scripts/count_gt_vs_naive.py          # audit JSON-05 + JSON-01
python scripts/benchmark_multidim.py         # benchmark 4-dimensi 11 algoritma → reports/benchmark_multidim/
python scripts/generate_method_reports.py    # regenerate reports/methods/<method>.md dari CSV di atas
python scripts/dedup_research_v9.py          # research script v9
python scripts/dedup_all_953.py              # semua metode pada 945 pohon
python scripts/dedup_nonjson_compare.py      # validasi non-JSON
```

## Struktur Repo

```
json/                    228 file JSON multi-view bunch links
dataset/                 image + label YOLO
algorithms/              satu file per algoritma
scripts/                 audit, counting, dedup research, report generator
reports/
  benchmark_multidim/    CSV mentah: accuracy_per_tree, accuracy_per_class, speed, robustness, domain_breakdown
  methods/               per-method markdown breakdown + per-tree CSV slice (linked dari README)
  dedup_research_v9/     research script terbaru
  dedup_all_953/         semua metode pada 945 pohon
RESEARCH.md              dokumen riset panjang (baca Section 0)
AGENTS.md                instruksi operasional
```

## Schema JSON

```json
{
  "tree_id": "20260422-DAMIMAS-001",
  "images": {
    "sisi_1": {"annotations": [{"class_name": "B3", "bbox_yolo": [0.5, 0.5, 0.1, 0.2], "box_index": 0}]}
  },
  "bunches": [{"bunch_id": 1, "class": "B3", "appearance_count": 2}],
  "summary": {"by_class": {"B1": 1, "B2": 2, "B3": 5, "B4": 0}}
}
```

`summary.by_class` = GT unique count per kelas.

## Batasan Riset

100% algoritmik. Tidak boleh: Siamese / CNN embedding, MLP pada fitur bbox, learned threshold via backprop, strict matching (Hungarian / graph / cluster) pada TXT (broken oleh coordinate noise).

Laporan multi-dimensi lengkap: [`reports/benchmark_multidim/REPORT.md`](reports/benchmark_multidim/REPORT.md).
