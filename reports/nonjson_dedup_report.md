# Laporan Dedup Multi-View Non-JSON

**Tanggal:** 2026-04-23  
**Dataset:** DAMIMAS + LONSUM, 953 pohon total  
**Fokus:** 717 pohon tanpa JSON ground truth  
**Referensi validasi:** 228 pohon ber-JSON

---

## 1. Ringkasan

- Benchmark JSON terbaru sekarang ada di `v6`, `v7`, dan `v8`.
- **Metode terbaik pada 228 pohon JSON adalah `v6_selector`**:
  - `Acc ±1`: **96.49%**
  - `MAE`: **0.2632**
  - `Mean Total Error`: **1.05**
- **Metode terbaik untuk 717 pohon non-JSON belum tentu sama**, karena non-JSON tidak punya GT.
- Untuk non-JSON, metode yang paling masuk akal tetap dinilai dari **kedekatan rasio dedup ke target ~0.56** dari subset JSON terverifikasi.

---

## 2. Validasi pada 228 Pohon JSON

Sumber: `reports/dedup_all_trees_final/json_228_accuracy.csv`

| Method | Mean MAE | Acc ±1 | Mean Total Err | Score |
|--------|---------:|-------:|---------------:|------:|
| **v6_selector** | **0.2632** | **96.49%** | **1.05** | **93.86** |
| stacking_bracketed_v7 | 0.2643 | 94.30% | 1.06 | 91.66 |
| stacking_density_v7 | 0.2708 | 94.30% | 1.08 | 91.59 |
| entropy_modulated_v8 | 0.2763 | 94.30% | 1.11 | 91.54 |
| v8_entropy_stacking | 0.2763 | 94.30% | 1.11 | 91.54 |
| adaptive_corrected | 0.2774 | 93.86% | 1.11 | 91.09 |
| best_ensemble_grid | 0.2774 | 93.86% | 1.11 | 91.09 |
| best_visibility_grid | 0.2664 | 92.54% | 1.07 | 89.88 |
| visibility | 0.2719 | 92.11% | 1.09 | 89.39 |
| corrected | 0.2851 | 90.79% | 1.14 | 87.94 |

Interpretasi:
- `v6_selector` sekarang **benchmark JSON internal terbaik**.
- `v7` dan `v8` tetap penting sebagai baseline baru, tetapi belum melewati `v6`.
- Jadi urutan saat ini adalah:
  - `v6_selector` > `v7/v8 best` > `adaptive_corrected v5` > `visibility v2/v4`

---

## 3. Hasil Dedup pada 717 Pohon Non-JSON

Sumber: `reports/dedup_all_trees_final/nonjson_725_summary.csv`

### 3.1 Total Count per Metode

| Method | B1 | B2 | B3 | B4 | Total | Rasio vs Naive |
|---|---:|---:|---:|---:|---:|---:|
| naive | 1618 | 2974 | 6417 | 2656 | 13665 | 100.0% |
| class_aware_vis | 917 | 1841 | 3974 | 1478 | 8210 | 60.4% |
| entropy_modulated_v8 | 898 | 1761 | 3702 | 1712 | 8073 | 59.1% |
| v8_entropy_stacking | 898 | 1761 | 3702 | 1712 | 8073 | 59.1% |
| adaptive_corrected | 855 | 1760 | 3724 | 1671 | 8010 | 58.6% |
| adaptive_visibility | 956 | 1756 | 3728 | 1559 | 7999 | 58.5% |
| v6_selector | 857 | 1764 | 3683 | 1647 | 7951 | 58.2% |
| stacking_bracketed_v7 | 864 | 1736 | 3672 | 1652 | 7924 | 58.0% |
| best_class_aware_grid | 919 | 1744 | 3718 | 1481 | 7862 | 57.5% |
| stacking_density_v7 | 831 | 1719 | 3662 | 1636 | 7848 | 57.4% |
| best_ensemble_grid | 880 | 1732 | 3653 | 1565 | 7830 | 57.3% |
| corrected | 917 | 1691 | 3573 | 1598 | 7779 | 56.9% |
| hybrid_vis_corr | 919 | 1668 | 3504 | 1505 | 7596 | 55.6% |
| side_coverage | 921 | 1665 | 3465 | 1500 | 7551 | 55.3% |
| density_scaled_vis | 919 | 1657 | 3486 | 1484 | 7546 | 55.2% |
| best_visibility_grid | 919 | 1655 | 3486 | 1481 | 7541 | 55.2% |
| visibility | 919 | 1650 | 3458 | 1483 | 7510 | 55.0% |
| ordinal_prior | 919 | 1674 | 3434 | 1483 | 7510 | 55.0% |
| naive_mean_ensemble | 917 | 1646 | 3394 | 1483 | 7440 | 54.4% |
| best_relaxed_grid | 620 | 818 | 879 | 748 | 3065 | 22.4% |
| relaxed_match | 512 | 674 | 720 | 623 | 2529 | 18.5% |

### 3.2 Rasio Dedup per Pohon

Sumber: `reports/dedup_all_trees_final/nonjson_725_ratios.csv`

| Method | Mean Ratio | Median Ratio | Std Dev |
|---|---:|---:|---:|
| class_aware_vis | 0.6042 | 0.6000 | 0.0497 |
| entropy_modulated_v8 | 0.5872 | 0.5833 | 0.0583 |
| v8_entropy_stacking | 0.5872 | 0.5833 | 0.0583 |
| best_class_aware_grid | 0.5799 | 0.5714 | 0.0492 |
| stacking_bracketed_v7 | 0.5763 | 0.5714 | 0.0589 |
| adaptive_visibility | 0.5754 | 0.5833 | 0.0623 |
| corrected | 0.5741 | 0.5714 | 0.0480 |
| adaptive_corrected | 0.5736 | 0.5769 | 0.0574 |
| v6_selector | 0.5714 | 0.5714 | 0.0575 |
| best_ensemble_grid | 0.5660 | 0.5714 | 0.0514 |
| stacking_density_v7 | 0.5636 | 0.5652 | 0.0555 |
| side_coverage | 0.5615 | 0.5556 | 0.0553 |
| hybrid_vis_corr | 0.5601 | 0.5556 | 0.0510 |
| best_visibility_grid | 0.5587 | 0.5556 | 0.0501 |
| density_scaled_vis | 0.5582 | 0.5500 | 0.0509 |
| visibility | 0.5570 | 0.5500 | 0.0509 |
| ordinal_prior | 0.5570 | 0.5500 | 0.0509 |
| naive_mean_ensemble | 0.5525 | 0.5455 | 0.0521 |
| best_relaxed_grid | 0.2604 | 0.2143 | 0.1578 |
| relaxed_match | 0.2156 | 0.1875 | 0.1208 |

---

## 4. Ranking Non-JSON

Karena non-JSON tidak punya GT, ranking yang paling defensible adalah:

`ranking = kedekatan ke target ratio ~0.56`

Target `~0.56` berasal dari verifikasi JSON-05:

```text
unique_count ≈ naive_count / 1.788 ≈ naive_count × 0.559
```

### 4.1 Ranking Kedekatan ke 0.56

| Rank | Method | Mean Ratio | Jarak ke 0.56 |
|---:|---|---:|---:|
| 1 | hybrid_vis_corr | 0.5601 | 0.0001 |
| 2 | side_coverage | 0.5615 | 0.0015 |
| 3 | stacking_density_v7 | 0.5636 | 0.0036 |
| 4 | best_ensemble_grid | 0.5660 | 0.0060 |
| 5 | v6_selector | 0.5714 | 0.0114 |
| 6 | adaptive_corrected | 0.5736 | 0.0136 |
| 7 | corrected | 0.5741 | 0.0141 |
| 8 | stacking_bracketed_v7 | 0.5763 | 0.0163 |
| 9 | best_class_aware_grid | 0.5799 | 0.0199 |
| 10 | entropy_modulated_v8 | 0.5872 | 0.0272 |
| 10 | v8_entropy_stacking | 0.5872 | 0.0272 |
| 12 | class_aware_vis | 0.6042 | 0.0442 |
| 13 | best_relaxed_grid | 0.2604 | 0.2996 |
| 14 | relaxed_match | 0.2156 | 0.3444 |

### 4.2 Implikasi

- **`v6_selector` adalah yang terbaik di JSON**, tapi **bukan yang paling dekat ke rasio non-JSON target**.
- Untuk non-JSON, metode paling stabil terhadap target dedup tetap:
  - `hybrid_vis_corr`
  - `side_coverage`
  - `stacking_density_v7`
  - `best_ensemble_grid`
  - `visibility` / `best_visibility_grid`
- `v8` cenderung lebih tinggi dari target, jadi berpotensi sedikit overcount pada data tanpa GT.

---

## 5. Rekomendasi Praktis

### 5.1 Jika ada GT / benchmark JSON

Gunakan:
- **`v6_selector`**

Karena ini sekarang yang terbaik secara akurasi:
- `96.49% Acc ±1`
- `MAE 0.2632`

### 5.2 Jika tidak ada GT / pada 717 non-JSON

Gunakan salah satu dari:
1. **`hybrid_vis_corr`**
2. **`side_coverage`**
3. **`stacking_density_v7`**
4. **`best_visibility_grid`**
5. **`visibility`**

Alasan:
- paling dekat ke rasio target `~0.56`
- lebih aman untuk inferensi tanpa label pembanding

### 5.3 Jika ingin kompromi antara benchmark JSON dan non-JSON

Gunakan:
- **`stacking_density_v7`**

Karena:
- performa JSON kuat: `94.30%`
- rasio non-JSON masih dekat target: `0.5636`
- lebih seimbang daripada `v6_selector` jika fokus Anda adalah generalisasi operasional

---

## 6. Kesimpulan

- `v6_selector` sekarang **benchmark internal terbaik** untuk 228 pohon JSON.
- `v7` dan `v8` tetap relevan, tetapi **tidak melewati v6** pada benchmark JSON.
- Untuk 717 pohon non-JSON, **metode terbaik bukan otomatis v6**.
- Tanpa GT, keputusan terbaik tetap berbasis:
  - rasio dedup yang masuk akal,
  - stabilitas antar-pohon,
  - dan kedekatan ke target `~0.56`.

Jadi garis besarnya:
- **JSON benchmark terbaik:** `v6_selector`
- **Non-JSON produksi paling aman:** `hybrid_vis_corr` / `side_coverage` / `stacking_density_v7`

---

## 7. File Output

| File | Lokasi | Isi |
|---|---|---|
| `json_228_accuracy.csv` | `reports/dedup_all_trees_final/` | Benchmark JSON semua metode termasuk v6/v7/v8 |
| `nonjson_725_counts.csv` | `reports/dedup_all_trees_final/` | Count per pohon non-JSON |
| `nonjson_725_summary.csv` | `reports/dedup_all_trees_final/` | Total count per metode |
| `nonjson_725_ratios.csv` | `reports/dedup_all_trees_final/` | Rasio dedup per metode |
| `summary_v6.md` | `reports/dedup_research_v6/` | Ringkasan v6 |
| `summary_v7.md` | `reports/dedup_research_v7/` | Ringkasan v7 |
| `summary_v8.md` | `reports/dedup_research_v8/` | Ringkasan v8 |

