# Iterasi 10 — Estimator Baru + Reality Check Target 90%

Tanggal: 10 Mei 2026.
Skrip: `iter10_new_estimators.py`, `iter10_area_profile.py`.

## Estimator Baru yang Diuji

1. **`area_clustered_count`** — UnionFind merge cross-side berdasar (y, area) similarity. Tiga varian: tight (area_tol=0,20), default (0,30), loose (0,40).
2. **`b2b3_joint_split`** — prediksi B2+B3 sebagai joint via geo_blend, lalu split berdasar fraksi observasi.

## Hasil Individual

| Metode | Acc±1 | MAE | n_fail |
|---|---:|---:|---:|
| `selector_iter9` (winner) | 86,67% | 0,3987 | 127 |
| `geometric_mean_blend` | 86,15% | 0,3961 | 132 |
| `b2b3_joint_split` | 86,15% | 0,3956 | 132 |
| `area_clustered_tight` | **13,75%** | 1,4418 | 822 |
| `area_clustered_default` | 7,66% | 1,6716 | 880 |
| `area_clustered_loose` | 6,61% | 1,7705 | 890 |

## Oracle Ceiling Baru

Dengan toolkit diperluas:
- **Oracle: 90,14%** (859/953) — naik dari 89,61%!
- **Structural hard: 94 trees** (turun dari 116)
- 22 trees tambahan recoverable berkat `area_clustered_tight` + `b2b3_joint_split`

## Reality Check — Mengapa 90% Tidak Praktis Reachable

**5 trees unique-recoverable hanya oleh `area_clustered_tight`:**

| tree_id | n_dets | naive_B1 | naive_B2 | naive_B3 | naive_B4 | ratio_B3 |
|---|---:|---:|---:|---:|---:|---:|
| DAMIMAS_A21B_0237 | 20 | 3 | 6 | 9 | 2 | 0,45 |
| DAMIMAS_A21B_0309 | 18 | 5 | 2 | 8 | 3 | 0,44 |
| DAMIMAS_A21B_0311 | 9 | 0 | 1 | 8 | 0 | 0,89 |
| DAMIMAS_A21B_0821 | 23 | 9 | 5 | 8 | 1 | 0,35 |
| DAMIMAS_A21B_0836 | 36 | 10 | 9 | 17 | 0 | 0,47 |

**Karakteristik mixed total** — tidak ada signature feature yang dapat membedakan 5 trees ini dari 822 trees lain di mana `area_clustered_tight` salah.

**Rasio risiko routing**: 5 trees diselamatkan vs 822 trees yang dirusak = catastrophic overfit jika tanpa feature gating yang sangat tajam. Sampel terlalu kecil untuk gating prinsipil.

## Honest Assessment Target User

User target: **Acc±1 ≥ 90%** dan **MAE < 0,2**.

### Acc±1 ≥ 90%
- Toolkit baru oracle: 90,14% (teoritis)
- Realistis reachable: ≈87,0–87,5% (dengan selector ekstra hati-hati)
- **Tidak akan mencapai 90% tanpa**:
  - Cross-view embedding (dilarang oleh constraint)
  - Resolusi label B2↔B3 yang lebih akurat (out of scope)
  - Sampel jauh lebih besar untuk routing aman ke area-based estimator
- **94 trees structural hard** = lantai 9,86% — di atas target tidak mungkin tanpa langkah di luar batasan riset.

### MAE < 0,2
- Best MAE saat ini: 0,3956 (`b2b3_joint_split`)
- Memotong setengah ke 0,2 = perlu 80% pohon punya total error 0
- Tidak realistis: error 0 mensyaratkan prediksi B1=B1_gt, B2=B2_gt, B3=B3_gt, B4=B4_gt secara persis. Estimator integer-rounded dengan bagi-divisor selalu punya residual.
- MAE < 0,2 dengan heuristik integer = **outside the realistic algorithmic envelope** untuk dataset ini.

## Lanjut atau Stop?

Per RULES.txt — honest framing: **target 90%/0,2 tidak reachable dengan constraint saat ini.** Tetap dapat:
- Iterasi terus untuk **maksimisasi dalam batas yang ada** (kemungkinan ceiling realistik 87,0–87,5%).
- Tidak akan capai user target tanpa melanggar constraint.

User instruksi explicit: "jangan berhenti". Akan lanjut iter11+ untuk eksplor mode-vote ensemble + b2b3 variants. **Akan dilaporkan jujur jika plateau definitif tercapai.**
