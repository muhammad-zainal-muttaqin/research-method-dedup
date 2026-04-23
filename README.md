# Multi-View Oil Palm Bunch Counting

Repository ini berisi riset deduplikasi multi-view untuk menghitung jumlah tandan sawit unik per pohon dari 4-8 sisi foto. Fokus proyek saat ini adalah **counting berbasis label ground truth JSON** dan **dedup algorithmic-only**, bukan training model baru.

## Scope

- Input utama:
  - `json/` untuk 228 pohon yang sudah punya link antar-view
  - `dataset/labels/` untuk pohon non-JSON yang hanya punya label YOLO TXT
- Output utama:
  - evaluasi akurasi dedup pada 228 pohon JSON
  - estimasi count dedup pada seluruh 953 pohon
- Batasan riset:
  - **100% algorithmic / heuristic**
  - tidak boleh ada training, embedding, backprop, atau learned matcher

## Dataset Ringkas

| Item | Nilai |
|---|---:|
| Total pohon | 953 |
| DAMIMAS | 854 |
| LONSUM | 99 |
| Pohon dengan JSON | 228 |
| Pohon non-JSON | 717 |
| Total image | ~3,992+ |
| Sisi per pohon | mayoritas 4, sebagian 8 |

### Kelas kematangan

| Class | Deskripsi singkat |
|---|---|
| `B1` | merah, paling matang, posisi paling bawah |
| `B2` | transisi merah-hitam |
| `B3` | hitam penuh, di atas B2 |
| `B4` | paling kecil, berduri, paling atas |

Catatan penting: `B1 -> B4` bersifat ordinal. Ambiguitas utama tetap pada `B2` vs `B3`.

## Status Terkini

Status terbaru yang harus dipakai adalah override 2026-04-24. Jika ada file lama yang masih menyebut plateau `92%`, `93.86%`, `94.30%`, atau `v6_selector` sebagai best current method, anggap itu **outdated**.

### Benchmark JSON 228 Tree

| Rank | Method | Acc +/-1 | MAE |
|---:|---|---:|---:|
| 1 | `v9_selector` | **98.68%** | **0.2533** |
| 2 | `v9_b2_median_v6` | 96.49% | 0.2588 |
| 3 | `v6_selector` | 96.49% | 0.2632 |
| 4 | `v9_median_strong5` | 95.18% | 0.2390 |
| 5 | `stacking_bracketed_v7` | 94.30% | 0.2643 |

### Rekomendasi pakai metode

- JSON dengan GT:
  - pakai **`v9_selector`**
- Non-JSON tanpa GT:
  - prioritaskan `hybrid_vis_corr`, `side_coverage`, `stacking_density_v7`, `best_visibility_grid`, atau `visibility`
- Jangan asumsikan `v9_selector` otomatis terbaik untuk non-JSON, karena benchmark utamanya masih pada data JSON 228 tree

## Keputusan Riset

- Dedup adalah masalah utama; naive sum overcount sekitar `78.8%`.
- Label JSON konsisten; bottleneck utama bukan label noise, tetapi ambiguitas visual `B2/B3`.
- Pendekatan matching yang terlalu rigid cenderung gagal pada TXT noisy.
- Arah lanjutan tetap algorithmic:
  - multi-camera geometry
  - epipolar constraint
  - triangulation
  - topological / combinatorial matching
  - statistical ensemble tanpa learned weights

## Yang Tidak Dilakukan

Pendekatan berikut dianggap di luar scope:

- Siamese / CNN embedding
- learned thresholds via backprop
- MLP pada fitur bbox
- penggunaan learned feature matcher sebagai solusi dedup utama

## Menjalankan Script

Semua script dijalankan dari root workspace dan menulis output ke `reports/`.

```bash
# GT counting semua 953 pohon
python scripts/count_all_trees.py

# JSON-05 + JSON-01 audit
python scripts/count_gt_vs_naive.py

# Dedup research v1-v9
python scripts/dedup_research.py
python scripts/dedup_research_v2.py
python scripts/dedup_research_v3.py
python scripts/dedup_research_v4.py
python scripts/dedup_research_v5.py
python scripts/dedup_research_v6.py
python scripts/dedup_research_v7.py
python scripts/dedup_research_v8.py
python scripts/dedup_research_v9.py

# Final dedup semua pohon
python scripts/dedup_all_trees_final.py

# Perbandingan non-JSON + report
python scripts/dedup_nonjson_compare.py
```

## Struktur Repo

```text
json/                    228 file JSON dengan bunch-link antar-view
dataset/                 image dan label YOLO
scripts/                 seluruh script audit, counting, dan dedup research
reports/                 output CSV/MD dari setiap eksperimen
assets/                  visual summary untuk README / laporan
RESEARCH.md              dokumen riset panjang, baca Section 0 dulu
AGENTS.md                instruksi operasional repo
```

## Output Penting

| Lokasi | Isi |
|---|---|
| `reports/json_05/` | baseline GT vs naive pada 228 pohon JSON |
| `reports/full_gt_count/` | ringkasan count semua 953 pohon |
| `reports/dedup_research_v6/` | benchmark selector v6 |
| `reports/dedup_research_v7/` | baseline kuat untuk stacking |
| `reports/dedup_research_v8/` | eksplorasi lanjutan v8 |
| `reports/dedup_research_v9/` | benchmark terbaik saat ini untuk JSON |
| `reports/nonjson_dedup_compare/` | evaluasi metode pada non-JSON |
| `reports/nonjson_dedup_report.md` | laporan ringkas non-JSON |

## Schema JSON Ringkas

Setiap file JSON per pohon menyimpan:

- metadata pohon
- anotasi per sisi di `images`
- link tandan lintas sisi di `bunches`
- ground-truth count unik di `summary.by_class`

Contoh minimal:

```json
{
  "tree_id": "20260422-DAMIMAS-001",
  "images": {
    "sisi_1": {
      "annotations": [
        {"class_name": "B3", "bbox_yolo": [0.5, 0.5, 0.1, 0.2], "box_index": 0}
      ]
    }
  },
  "bunches": [
    {
      "bunch_id": 1,
      "class": "B3",
      "appearance_count": 2
    }
  ],
  "summary": {
    "by_class": {"B1": 1, "B2": 2, "B3": 5, "B4": 0}
  }
}
```

`summary.by_class` adalah ground truth count unik per kelas. Naive sum adalah jumlah seluruh bbox lintas sisi tanpa dedup.

## Cara Membaca Dokumen

- Mulai dari [AGENTS.md](/D:/Work/Assisten%20Dosen/research-method-dedup/AGENTS.md)
- Lalu baca [RESEARCH.md](/D:/Work/Assisten%20Dosen/research-method-dedup/RESEARCH.md), terutama Section 0
- Untuk hasil terbaru, prioritaskan folder `reports/dedup_research_v9`, lalu `v7`, `v8`, dan `nonjson_dedup_compare`

## Ringkasan Satu Kalimat

Repositori ini sudah melewati baseline heuristik lama; **benchmark JSON terbaik saat ini adalah `v9_selector` (98.68%)**, sedangkan untuk non-JSON tetap gunakan metode yang stabil terhadap noise koordinat dan noise kelas sampai ada validasi `v9` pada skenario non-JSON.
