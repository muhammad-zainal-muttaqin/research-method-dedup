"""
GT-based bunch counting untuk SEMUA pohon (953 total).

Sumber data:
  - Pohon dengan JSON (228)  → summary.by_class dari JSON (dedup akurat)
  - Pohon tanpa JSON  (725)  → naive sum dari label YOLO TXT (overcounting)

Output (reports/full_gt_count/):
  count_all_trees.csv      — 1 baris per pohon
  summary_by_domain.csv    — DAMIMAS vs LONSUM
  summary_by_split.csv     — train / val / test
  summary.md               — laporan ringkas

Run dari workspace root:
    python scripts/count_all_trees.py
"""

import json
import csv
from pathlib import Path
from collections import defaultdict, Counter

BASE      = Path(__file__).resolve().parent.parent
LABEL_DIR = BASE / "dataset" / "labels"
JSON_DIR  = BASE / "json"
OUT_DIR   = BASE / "reports" / "full_gt_count"
NAMES     = ["B1", "B2", "B3", "B4"]
CLASS_MAP = {0: "B1", 1: "B2", 2: "B3", 3: "B4"}

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── 1. Bangun index semua pohon dari label TXT ────────────────────────────────

# tree_id → {split, sides: set, files: {side_no: Path}}
trees = defaultdict(lambda: {"split": None, "sides": set(), "files": {}})

for split in ("train", "val", "test"):
    for f in sorted((LABEL_DIR / split).glob("*.txt")):
        stem = f.stem  # e.g. DAMIMAS_A21B_0001_1
        parts = stem.rsplit("_", 1)
        if len(parts) != 2:
            continue
        tree_id, side_str = parts
        try:
            side_no = int(side_str)
        except ValueError:
            continue
        trees[tree_id]["split"] = split
        trees[tree_id]["sides"].add(side_no)
        trees[tree_id]["files"][side_no] = f


# ── 2. Bangun index JSON (tree_name → data) ───────────────────────────────────

json_index = {}
for jp in sorted(JSON_DIR.glob("*.json")):
    data = json.loads(jp.read_text(encoding="utf-8"))
    name = data.get("tree_name", data.get("tree_id", jp.stem))
    json_index[name] = data


# ── 3. Helper: hitung dari TXT ────────────────────────────────────────────────

def count_from_txt(file_paths) -> dict:
    counts = Counter()
    for fp in file_paths:
        for line in fp.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            cls_id = int(line.split()[0])
            cls_name = CLASS_MAP.get(cls_id)
            if cls_name:
                counts[cls_name] += 1
    return {c: counts.get(c, 0) for c in NAMES}


# ── 4. Proses setiap pohon ────────────────────────────────────────────────────

rows = []

for tree_id, info in sorted(trees.items()):
    domain = "DAMIMAS" if "DAMIMAS" in tree_id else ("LONSUM" if "LONSUM" in tree_id else "OTHER")
    split  = info["split"]
    n_sides = len(info["sides"])

    if tree_id in json_index:
        # Pakai JSON dedup count
        by_class = json_index[tree_id]["summary"]["by_class"]
        counts   = {c: by_class.get(c, 0) for c in NAMES}
        source   = "json_dedup"
    else:
        # Naive sum dari TXT
        counts = count_from_txt(info["files"].values())
        source = "txt_naive"

    row = {
        "tree_id":  tree_id,
        "domain":   domain,
        "split":    split,
        "n_sides":  n_sides,
        "has_json": tree_id in json_index,
        "source":   source,
    }
    for c in NAMES:
        row[c] = counts[c]
    row["total"] = sum(counts[c] for c in NAMES)
    rows.append(row)


# ── 5. Tulis count_all_trees.csv ─────────────────────────────────────────────

fields = ["tree_id", "domain", "split", "n_sides", "has_json", "source",
          "B1", "B2", "B3", "B4", "total"]
with open(OUT_DIR / "count_all_trees.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)


# ── 6. Aggregate helpers ──────────────────────────────────────────────────────

def aggregate(subset, label_col, label_val):
    sub = [r for r in subset if r[label_col] == label_val]
    if not sub:
        return None
    n_json = sum(1 for r in sub if r["has_json"])
    result = {
        label_col:     label_val,
        "n_trees":     len(sub),
        "n_json":      n_json,
        "n_txt_naive": len(sub) - n_json,
    }
    for c in NAMES:
        result[c] = sum(r[c] for r in sub)
    result["total"] = sum(r["total"] for r in sub)
    return result


# ── 7. summary_by_domain.csv ─────────────────────────────────────────────────

domain_rows = []
for dom in ("DAMIMAS", "LONSUM"):
    agg = aggregate(rows, "domain", dom)
    if agg:
        domain_rows.append(agg)

dom_fields = ["domain", "n_trees", "n_json", "n_txt_naive"] + NAMES + ["total"]
with open(OUT_DIR / "summary_by_domain.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=dom_fields)
    w.writeheader()
    w.writerows(domain_rows)


# ── 8. summary_by_split.csv ──────────────────────────────────────────────────

split_rows = []
for sp in ("train", "val", "test"):
    agg = aggregate(rows, "split", sp)
    if agg:
        split_rows.append(agg)

sp_fields = ["split", "n_trees", "n_json", "n_txt_naive"] + NAMES + ["total"]
with open(OUT_DIR / "summary_by_split.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=sp_fields)
    w.writeheader()
    w.writerows(split_rows)


# ── 9. summary.md ─────────────────────────────────────────────────────────────

total_trees  = len(rows)
n_json_trees = sum(1 for r in rows if r["has_json"])
n_txt_trees  = total_trees - n_json_trees
n_damimas    = sum(1 for r in rows if r["domain"] == "DAMIMAS")
n_lonsum     = sum(1 for r in rows if r["domain"] == "LONSUM")
n_8side      = sum(1 for r in rows if r["n_sides"] == 8)

total_bunches_all   = sum(r["total"] for r in rows)
total_bunches_json  = sum(r["total"] for r in rows if r["has_json"])
total_bunches_naive = sum(r["total"] for r in rows if not r["has_json"])

# Per-class total
per_class_all   = {c: sum(r[c] for r in rows) for c in NAMES}
per_class_json  = {c: sum(r[c] for r in rows if r["has_json"]) for c in NAMES}
per_class_naive = {c: sum(r[c] for r in rows if not r["has_json"]) for c in NAMES}

# Overcounting context from JSON-05 (sudah diketahui: ~78.8%)
OVERCOUNT_RATE = 0.788  # dari JSON-05

md = f"""# Laporan GT Bunch Counting — Semua Pohon
**Tanggal:** 2026-04-23
**Dataset:** DAMIMAS + LONSUM (seluruh data GT yang tersedia)

---

## 1. Ringkasan Dataset

| Item | Nilai |
|------|-------|
| Total pohon diproses | **{total_trees}** |
| Domain DAMIMAS | {n_damimas} |
| Domain LONSUM | {n_lonsum} |
| Pohon 4-sisi | {total_trees - n_8side} |
| Pohon 8-sisi | {n_8side} |
| Pohon dengan JSON (dedup akurat) | **{n_json_trees}** |
| Pohon tanpa JSON (naive sum) | **{n_txt_trees}** |

---

## 2. Jumlah Tandan per Kelas (Seluruh Pohon)

> Pohon ber-JSON: hitungan **unik/dedup** (akurat).
> Pohon non-JSON: hitungan **naif** (tanpa dedup — estimasi overcounting ~79%).

| Kelas | JSON-Dedup ({n_json_trees} pohon) | Naive-Sum ({n_txt_trees} pohon) | Total |
|-------|---:|---:|---:|
| B1 | {per_class_json['B1']:,} | {per_class_naive['B1']:,} | {per_class_all['B1']:,} |
| B2 | {per_class_json['B2']:,} | {per_class_naive['B2']:,} | {per_class_all['B2']:,} |
| B3 | {per_class_json['B3']:,} | {per_class_naive['B3']:,} | {per_class_all['B3']:,} |
| B4 | {per_class_json['B4']:,} | {per_class_naive['B4']:,} | {per_class_all['B4']:,} |
| **TOTAL** | **{total_bunches_json:,}** | **{total_bunches_naive:,}** | **{total_bunches_all:,}** |

### Estimasi True Count untuk Pohon Non-JSON
Berdasarkan hasil JSON-05 (overcounting rate 78.8%), estimasi tandan unik sesungguhnya
untuk {n_txt_trees} pohon non-JSON:

| Kelas | Naive Count | Est. Unique (÷1.788) |
|-------|---:|---:|
| B1 | {per_class_naive['B1']:,} | {int(per_class_naive['B1'] / (1 + OVERCOUNT_RATE)):,} |
| B2 | {per_class_naive['B2']:,} | {int(per_class_naive['B2'] / (1 + OVERCOUNT_RATE)):,} |
| B3 | {per_class_naive['B3']:,} | {int(per_class_naive['B3'] / (1 + OVERCOUNT_RATE)):,} |
| B4 | {per_class_naive['B4']:,} | {int(per_class_naive['B4'] / (1 + OVERCOUNT_RATE)):,} |
| **TOTAL** | **{total_bunches_naive:,}** | **{int(total_bunches_naive / (1 + OVERCOUNT_RATE)):,}** |

---

## 3. Breakdown per Domain

"""

for dr in domain_rows:
    dom = dr["domain"]
    md += f"### {dom} ({dr['n_trees']} pohon)\n\n"
    md += f"| Kelas | Count | % |\n|-------|------:|---:|\n"
    dom_total = dr["total"]
    for c in NAMES:
        pct = dr[c] / max(dom_total, 1) * 100
        md += f"| {c} | {dr[c]:,} | {pct:.1f}% |\n"
    md += f"| **Total** | **{dom_total:,}** | 100% |\n"
    md += f"\n- Pohon ber-JSON: {dr['n_json']} | Non-JSON: {dr['n_txt_naive']}\n\n"

md += """---

## 4. Breakdown per Split

"""

for sr in split_rows:
    sp = sr["split"]
    md += f"### Split: {sp.upper()} ({sr['n_trees']} pohon)\n\n"
    md += f"| Kelas | Count |\n|-------|------:|\n"
    for c in NAMES:
        md += f"| {c} | {sr[c]:,} |\n"
    md += f"| **Total** | **{sr['total']:,}** |\n"
    md += f"\n- Pohon ber-JSON: {sr['n_json']} | Non-JSON: {sr['n_txt_naive']}\n\n"

md += f"""---

## 5. Catatan Metodologi

- **Sumber data:** Ground truth label (bukan prediksi model) — sesuai arahan dosen
- **JSON dedup:** 228 pohon sudah di-link manual antar sisi → hitungan tandan unik akurat
- **TXT naive:** 725 pohon dihitung langsung dari file label YOLO → setiap penampakan dihitung 1×
- **Overcounting rate** (dari JSON-05): naive sum rata-rata **78.8% lebih tinggi** dari count unik
- **Pohon 8-sisi ({n_8side} pohon):** data baru dengan 8 sudut foto — dihitung naive sum (belum ada JSON)
- File detail per pohon tersimpan di: `reports/full_gt_count/count_all_trees.csv`

---

## 6. File Output

| File | Isi |
|------|-----|
| `count_all_trees.csv` | 953 baris — detail per pohon |
| `summary_by_domain.csv` | Agregat DAMIMAS vs LONSUM |
| `summary_by_split.csv` | Agregat train / val / test |
| `summary.md` | Dokumen ini |
"""

(OUT_DIR / "summary.md").write_text(md, encoding="utf-8")


# ── 10. Print ringkasan ke terminal ──────────────────────────────────────────

print(f"\n{'='*60}")
print(f"GT Bunch Counting — Semua Pohon")
print(f"{'='*60}")
print(f"Total pohon   : {total_trees} (DAMIMAS: {n_damimas}, LONSUM: {n_lonsum})")
print(f"Pohon 8-sisi  : {n_8side}")
print(f"Pohon ber-JSON: {n_json_trees}  (dedup akurat)")
print(f"Pohon non-JSON: {n_txt_trees}  (naive sum)")
print()
print(f"{'Kelas':<6} {'JSON-Dedup':>12} {'Naive-Sum':>12} {'Total':>10}")
print("-" * 44)
for c in NAMES:
    print(f"{c:<6} {per_class_json[c]:>12,} {per_class_naive[c]:>12,} {per_class_all[c]:>10,}")
print("-" * 44)
print(f"{'TOTAL':<6} {total_bunches_json:>12,} {total_bunches_naive:>12,} {total_bunches_all:>10,}")
print()
print(f"Output tersimpan di: reports/full_gt_count/")
