"""
Audit: scan semua JSON di Brand-New-Dataset-YOLO/json/ cari bunch yang punya
2+ appearance dengan side_index sama. Secara fisik mustahil — satu bunch tidak
bisa muncul dua kali di sisi yang sama. Indikator wrap-around link bug atau
kesalahan annotator lain di `_confirmedLinks`.

Read-only. Output:
  reports/audit_same_side_dup/findings.csv   — 1 baris per bunch yang flagged
  reports/audit_same_side_dup/summary.md     — ringkas

Run:
    python scripts/audit_same_side_dup.py
"""

import json
import csv
from pathlib import Path
from collections import Counter, defaultdict

BASE     = Path(__file__).resolve().parent.parent
JSON_DIR = BASE / "Brand-New-Dataset-YOLO" / "json"
OUT_DIR  = BASE / "reports" / "audit_same_side_dup"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Bug report dari RA 2026-05-14
KNOWN_BUGS = {
    "DAMIMAS_A21B_0287", "DAMIMAS_A21B_0309", "DAMIMAS_A21B_0320",
    "DAMIMAS_A21B_0335", "DAMIMAS_A21B_0336", "DAMIMAS_A21B_0359",
    "DAMIMAS_A21B_0323", "DAMIMAS_A21B_0362",
}


def find_links_touching(links, side_index, box_index):
    """Return linkId yang menyentuh node (side_index, box_index)."""
    bid = f"b{box_index}"
    out = []
    for lk in links:
        if (lk["sideA"] == side_index and lk["bboxIdA"] == bid) or \
           (lk["sideB"] == side_index and lk["bboxIdB"] == bid):
            out.append(lk["linkId"])
    return out


rows = []
trees_flagged = set()
bunches_by_tree = Counter()

for jp in sorted(JSON_DIR.glob("*.json")):
    data = json.loads(jp.read_text(encoding="utf-8-sig"))
    tree_id = data.get("tree_id", jp.stem)
    bunches = data.get("bunches", [])
    links   = data.get("_confirmedLinks", [])

    for bunch in bunches:
        side_counts = Counter(a["side_index"] for a in bunch["appearances"])
        dup_sides = [s for s, c in side_counts.items() if c >= 2]
        if not dup_sides:
            continue
        for ds in dup_sides:
            box_idxs = [a["box_index"] for a in bunch["appearances"] if a["side_index"] == ds]
            link_chain = []
            for bi in box_idxs:
                link_chain.extend(find_links_touching(links, ds, bi))
            rows.append({
                "tree_id":     tree_id,
                "bunch_id":    bunch["bunch_id"],
                "bunch_class": bunch["class"],
                "n_appearances": bunch["appearance_count"],
                "dup_side_index": ds,
                "dup_side":    f"sisi_{ds + 1}",
                "box_indices": ",".join(str(b) for b in sorted(box_idxs)),
                "link_chain":  ",".join(sorted(set(link_chain))),
                "in_known_bug_report": tree_id in KNOWN_BUGS,
            })
            trees_flagged.add(tree_id)
            bunches_by_tree[tree_id] += 1

# Tulis CSV
csv_path = OUT_DIR / "findings.csv"
fields = [
    "tree_id", "bunch_id", "bunch_class", "n_appearances",
    "dup_side_index", "dup_side", "box_indices", "link_chain",
    "in_known_bug_report",
]
with csv_path.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

# Tulis summary
extras = sorted(trees_flagged - KNOWN_BUGS)
missing = sorted(KNOWN_BUGS - trees_flagged)

md_lines = [
    "# Audit: same-side duplicate appearances",
    "",
    f"- Total JSON scanned: {len(list(JSON_DIR.glob('*.json')))}",
    f"- Trees flagged: {len(trees_flagged)}",
    f"- Bunch records flagged: {len(rows)}",
    f"- Trees in known bug report: {len(KNOWN_BUGS)}",
    f"- Extra trees flagged (not in bug report): {len(extras)}",
    f"- Bug-report trees missing from findings: {len(missing)}",
    "",
    "## Trees flagged",
    "",
    "| tree_id | bunches flagged | known bug? |",
    "|---|---:|:---:|",
]
for tid in sorted(trees_flagged):
    mark = "yes" if tid in KNOWN_BUGS else "**NEW**"
    md_lines.append(f"| {tid} | {bunches_by_tree[tid]} | {mark} |")

if extras:
    md_lines += ["", "## Extras (not in bug report)", ""]
    for t in extras:
        md_lines.append(f"- {t}")
if missing:
    md_lines += ["", "## Missing (in bug report but not flagged)", ""]
    for t in missing:
        md_lines.append(f"- {t}")

(OUT_DIR / "summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

print(f"Trees flagged: {len(trees_flagged)}  (known: {len(KNOWN_BUGS)})")
print(f"Bunch records flagged: {len(rows)}")
print(f"Extras: {extras or 'none'}")
print(f"Missing: {missing or 'none'}")
print(f"CSV:     {csv_path}")
print(f"Summary: {OUT_DIR / 'summary.md'}")
