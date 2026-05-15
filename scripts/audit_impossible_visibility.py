"""
Audit: deteksi bunch yang ke-link di KOMBINASI SISI YG TIDAK MUNGKIN
secara geometri.

Rule (dari domain knowledge RA, 2026-05-15):

Bunch hanya bisa terlihat dari sisi yang ber-adjacent dgn HOME side
(posisi fisik bunch di pohon). Distance antar sisi = circular hop count.

  - Pohon 4-sisi: max circular distance = 1 (≤ 3 sisi visible total)
    Contoh: home=sisi_1 → visible {sisi_4, sisi_1, sisi_2}, mustahil sisi_3.
  - Pohon 8-sisi: max circular distance = 2 (≤ 5 sisi visible total)
    Contoh: home=sisi_3 → visible {sisi_1, sisi_2, sisi_3, sisi_4, sisi_5},
    mustahil sisi_6, sisi_7, sisi_8.

Bunch wajib punya appearance di home side (camera di posisi asal pasti
melihat). Validity test: cari candidate home di antara appearance sides
yg semua appearance lain ada di dalam max_dist hop.

Severity:
  - violation: tidak ada candidate home valid (mustahil geometri)
  - warn:      valid, tapi jumlah sisi > normal max (borderline, full reach)
  - ok:        valid, ≤ normal max (tidak diflag)

Read-only. Output:
  reports/audit_impossible_visibility/findings.csv
  reports/audit_impossible_visibility/summary.md
"""

import json
import csv
from pathlib import Path
from collections import Counter

BASE     = Path(__file__).resolve().parent.parent
JSON_DIR = BASE / "Brand-New-Dataset-YOLO" / "json"
OUT_DIR  = BASE / "reports" / "audit_impossible_visibility"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# n_sides_total -> (max_dist, max_normal_sides, max_possible_sides)
# max_dist: hop limit dari home (geometric)
# max_normal_sides: normal visible count (home + immediate neighbors)
# max_possible_sides: hard ceiling — visual override allowed up to here
# Updated 2026-05-16 per RA visual validation: 8-side bunches CAN span
# up to 6 sides (large/prominent bunches with wider camera reach).
LIMITS = {
    4: {"max_dist": 1, "max_normal_sides": 2, "max_possible_sides": 3},
    8: {"max_dist": 3, "max_normal_sides": 4, "max_possible_sides": 6},
}


def circular_dist(a, b, N):
    d = abs(a - b)
    return min(d, N - d)


def find_valid_home(sides, N, max_dist):
    """Cari candidate home di antara appearance sides yg semua sides lain
    dalam max_dist. Return home idx atau None."""
    for h in sides:
        if all(circular_dist(h, s, N) <= max_dist for s in sides):
            return h
    return None


def find_offending_sides(sides, N, max_dist):
    """Untuk bunch invalid, cari sisi yg paling 'jauh' dari klaster utama.
    Pick home yg minimize jumlah sisi melanggar; return offending side list."""
    best_offenders = sides
    for h in sides:
        offenders = [s for s in sides if circular_dist(h, s, N) > max_dist]
        if len(offenders) < len(best_offenders):
            best_offenders = offenders
    return best_offenders


def find_links_for_bunch(links, bunch_appearances):
    nodes = {(a["side_index"], f"b{a['box_index']}") for a in bunch_appearances}
    out = []
    for lk in links:
        a = (lk["sideA"], lk["bboxIdA"])
        b = (lk["sideB"], lk["bboxIdB"])
        if a in nodes or b in nodes:
            out.append(lk["linkId"])
    return out


rows = []
trees_violation = set()
trees_warn      = set()
trees_no_limit  = []

for jp in sorted(JSON_DIR.glob("*.json")):
    data = json.loads(jp.read_text(encoding="utf-8-sig"))
    tree_id = data.get("tree_id", jp.stem)
    bunches = data.get("bunches", [])
    links   = data.get("_confirmedLinks", [])
    images  = data.get("images", {})
    N = len(images)

    if N not in LIMITS:
        trees_no_limit.append((tree_id, N))
        continue

    cfg = LIMITS[N]

    for bunch in bunches:
        sides = sorted({a["side_index"] for a in bunch["appearances"]})
        if len(sides) <= 1:
            continue  # singleton — selalu valid

        valid_home = find_valid_home(sides, N, cfg["max_dist"])
        n = len(sides)

        if valid_home is None:
            severity = "violation"
            offenders = find_offending_sides(sides, N, cfg["max_dist"])
            offender_str = ",".join(f"sisi_{s+1}" for s in offenders)
            reason = f"no_valid_home_max_dist_{cfg['max_dist']}"
        elif n > cfg["max_normal_sides"]:
            severity = "warn"
            offender_str = ""
            reason = f"valid_but_n_sides_{n}_exceeds_normal_{cfg['max_normal_sides']}"
        else:
            continue  # ok, skip

        record = {
            "tree_id":          tree_id,
            "bunch_id":         bunch["bunch_id"],
            "bunch_class":      bunch["class"],
            "n_sides_total":    N,
            "n_sides_bunch":    n,
            "max_dist_allowed": cfg["max_dist"],
            "max_normal":       cfg["max_normal_sides"],
            "max_possible":     cfg["max_possible_sides"],
            "valid_home":       f"sisi_{valid_home+1}" if valid_home is not None else "",
            "appearance_sides": ",".join(f"sisi_{s+1}" for s in sides),
            "offending_sides":  offender_str,
            "reason":           reason,
            "link_chain":       ",".join(sorted(set(find_links_for_bunch(links, bunch["appearances"])))),
            "severity":         severity,
        }
        rows.append(record)
        if severity == "violation":
            trees_violation.add(tree_id)
        else:
            trees_warn.add(tree_id)

# CSV
fields = [
    "tree_id", "bunch_id", "bunch_class",
    "n_sides_total", "n_sides_bunch", "max_dist_allowed", "max_normal", "max_possible",
    "valid_home", "appearance_sides", "offending_sides", "reason", "link_chain",
    "severity",
]
with (OUT_DIR / "findings.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

violations = [r for r in rows if r["severity"] == "violation"]
warnings_  = [r for r in rows if r["severity"] == "warn"]

# Summary
md = [
    "# Audit: impossible bunch visibility (geometric adjacency rule)",
    "",
    "## Rule",
    "",
    "Bunch wajib punya appearance di **home side** (posisi fisik bunch).",
    "Appearance lain harus dalam circular distance ≤ `max_dist` dari home.",
    "",
    "| n_sides_total | max_dist (hop) | normal max sides | hard max sides |",
    "|---:|---:|---:|---:|",
]
for k in sorted(LIMITS.keys()):
    cfg = LIMITS[k]
    md.append(f"| {k} | {cfg['max_dist']} | {cfg['max_normal_sides']} | {cfg['max_possible_sides']} |")

md += [
    "",
    "## Results",
    "",
    f"- JSON scanned: {len(list(JSON_DIR.glob('*.json')))}",
    f"- Trees with violation: **{len(trees_violation)}**",
    f"- Trees with warning only: {len(trees_warn - trees_violation)}",
    f"- Bunches violation: **{len(violations)}**",
    f"- Bunches warning: {len(warnings_)}",
    f"- Trees skipped (n_sides not in {sorted(LIMITS.keys())}): {len(trees_no_limit)}",
    "",
    "## Violations",
    "",
    "Bunch yg tidak punya geometric valid home — secara fisik mustahil.",
    "",
]
if violations:
    md += [
        "| tree_id | bunch | class | sides_bunch | sides total | appearance_sides | offending |",
        "|---|---:|:---:|:---:|:---:|---|---|",
    ]
    for r in violations[:80]:
        md.append(
            f"| {r['tree_id']} | {r['bunch_id']} | {r['bunch_class']} | "
            f"{r['n_sides_bunch']} | {r['n_sides_total']} | "
            f"{r['appearance_sides']} | {r['offending_sides']} |"
        )
    if len(violations) > 80:
        md.append(f"\n(... {len(violations) - 80} more — see findings.csv)")
else:
    md.append("(none)")

md += [
    "",
    "## Warnings",
    "",
    "Bunch valid (geometric OK) tapi pakai full reach — borderline normal.",
    "",
]
if warnings_:
    md += [
        "| tree_id | bunch | class | sides_bunch | sides total | appearance_sides | valid_home |",
        "|---|---:|:---:|:---:|:---:|---|:---:|",
    ]
    for r in warnings_[:80]:
        md.append(
            f"| {r['tree_id']} | {r['bunch_id']} | {r['bunch_class']} | "
            f"{r['n_sides_bunch']} | {r['n_sides_total']} | "
            f"{r['appearance_sides']} | {r['valid_home']} |"
        )
    if len(warnings_) > 80:
        md.append(f"\n(... {len(warnings_) - 80} more — see findings.csv)")
else:
    md.append("(none)")

(OUT_DIR / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

print(f"VIOLATIONS: {len(violations)} bunches in {len(trees_violation)} trees")
print(f"WARNINGS:   {len(warnings_)} bunches in {len(trees_warn - trees_violation)} trees")
print()
print("Top violations:")
for r in violations[:10]:
    print(f"  {r['tree_id']} bunch#{r['bunch_id']} ({r['bunch_class']}): "
          f"{r['n_sides_bunch']}/{r['n_sides_total']} sides — {r['appearance_sides']} "
          f"(offending: {r['offending_sides']})")
print()
print(f"CSV:     {OUT_DIR / 'findings.csv'}")
print(f"Summary: {OUT_DIR / 'summary.md'}")
