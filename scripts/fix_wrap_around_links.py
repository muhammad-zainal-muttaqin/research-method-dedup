"""
Fix wrap-around link bug pada `_confirmedLinks` di Brand-New-Dataset-YOLO/json/.

Bug source: annotator pilih box_index salah waktu tutup loop sisi_4 → sisi_1.
Akibat: bbox extra (kelas sama) ke-tarik ke bunch yang salah lewat connected
components → bunch jadi punya 2+ appearance dari same side_index (mustahil).

Fix per tree (5 simple — bad_link_ids tabel hardcoded dari bug report RA
2026-05-14):
  1. Backup file asli ke archive/json_pre_wrap_fix_2026-05-15/<tree_id>.json
     (sekali, skip kalau sudah ada — preserve original).
  2. Filter `_confirmedLinks` buang entry yg linkId ∈ bad_link_ids.
  3. Rebuild `bunches` via UnionFind dari filtered links.
     - Node = (side_index, box_index) untuk setiap annotation di setiap sisi.
     - Edge = setiap entry filtered links.
     - Komponen connected = satu bunch.
  4. Recompute `summary.total_unique_bunches`, `duplicates_linked`, `by_class`.
     `by_side` tidak berubah (raw bbox per sisi tetap).
  5. Bump version 2 → 3, append `metadata.fix_log`.
  6. Internal assertion: tidak ada bunch yg punya 2+ appearance same side.
     Kalau gagal → JANGAN tulis file.

3 hard trees (0335, 0323, 0362): bad_link_ids kosong di tabel → script print
warning dan SKIP. Diisi nanti setelah RA kasih keputusan.

Run:
    python scripts/fix_wrap_around_links.py                  # apply 5 simple
    python scripts/fix_wrap_around_links.py --dry-run        # print diff doang
    python scripts/fix_wrap_around_links.py --only DAMIMAS_A21B_0287
"""

import argparse
import json
import shutil
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

BASE        = Path(__file__).resolve().parent.parent
JSON_DIR    = BASE / "Brand-New-Dataset-YOLO" / "json"
BACKUP_DIR  = BASE / "archive" / "json_pre_wrap_fix_2026-05-15"
NAMES       = ["B1", "B2", "B3", "B4"]
BY_CLASS_KEYS = ["B1", "B2", "B3", "B4", "other"]
SIDE_KEYS   = ["sisi_1", "sisi_2", "sisi_3", "sisi_4"]
FIX_DATE    = "2026-05-15"

# tree_id -> list of bad linkId yang harus dihapus
# Sumber: bug-report-duplicate-side-links.md (RA 2026-05-14)
FIX_TABLE = {
    "DAMIMAS_A21B_0287": ["lnk-3"],
    "DAMIMAS_A21B_0309": ["lnk-2"],
    "DAMIMAS_A21B_0320": ["lnk-4"],
    "DAMIMAS_A21B_0336": ["lnk-3"],
    "DAMIMAS_A21B_0359": ["lnk-6"],
    # Hard — butuh decision RA, slot kosong placeholder
    "DAMIMAS_A21B_0335": [],
    "DAMIMAS_A21B_0323": [],
    "DAMIMAS_A21B_0362": [],
}


# ── UnionFind ────────────────────────────────────────────────────────────────

class UF:
    def __init__(self):
        self.parent = {}

    def add(self, x):
        if x not in self.parent:
            self.parent[x] = x

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        self.add(a); self.add(b)
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def parse_bid(s: str) -> int:
    """`b0` -> 0, `b12` -> 12."""
    if not s.startswith("b"):
        raise ValueError(f"unexpected bbox id format: {s!r}")
    return int(s[1:])


# ── Rebuild bunches ──────────────────────────────────────────────────────────

def rebuild_bunches(data: dict, kept_links: list[dict]) -> list[dict]:
    """Bangun ulang `bunches[]` dari connected components atas filtered links.

    Setiap (side_index, box_index) dari setiap annotations adalah node. Edge
    adalah setiap entry di kept_links. Bunch_id assignment deterministik:
    sort component by (min side_index, min box_index).
    """
    # 1. Kumpulkan semua nodes dari annotations
    images = data["images"]
    nodes = []          # list of (side_index, box_index, class_name, bbox_pixel, side_key)
    node_lookup = {}    # (side_idx, box_idx) -> node tuple index
    for sk in SIDE_KEYS:
        if sk not in images:
            continue
        side = images[sk]
        sidx = side["side_index"]
        for ann in side["annotations"]:
            bidx = ann["box_index"]
            nodes.append({
                "side":       sk,
                "side_index": sidx,
                "box_index":  bidx,
                "class_name": ann["class_name"],
                "bbox_pixel": ann["bbox_pixel"],
            })
            node_lookup[(sidx, bidx)] = len(nodes) - 1

    # 2. UnionFind
    uf = UF()
    for n in nodes:
        uf.add((n["side_index"], n["box_index"]))
    for lk in kept_links:
        a = (lk["sideA"], parse_bid(lk["bboxIdA"]))
        b = (lk["sideB"], parse_bid(lk["bboxIdB"]))
        if a not in node_lookup or b not in node_lookup:
            raise ValueError(
                f"link {lk['linkId']} references non-existent node: {a} or {b}"
            )
        uf.union(a, b)

    # 3. Group nodes per root
    components = defaultdict(list)
    for n in nodes:
        root = uf.find((n["side_index"], n["box_index"]))
        components[root].append(n)

    # 4. Sort components by deterministic key, build bunches
    def comp_sort_key(comp):
        return (
            min(n["side_index"] for n in comp),
            min(n["box_index"] for n in comp if n["side_index"] == min(m["side_index"] for m in comp)),
        )

    sorted_comps = sorted(components.values(), key=comp_sort_key)

    bunches = []
    for i, comp in enumerate(sorted_comps, start=1):
        comp_sorted = sorted(comp, key=lambda n: (n["side_index"], n["box_index"]))
        classes = [n["class_name"] for n in comp_sorted]
        class_counter = Counter(classes)
        # tie-break: ordinal terbesar (B4 > B3 > B2 > B1) — sesuai konvensi
        # domain bahwa kelas matang lebih kritikal. No `class_mismatch: true`
        # exists in corpus, so this convention is being established here for
        # the rare case kalau pernah terjadi.
        max_count = max(class_counter.values())
        tied = [c for c, cnt in class_counter.items() if cnt == max_count]
        bunch_class = max(tied)  # alphabetical max == ordinal max for B1..B4
        mismatch = len(set(classes)) > 1

        bunch = {
            "bunch_id":         i,
            "class":            bunch_class,
            "class_mismatch":   mismatch,
            "appearance_count": len(comp_sorted),
            "appearances": [
                {
                    "side":       n["side"],
                    "side_index": n["side_index"],
                    "box_index":  n["box_index"],
                    "class_name": n["class_name"],
                    "bbox_pixel": n["bbox_pixel"],
                }
                for n in comp_sorted
            ],
        }
        bunches.append(bunch)
    return bunches


def assert_no_same_side_dup(bunches: list[dict], tree_id: str):
    for b in bunches:
        sides = [a["side_index"] for a in b["appearances"]]
        dup = [s for s, c in Counter(sides).items() if c >= 2]
        if dup:
            raise AssertionError(
                f"{tree_id}: bunch {b['bunch_id']} masih punya same-side dup "
                f"di side_index {dup}; fix table salah / butuh review manual"
            )


def recompute_summary(data: dict, bunches: list[dict]) -> dict:
    images = data["images"]
    total_detections = sum(images[sk]["bbox_count"] for sk in SIDE_KEYS if sk in images)
    by_class = Counter()
    for b in bunches:
        c = b["class"] if b["class"] in BY_CLASS_KEYS else "other"
        by_class[c] += 1
    by_side = {sk: images[sk]["bbox_count"] for sk in SIDE_KEYS if sk in images}
    return {
        "total_unique_bunches": len(bunches),
        "total_detections":     total_detections,
        "duplicates_linked":    total_detections - len(bunches),
        "by_class":             {k: by_class.get(k, 0) for k in BY_CLASS_KEYS},
        "by_side":              by_side,
    }


# ── Per-tree processor ──────────────────────────────────────────────────────

def process_tree(tree_id: str, bad_link_ids: list[str], dry_run: bool):
    json_path = JSON_DIR / f"{tree_id}.json"
    if not json_path.is_file():
        print(f"[SKIP] {tree_id}: file not found")
        return False

    if not bad_link_ids:
        print(f"[SKIP] {tree_id}: bad_link_ids belum diisi (hard tree, butuh RA)")
        return False

    data = json.loads(json_path.read_text(encoding="utf-8-sig"))
    links = data.get("_confirmedLinks", [])
    bad_set = set(bad_link_ids)
    kept = [lk for lk in links if lk["linkId"] not in bad_set]
    removed = [lk for lk in links if lk["linkId"] in bad_set]

    if len(removed) != len(bad_link_ids):
        found = {lk["linkId"] for lk in removed}
        missing = bad_set - found
        raise AssertionError(
            f"{tree_id}: link(s) tidak ditemukan di _confirmedLinks: {sorted(missing)}"
        )

    new_bunches = rebuild_bunches(data, kept)
    assert_no_same_side_dup(new_bunches, tree_id)
    new_summary = recompute_summary(data, new_bunches)

    old_n = len(data.get("bunches", []))
    new_n = len(new_bunches)
    old_by_class = data.get("summary", {}).get("by_class", {})
    new_by_class = new_summary["by_class"]

    print(f"[FIX] {tree_id}")
    print(f"      removed_links: {[lk['linkId'] for lk in removed]}")
    print(f"      bunches: {old_n} -> {new_n}  (delta {new_n - old_n:+d})")
    print(f"      by_class: {dict(old_by_class)} -> {dict(new_by_class)}")

    if dry_run:
        print(f"      [DRY-RUN] no file written")
        return True

    # Backup (sekali)
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    backup_path = BACKUP_DIR / f"{tree_id}.json"
    if not backup_path.exists():
        shutil.copy2(json_path, backup_path)
        print(f"      backup: {backup_path.relative_to(BASE)}")
    else:
        print(f"      backup exists, skip: {backup_path.relative_to(BASE)}")

    # Update payload
    data["version"] = 3
    data["_confirmedLinks"] = kept
    data["bunches"] = new_bunches
    data["summary"] = new_summary
    fix_log = data.get("metadata", {}).setdefault("fix_log", [])
    fix_log.append({
        "date":           FIX_DATE,
        "action":         "removed_wrap_around_links",
        "removed":        bad_link_ids,
        "bunches_before": old_n,
        "bunches_after":  new_n,
    })

    json_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"      written: {json_path.relative_to(BASE)}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="print diff tanpa tulis file")
    ap.add_argument("--only", action="append", default=None,
                    help="batasi ke tree_id tertentu (boleh multiple)")
    args = ap.parse_args()

    targets = args.only or list(FIX_TABLE.keys())
    fixed = 0
    skipped = 0
    for tid in targets:
        if tid not in FIX_TABLE:
            print(f"[SKIP] {tid}: tidak ada di FIX_TABLE")
            skipped += 1
            continue
        ok = process_tree(tid, FIX_TABLE[tid], args.dry_run)
        if ok:
            fixed += 1
        else:
            skipped += 1

    print()
    print(f"Done. fixed={fixed} skipped={skipped} dry_run={args.dry_run}")


if __name__ == "__main__":
    main()
