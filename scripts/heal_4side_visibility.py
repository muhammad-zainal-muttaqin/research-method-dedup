"""
Auto-heal violations geometric visibility di pohon 4-sisi.

Bug: bunch ke-link di 4/4 sisi (mustahil — max distance dari home = 1, jadi
max 3 sisi visible).

Heal heuristic:
  1. Untuk tiap bunch violating: pick HOME = appearance side dgn bbox area
     terbesar (bunch terlihat paling prominent dari posisi terdekat).
  2. Identify OFFENDING side = side dgn circular distance > 1 dari home
     (untuk 4-sisi: opposite side, distance 2).
  3. Drop semua link yg menyentuh node (offending_side, *) — split offending
     box jadi standalone bunch baru.
  4. Rebuild bunches via UnionFind dari kept links.

Caveat heuristic: largest-bbox-area = home tidak selalu akurat. Spot-check
hasil sebelum commit. --dry-run untuk preview.

Output ke folder dataset (in-place edit) + backup ke
`archive/json_pre_visibility_heal_4side_2026-05-16/`.

Run:
    python scripts/heal_4side_visibility.py --dry-run     # preview
    python scripts/heal_4side_visibility.py               # apply
    python scripts/heal_4side_visibility.py --only DAMIMAS_A21B_0002
"""

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

BASE        = Path(__file__).resolve().parent.parent
JSON_DIR    = BASE / "Brand-New-Dataset-YOLO" / "json"
BACKUP_DIR  = BASE / "archive" / "json_pre_visibility_heal_4side_2026-05-16"
SIDE_KEYS_4 = ["side_1", "side_2", "side_3", "side_4"]
SIDE_KEYS_8 = SIDE_KEYS_4 + ["side_5", "side_6", "side_7", "side_8"]
BY_CLASS    = ["B1", "B2", "B3", "B4", "other"]
FIX_DATE    = "2026-05-16"

# Rule per CLAUDE.md "Ground-truth Validation Rules"
LIMITS = {
    4: {"max_dist": 1, "side_keys": SIDE_KEYS_4},
    8: {"max_dist": 3, "side_keys": SIDE_KEYS_8},
}


class UF:
    def __init__(self): self.parent = {}
    def add(self, x):
        if x not in self.parent: self.parent[x] = x
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def union(self, a, b):
        self.add(a); self.add(b)
        ra, rb = self.find(a), self.find(b)
        if ra != rb: self.parent[ra] = rb


def parse_bid(s): return int(s[1:])

def circular_dist(a, b, N): d = abs(a - b); return min(d, N - d)


def bbox_area(bbox_pixel):
    x1, y1, x2, y2 = bbox_pixel
    return abs((x2 - x1) * (y2 - y1))


def find_violation_bunches(data):
    """Return list of (bunch_idx, sides_set, home_side, offending_sides)
    for bunches violating geometric rule."""
    images = data["images"]
    N = len(images)
    if N not in LIMITS:
        return []
    max_dist = LIMITS[N]["max_dist"]
    out = []
    for i, bunch in enumerate(data["bunches"]):
        sides = sorted({a["side_index"] for a in bunch["appearances"]})
        if len(sides) <= 1:
            continue
        # Try every appearance side as home; if any home gives all sides
        # within max_dist, bunch is OK.
        valid = any(
            all(circular_dist(h, s, N) <= max_dist for s in sides)
            for h in sides
        )
        if valid:
            continue
        # Pick home = side dgn appearance bbox terbesar
        home = max(
            sides,
            key=lambda s: max(
                bbox_area(a["bbox_pixel"])
                for a in bunch["appearances"] if a["side_index"] == s
            ),
        )
        offenders = [s for s in sides if circular_dist(home, s, N) > max_dist]
        out.append((i, sides, home, offenders))
    return out


def heal_tree(data):
    """Apply heal heuristic. Mutate data in place. Return list of action records."""
    images = data["images"]
    N = len(images)
    if N not in LIMITS:
        return [], 0, 0
    side_keys = LIMITS[N]["side_keys"]
    max_dist = LIMITS[N]["max_dist"]

    violations = find_violation_bunches(data)
    if not violations:
        return [], len(data["bunches"]), len(data["bunches"])

    # Determine which (side_index, box_index) nodes harus DI-isolasi (drop links).
    # Untuk tiap violation: drop links yg menyentuh node (offender_side, *) untuk
    # box_index yg appear di bunch tsb.
    nodes_to_isolate = set()
    actions = []
    for bunch_idx, sides, home, offenders in violations:
        bunch = data["bunches"][bunch_idx]
        for off in offenders:
            for app in bunch["appearances"]:
                if app["side_index"] == off:
                    nodes_to_isolate.add((off, app["box_index"]))
        actions.append({
            "bunch_id":  bunch["bunch_id"],
            "class":     bunch["class"],
            "home_side": f"side_{home+1}",
            "offending": [f"side_{s+1}" for s in offenders],
            "isolated_nodes": [
                f"s{off+1}/b{app['box_index']}"
                for off in offenders
                for app in bunch["appearances"]
                if app["side_index"] == off
            ],
        })

    # Filter links: drop yg sentuh isolated node
    links = data["_confirmedLinks"]
    kept = []
    dropped_links = []
    for lk in links:
        a = (lk["sideA"], parse_bid(lk["bboxIdA"]))
        b = (lk["sideB"], parse_bid(lk["bboxIdB"]))
        if a in nodes_to_isolate or b in nodes_to_isolate:
            dropped_links.append(lk["linkId"])
        else:
            kept.append(lk)

    # Rebuild bunches via UnionFind
    nodes = []
    for sk in side_keys:
        if sk not in images: continue
        side = images[sk]
        sidx = side["side_index"]
        for ann in side["annotations"]:
            nodes.append({
                "side": sk,
                "side_index": sidx,
                "box_index": ann["box_index"],
                "class_name": ann["class_name"],
                "bbox_pixel": ann["bbox_pixel"],
            })

    uf = UF()
    for n in nodes:
        uf.add((n["side_index"], n["box_index"]))
    for lk in kept:
        uf.union(
            (lk["sideA"], parse_bid(lk["bboxIdA"])),
            (lk["sideB"], parse_bid(lk["bboxIdB"])),
        )

    components = defaultdict(list)
    for n in nodes:
        components[uf.find((n["side_index"], n["box_index"]))].append(n)

    def comp_key(c):
        min_si = min(n["side_index"] for n in c)
        min_bi = min(n["box_index"] for n in c if n["side_index"] == min_si)
        return (min_si, min_bi)

    sorted_comps = sorted(components.values(), key=comp_key)
    new_bunches = []
    for i, comp in enumerate(sorted_comps, start=1):
        cs = sorted(comp, key=lambda n: (n["side_index"], n["box_index"]))
        classes = [n["class_name"] for n in cs]
        cc = Counter(classes)
        mx = max(cc.values())
        tied = [c for c, n in cc.items() if n == mx]
        bclass = max(tied)
        new_bunches.append({
            "bunch_id": i,
            "class": bclass,
            "class_mismatch": len(set(classes)) > 1,
            "appearance_count": len(cs),
            "appearances": [
                {"side": n["side"], "side_index": n["side_index"],
                 "box_index": n["box_index"], "class_name": n["class_name"],
                 "bbox_pixel": n["bbox_pixel"]} for n in cs
            ],
        })

    # Verify no remaining violation
    for b in new_bunches:
        sides = sorted({a["side_index"] for a in b["appearances"]})
        if len(sides) > 1:
            valid = any(
                all(circular_dist(h, s, N) <= max_dist for s in sides)
                for h in sides
            )
            assert valid, f"bunch {b['bunch_id']} still violation after heal: {sides}"

    old_n = len(data["bunches"])
    data["_confirmedLinks"] = kept
    data["bunches"] = new_bunches

    total_det = sum(images[sk]["bbox_count"] for sk in side_keys if sk in images)
    by_class = Counter()
    for b in new_bunches:
        c = b["class"] if b["class"] in BY_CLASS else "other"
        by_class[c] += 1

    by_side_dict = {sk: images[sk]["bbox_count"] for sk in side_keys if sk in images}
    data["summary"] = {
        "total_unique_bunches": len(new_bunches),
        "total_detections": total_det,
        "duplicates_linked": total_det - len(new_bunches),
        "by_class": {k: by_class.get(k, 0) for k in BY_CLASS},
        "by_side": by_side_dict,
    }
    data["version"] = max(data.get("version", 2), 3)
    fix_log = data.setdefault("metadata", {}).setdefault("fix_log", [])
    fix_log.append({
        "date": FIX_DATE,
        "action": "auto_heal_visibility_drop_offender",
        "rule": f"largest_bbox=home, drop nodes at sides with dist>{max_dist}",
        "dropped_links": dropped_links,
        "actions": actions,
        "bunches_before": old_n,
        "bunches_after": len(new_bunches),
    })

    return actions, old_n, len(new_bunches)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", action="append", default=None,
                    help="batasi ke tree_id tertentu (boleh multiple)")
    args = ap.parse_args()

    targets = sorted(JSON_DIR.glob("*.json"))
    if args.only:
        only_set = set(args.only)
        targets = [p for p in targets if p.stem in only_set]

    n_healed = 0
    n_violations_fixed = 0
    for jp in targets:
        data = json.loads(jp.read_text(encoding="utf-8-sig"))
        violations = find_violation_bunches(data)
        if not violations:
            continue

        tree_id = data.get("tree_id", jp.stem)
        N = len(data["images"])
        if N != 4:
            continue  # this script only handles 4-sisi

        actions, old_n, new_n = heal_tree(data)
        if not actions:
            continue

        print(f"[HEAL] {tree_id}  bunches: {old_n} -> {new_n}")
        for a in actions:
            print(f"   bunch#{a['bunch_id']} ({a['class']}): home={a['home_side']}, "
                  f"drop nodes={a['isolated_nodes']}")

        if not args.dry_run:
            BACKUP_DIR.mkdir(parents=True, exist_ok=True)
            backup = BACKUP_DIR / f"{tree_id}.json"
            if not backup.exists():
                shutil.copy2(jp, backup)
            jp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                          encoding="utf-8")

        n_healed += 1
        n_violations_fixed += len(actions)

    print()
    print(f"Trees healed: {n_healed}")
    print(f"Violations fixed: {n_violations_fixed}")
    print(f"Mode: {'DRY-RUN' if args.dry_run else 'APPLIED'}")


if __name__ == "__main__":
    main()
