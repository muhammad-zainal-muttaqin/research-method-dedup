"""Regenerate stale tree_id to match tree_name.

Some canonical JSON files carry a tree_id from earlier dedup batches
(e.g. tree_name=DAMIMAS_A21B_0244 but tree_id=20260422-DAMIMAS-001).
Code uses tree_name as the authoritative key, but the stale tree_id
is a cosmetic mismatch. This rewrites tree_id := tree_name for any
file where they differ.

Backup: <dir>.backup_treeid_<timestamp>/.
Targets: 05 Mei 2026/Output JSON/, json_05 Mei 2026/.
"""
import json, os, shutil
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGETS = [
    os.path.join(ROOT, "05 Mei 2026", "Output JSON"),
    os.path.join(ROOT, "json_05 Mei 2026"),
]


def patch_dir(tdir: str) -> int:
    n = 0
    for jf in os.listdir(tdir):
        if not jf.endswith(".json"):
            continue
        p = os.path.join(tdir, jf)
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        tn = d.get("tree_name") or jf.replace(".json", "")
        if d.get("tree_id") != tn:
            d["tree_id"] = tn
            with open(p, "w", encoding="utf-8") as f:
                json.dump(d, f, ensure_ascii=False, indent=2)
            n += 1
    return n


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for tdir in TARGETS:
        if not os.path.isdir(tdir):
            print(f"[skip] {tdir}")
            continue
        bk = f"{tdir}.backup_treeid_{ts}"
        print(f"[backup] {os.path.basename(tdir)} -> {os.path.basename(bk)}")
        shutil.copytree(tdir, bk)
        n = patch_dir(tdir)
        print(f"  patched: {n} files")


if __name__ == "__main__":
    main()
