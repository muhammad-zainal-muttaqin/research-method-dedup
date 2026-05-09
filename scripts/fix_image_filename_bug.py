"""Fix image filename copy-paste bug in JSON GT.

Bug: 6 trees have side.filename / side.label_file pointing to a different
tree's image (UI state leak in tools_sawit/). Annotations themselves are
correct — bbox count matches the TXT lines of the ORIG image
({tree_name}_{sidx}.txt) for every fixable side.

Fix: rewrite each side's `filename` and `label_file` to:
  filename   = "{tree_name}_{sidx}.jpg"
  label_file = "{tree_name}_{sidx}.txt"
where sidx = side_index + 1 (or parsed from "sisi_N" key).

Also remove any stray `image_filename` field if present (legacy/UI bug).

Backup: <dir>.backup_<timestamp>/ created before any write.

Targets:
  - 05 Mei 2026/Output JSON/   (953 trees, raw export)
  - json_05 Mei 2026/          (882 canonical, dedup'd subset)
"""
import json, os, shutil
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGETS = [
    os.path.join(ROOT, "05 Mei 2026", "Output JSON"),
    os.path.join(ROOT, "json_05 Mei 2026"),
]
BAD_TREES = [
    "DAMIMAS_A21B_0117",
    "DAMIMAS_A21B_0119",
    "DAMIMAS_A21B_0854",
    "LONSUM_A21A_0011",
    "LONSUM_A21A_0013",
    "LONSUM_A21A_0015",
]


def patch_file(path: str, tree_name: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    changes = []
    for side, sd in data.get("images", {}).items():
        sidx = side.replace("sisi_", "")
        exp_img = f"{tree_name}_{sidx}.jpg"
        exp_txt = f"{tree_name}_{sidx}.txt"
        old_img = sd.get("filename")
        old_txt = sd.get("label_file")
        rec = {"side": side}
        if old_img != exp_img:
            sd["filename"] = exp_img
            rec["filename"] = {"old": old_img, "new": exp_img}
        if old_txt != exp_txt:
            sd["label_file"] = exp_txt
            rec["label_file"] = {"old": old_txt, "new": exp_txt}
        if "image_filename" in sd:
            sd.pop("image_filename")
            rec["dropped_image_filename"] = True
        if len(rec) > 1:
            changes.append(rec)
    if changes:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    return changes


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    total = 0
    for tdir in TARGETS:
        if not os.path.isdir(tdir):
            print(f"[skip] {tdir} not found")
            continue
        backup = f"{tdir}.backup_{ts}"
        print(f"[backup] {tdir} -> {os.path.basename(backup)}")
        shutil.copytree(tdir, backup)
        for tn in BAD_TREES:
            p = os.path.join(tdir, tn + ".json")
            if not os.path.exists(p):
                print(f"  [skip] {tn} (not in {os.path.basename(tdir)})")
                continue
            changes = patch_file(p, tn)
            print(f"  [patched] {tn}: {len(changes)} sides changed")
            total += len(changes)
    print(f"\nDone. {total} side-records patched.")


if __name__ == "__main__":
    main()
