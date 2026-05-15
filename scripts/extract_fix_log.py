"""Extract metadata.fix_log dari semua JSON GT → reports/gt_fix_log/.

Bikin dataset GT bersih untuk publish (HF). Audit trail tetap preserved
di reports/ + git history (and archive backup folders).

Mode:
  --dry-run    Preview only.
  --strip      Setelah extract, hapus metadata.fix_log dari JSON sumber.
               Tanpa ini, JSON tidak ter-modify (extract-only mode).

Output:
  reports/gt_fix_log/fix_log.jsonl   1 line per fix entry (full record)
  reports/gt_fix_log/fix_log.csv     flat tabular summary
  reports/gt_fix_log/summary.md      stats + breakdown
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

BASE     = Path(__file__).resolve().parent.parent
JSON_DIR = BASE / "Brand-New-Dataset-YOLO" / "json"
OUT_DIR  = BASE / "reports" / "gt_fix_log"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--strip", action="store_true",
                    help="hapus metadata.fix_log dari JSON sumber setelah extract")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    stripped = []
    for jp in sorted(JSON_DIR.glob("*.json")):
        data = json.loads(jp.read_text(encoding="utf-8-sig"))
        meta = data.get("metadata", {})
        fl   = meta.get("fix_log", [])
        if not fl:
            continue
        tid = data.get("tree_id", jp.stem)
        for i, entry in enumerate(fl):
            records.append({
                "tree_id":          tid,
                "fix_index":        i,
                "date":             entry.get("date", ""),
                "action":           entry.get("action", ""),
                "rule":             entry.get("rule", ""),
                "bunches_before":   entry.get("bunches_before", ""),
                "bunches_after":    entry.get("bunches_after", ""),
                "dropped_links":    json.dumps(entry.get("dropped_links", [])),
                "actions":          json.dumps(entry.get("actions", [])),
            })
        if args.strip and not args.dry_run:
            del meta["fix_log"]
            if not meta:
                data.pop("metadata", None)
            else:
                data["metadata"] = meta
            jp.write_text(
                json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            stripped.append(tid)

    # JSONL
    jsonl_path = OUT_DIR / "fix_log.jsonl"
    if not args.dry_run:
        with jsonl_path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # CSV
    csv_path = OUT_DIR / "fix_log.csv"
    if not args.dry_run and records:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            w.writeheader()
            w.writerows(records)

    # Summary
    by_action = Counter(r["action"] for r in records)
    by_date   = Counter(r["date"] for r in records)
    summary_path = OUT_DIR / "summary.md"
    lines = [
        "# GT Fix Log Summary",
        "",
        f"- Total fix entries: **{len(records)}**",
        f"- Trees affected: **{len({r['tree_id'] for r in records})}**",
        "",
        "## By action",
        "",
    ]
    for a, c in by_action.most_common():
        lines.append(f"- `{a}` — {c}")
    lines += ["", "## By date", ""]
    for d, c in by_date.most_common():
        lines.append(f"- `{d}` — {c}")
    if not args.dry_run:
        summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Extracted: {len(records)} fix entries from "
          f"{len({r['tree_id'] for r in records})} trees")
    if args.strip:
        print(f"Stripped fix_log from: {len(stripped)} files "
              f"({'DRY-RUN' if args.dry_run else 'APPLIED'})")
    if not args.dry_run:
        print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
