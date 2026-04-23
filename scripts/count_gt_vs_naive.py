"""
JSON-05: GT-based counting — compare naive sum vs JSON-dedup unique count.
JSON-01: Label consistency audit (bonus, no extra cost).

Input : json/*.json  (228 pohon)
Output: reports/json_05/count_mae_gt.csv
        reports/json_05/summary.md
        reports/label_audit/per_class_inconsistency.csv
        reports/label_audit/leak_pairs.csv
No GPU required. Run from workspace root:
    python scripts/count_gt_vs_naive.py
"""

import json
from pathlib import Path
from collections import Counter
import csv

BASE     = Path(__file__).resolve().parent.parent
JSON_DIR = BASE / "json"
OUT_05   = BASE / "reports/json_05"
OUT_AUD  = BASE / "reports/label_audit"
NAMES    = ["B1", "B2", "B3", "B4"]

OUT_05.mkdir(parents=True, exist_ok=True)
OUT_AUD.mkdir(parents=True, exist_ok=True)


def naive_count(data: dict) -> dict:
    counts = Counter()
    for side_data in data["images"].values():
        for ann in side_data["annotations"]:
            counts[ann["class_name"]] += 1
    return {c: counts.get(c, 0) for c in NAMES}


def gt_count(data: dict) -> dict:
    bc = data["summary"]["by_class"]
    return {c: bc.get(c, 0) for c in NAMES}


def main():
    rows_05  = []
    rows_aud = []
    per_class_total        = Counter()
    per_class_inconsistent = Counter()
    leak_pairs             = Counter()
    n_trees = 0
    n_bunches_multi = 0

    for jp in sorted(JSON_DIR.glob("*.json")):
        data = json.loads(jp.read_text(encoding="utf-8"))
        n_trees += 1

        tree_id = data.get("tree_name", data.get("tree_id", jp.stem))
        split   = data.get("split", "unknown")
        naive   = naive_count(data)
        gt      = gt_count(data)

        row = {"tree_id": tree_id, "split": split}
        total_abs_err = 0
        for c in NAMES:
            row[f"{c}_gt"]    = gt[c]
            row[f"{c}_naive"] = naive[c]
            row[f"{c}_err"]   = naive[c] - gt[c]
            total_abs_err    += abs(naive[c] - gt[c])
        row["total_gt"]    = sum(gt[c]    for c in NAMES)
        row["total_naive"] = sum(naive[c] for c in NAMES)
        row["MAE_overall"] = total_abs_err / len(NAMES)
        rows_05.append(row)

        for bunch in data["bunches"]:
            if bunch["appearance_count"] < 2:
                continue
            n_bunches_multi += 1
            base   = bunch["class"]
            labels = [app["class_name"] for app in bunch["appearances"]]
            per_class_total[base] += 1
            if bunch.get("class_mismatch", len(set(labels)) > 1):
                per_class_inconsistent[base] += 1
                rows_aud.append({
                    "tree_id": tree_id, "bunch_id": bunch["bunch_id"],
                    "json_class": base, "labels": "|".join(labels),
                    "n_views": len(labels),
                })
                for a in labels:
                    for b in labels:
                        if a != b:
                            leak_pairs[(a, b)] += 1

    # ── write JSON-05 CSV ──────────────────────────────────────────────────
    fieldnames_05 = (
        ["tree_id", "split"]
        + [f"{c}_{s}" for c in NAMES for s in ("gt", "naive", "err")]
        + ["total_gt", "total_naive", "MAE_overall"]
    )
    with open(OUT_05 / "count_mae_gt.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames_05)
        w.writeheader(); w.writerows(rows_05)

    # ── write JSON-01 audit CSVs ───────────────────────────────────────────
    with open(OUT_AUD / "inconsistent_bunches.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["tree_id","bunch_id","json_class","labels","n_views"])
        w.writeheader(); w.writerows(rows_aud)

    inc_rows = []
    for c in NAMES:
        tot = per_class_total[c]; inc = per_class_inconsistent[c]
        inc_rows.append({"class": c, "total": tot, "inconsistent": inc,
                         "rate": inc / max(tot, 1)})
    with open(OUT_AUD / "per_class_inconsistency.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["class","total","inconsistent","rate"])
        w.writeheader(); w.writerows(inc_rows)

    leak_rows = [{"true": a, "leaked": b, "count": n}
                 for (a,b), n in leak_pairs.most_common(20)]
    with open(OUT_AUD / "leak_pairs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["true","leaked","count"])
        w.writeheader(); w.writerows(leak_rows)

    # ── print summary ──────────────────────────────────────────────────────
    total_gt_all    = sum(r["total_gt"]    for r in rows_05)
    total_naive_all = sum(r["total_naive"] for r in rows_05)

    print(f"\n{'='*55}")
    print(f"JSON-05: GT Counting vs Naive Sum -- {n_trees} trees")
    print(f"{'='*55}")
    print(f"{'Class':<6} {'GT (unique)':>12} {'Naive (sum)':>12} {'Overcounting':>13} {'OC %':>7}")
    print("-"*55)
    for c in NAMES:
        gt_s  = sum(r[f"{c}_gt"]    for r in rows_05)
        na_s  = sum(r[f"{c}_naive"] for r in rows_05)
        oc    = na_s - gt_s
        print(f"{c:<6} {gt_s:>12} {na_s:>12} {oc:>13} {oc/max(gt_s,1)*100:>6.1f}%")
    oc_total = total_naive_all - total_gt_all
    print("-"*55)
    print(f"{'TOTAL':<6} {total_gt_all:>12} {total_naive_all:>12} "
          f"{oc_total:>13} {oc_total/max(total_gt_all,1)*100:>6.1f}%")

    print(f"\nJSON-01: Label consistency (multi-view bunches = {n_bunches_multi})")
    print(f"{'Class':<6} {'Inconsistent':>13} {'Total':>8} {'Rate':>8}")
    print("-"*40)
    b23_inc = b23_tot = b14_inc = b14_tot = 0
    for r in inc_rows:
        tag = " <-- B2/B3" if r["class"] in ("B2","B3") else ""
        print(f"{r['class']:<6} {r['inconsistent']:>13} {r['total']:>8} {r['rate']:>7.1%}{tag}")
        if r["class"] in ("B2","B3"): b23_inc += r["inconsistent"]; b23_tot += r["total"]
        else:                          b14_inc += r["inconsistent"]; b14_tot += r["total"]
    b23_rate = b23_inc / max(b23_tot, 1)
    b14_rate = b14_inc / max(b14_tot, 1)
    ratio    = b23_rate / max(b14_rate, 1e-9)
    print("-"*40)
    print(f"B2/B3: {b23_rate:.1%}  |  B1/B4: {b14_rate:.1%}  |  ratio: {ratio:.1f}x")

    if b23_rate > 2 * b14_rate and b23_rate > 0.05:
        verdict = "H-LBL-1 CONFIRMED -- label-ceiling on B2/B3 likely"
    elif b23_rate < 0.05:
        verdict = "H-LBL-1 FALSIFIED -- B2/B3 labels consistent"
    else:
        verdict = "H-LBL-1 INCONCLUSIVE"
    print(f"\nVerdict JSON-01: {verdict}")
    print(f"\nOutputs saved to reports/")


if __name__ == "__main__":
    main()
