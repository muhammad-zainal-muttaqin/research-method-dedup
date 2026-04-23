"""Eval GeoLinker pada 228 JSON pohon.

Baca reports/geo_linker/best.json (config hasil tuner), jalankan linker di semua 228 pohon,
tulis per-tree CSV + summary.md.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dedup.geo_linker import GeoLinker, LinkerConfig, NAMES

BASE = Path(__file__).resolve().parent.parent
JSON_DIR = BASE / "json"
OUT_DIR = BASE / "reports" / "geo_linker"
BEST_JSON = OUT_DIR / "best.json"


def main():
    cfg_dict = json.loads(BEST_JSON.read_text())["best_config"]
    cfg = LinkerConfig(**{k: v for k, v in cfg_dict.items() if k in LinkerConfig().as_dict()})
    linker = GeoLinker(cfg)
    print(f"Using config: {cfg.as_dict()}")

    rows = []
    per_class_err_total = Counter()
    per_split: dict[str, list[dict]] = {"train": [], "val": [], "test": []}

    for jp in sorted(JSON_DIR.glob("*.json")):
        d = json.loads(jp.read_text(encoding="utf-8"))
        tree_id = d.get("tree_name", d.get("tree_id", jp.stem))
        split = d.get("split", "train")
        gt = {c: d["summary"]["by_class"].get(c, 0) for c in NAMES}
        naive = Counter()
        for side_data in d["images"].values():
            for ann in side_data.get("annotations", []):
                if ann["class_name"] in NAMES:
                    naive[ann["class_name"]] += 1
        pred = linker.count(d)

        total_gt = sum(gt.values())
        total_pred = sum(pred.values())
        total_naive = sum(naive[c] for c in NAMES)
        total_err = abs(total_pred - total_gt)

        row = {"tree_id": tree_id, "split": split, "n_sides": len(d["images"])}
        for c in NAMES:
            row[f"gt_{c}"] = gt[c]
            row[f"pred_{c}"] = pred[c]
            row[f"err_{c}"] = abs(pred[c] - gt[c])
            row[f"naive_{c}"] = naive[c]
            per_class_err_total[c] += abs(pred[c] - gt[c])
        row["total_gt"] = total_gt
        row["total_pred"] = total_pred
        row["total_naive"] = total_naive
        row["total_err"] = total_err
        row["within_1"] = int(total_err <= 1)
        row["within_2"] = int(total_err <= 2)
        row["naive_ratio"] = total_pred / max(total_naive, 1)
        rows.append(row)
        per_split[split].append(row)

    # Write per-tree CSV
    fields = list(rows[0].keys())
    with open(OUT_DIR / "per_tree.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)

    def agg(rs: list[dict]) -> dict:
        n = len(rs)
        if n == 0:
            return {}
        return {
            "n": n,
            "pct_exact": sum(1 for r in rs if r["total_err"] == 0) / n,
            "pct_within_1": sum(r["within_1"] for r in rs) / n,
            "pct_within_2": sum(r["within_2"] for r in rs) / n,
            "mae_total": mean(r["total_err"] for r in rs),
            "mae_B1": mean(r["err_B1"] for r in rs),
            "mae_B2": mean(r["err_B2"] for r in rs),
            "mae_B3": mean(r["err_B3"] for r in rs),
            "mae_B4": mean(r["err_B4"] for r in rs),
            "sum_gt": sum(r["total_gt"] for r in rs),
            "sum_pred": sum(r["total_pred"] for r in rs),
            "sum_naive": sum(r["total_naive"] for r in rs),
        }

    overall = agg(rows)
    per_split_agg = {sp: agg(rs) for sp, rs in per_split.items()}

    # Error distribution
    err_dist = Counter()
    for r in rows:
        signed = r["total_pred"] - r["total_gt"]
        err_dist[signed] += 1

    # Write summary.md
    md = ["# Laporan GeoLinker — GT-Based Dedup (228 pohon JSON)",
          f"**Config:** `{cfg.as_dict()}`", ""]
    md.append("## 1. Metrik Headline (Seluruh 228 pohon)")
    md.append("")
    md.append(f"| Metrik | Nilai |")
    md.append("|---|---:|")
    md.append(f"| n_pohon | {overall['n']} |")
    md.append(f"| `pct_exact` (pred == gt) | **{overall['pct_exact']*100:.1f}%** |")
    md.append(f"| `pct_within_1` (≤ 1 error) | **{overall['pct_within_1']*100:.1f}%** |")
    md.append(f"| `pct_within_2` (≤ 2 error) | **{overall['pct_within_2']*100:.1f}%** |")
    md.append(f"| MAE total (tree-level) | {overall['mae_total']:.3f} |")
    md.append(f"| MAE B1 | {overall['mae_B1']:.3f} |")
    md.append(f"| MAE B2 | {overall['mae_B2']:.3f} |")
    md.append(f"| MAE B3 | {overall['mae_B3']:.3f} |")
    md.append(f"| MAE B4 | {overall['mae_B4']:.3f} |")
    md.append(f"| Σ GT bunches | {overall['sum_gt']:,} |")
    md.append(f"| Σ PRED bunches | {overall['sum_pred']:,} |")
    md.append(f"| Σ NAIVE bunches | {overall['sum_naive']:,} |")
    md.append(f"| Aggregate pred/gt ratio | {overall['sum_pred']/max(overall['sum_gt'],1):.4f} |")
    md.append(f"| Aggregate naive/gt ratio | {overall['sum_naive']/max(overall['sum_gt'],1):.4f} |")
    md.append("")

    md.append("## 2. Perbandingan vs Baseline")
    md.append("")
    md.append("| Metode | pct_within_1 | pct_within_2 | MAE total | ratio pred/gt |")
    md.append("|---|---:|---:|---:|---:|")
    naive_w1 = sum(1 for r in rows if abs(r["total_naive"] - r["total_gt"]) <= 1) / len(rows)
    naive_w2 = sum(1 for r in rows if abs(r["total_naive"] - r["total_gt"]) <= 2) / len(rows)
    naive_mae = mean(abs(r["total_naive"] - r["total_gt"]) for r in rows)
    md.append(f"| **Naive sum** | {naive_w1*100:.1f}% | {naive_w2*100:.1f}% | {naive_mae:.3f} | {overall['sum_naive']/overall['sum_gt']:.3f} |")
    md.append(f"| **GeoLinker (ini)** | **{overall['pct_within_1']*100:.1f}%** | **{overall['pct_within_2']*100:.1f}%** | **{overall['mae_total']:.3f}** | {overall['sum_pred']/overall['sum_gt']:.3f} |")
    md.append("")

    md.append("## 3. Breakdown per Split")
    md.append("")
    md.append("| Split | n | pct_exact | pct_within_1 | pct_within_2 | MAE total |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for sp in ("train", "val", "test"):
        s = per_split_agg.get(sp)
        if not s: continue
        md.append(f"| {sp} | {s['n']} | {s['pct_exact']*100:.1f}% | {s['pct_within_1']*100:.1f}% | {s['pct_within_2']*100:.1f}% | {s['mae_total']:.3f} |")
    md.append("")

    md.append("## 4. Distribusi Signed Error (pred_total − gt_total)")
    md.append("")
    md.append("| Error | n_pohon | % |")
    md.append("|---:|---:|---:|")
    for e in sorted(err_dist):
        md.append(f"| {e:+d} | {err_dist[e]} | {err_dist[e]/len(rows)*100:.1f}% |")
    md.append("")

    md.append("## 5. Top-10 Pohon dengan Error Terbesar")
    md.append("")
    md.append("| tree_id | split | GT total | PRED total | NAIVE total | err |")
    md.append("|---|---|---:|---:|---:|---:|")
    for r in sorted(rows, key=lambda x: -x["total_err"])[:10]:
        md.append(f"| {r['tree_id']} | {r['split']} | {r['total_gt']} | {r['total_pred']} | {r['total_naive']} | {r['total_err']} |")
    md.append("")

    md.append("## 6. Catatan")
    md.append("")
    md.append("- Algoritma murni geometri (tanpa ML/embedding).")
    md.append("- Input = GT bbox + class (label bersih per JSON-01 verdict).")
    md.append("- Fitur: |Δcy_center|, log rasio area bbox.")
    md.append("- Constraint: intra-kelas saja, one-per-side per cluster, adjacency-aware (sisi opposite threshold lebih ketat).")
    md.append("- Ceiling: pure geometri tidak bisa memisahkan dua tandan fisik yang berdekatan-ketinggian tanpa cue visual.")
    md.append(f"- Reproduce: `python scripts/tune_geo_linker.py && python scripts/eval_geo_linker.py`")

    (OUT_DIR / "summary.md").write_text("\n".join(md))
    print(f"\nWrote: {OUT_DIR/'per_tree.csv'}\nWrote: {OUT_DIR/'summary.md'}")
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
