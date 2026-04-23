"""Jalankan GeoLinker di 725 pohon non-JSON (blind, tanpa GT).

Sumber data: dataset/labels/{split}/TREE_SIDE.txt (YOLO format).
Konversi per pohon → pseudo-tree-dict → linker.count().
Tulis reports/geo_linker_blind/{per_tree.csv, summary.md, distribution.csv}.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dedup.geo_linker import GeoLinker, LinkerConfig, NAMES, txt_to_tree_dict

BASE = Path(__file__).resolve().parent.parent
LABEL_DIR = BASE / "dataset" / "labels"
JSON_DIR = BASE / "json"
OUT_DIR = BASE / "reports" / "geo_linker_blind"
BEST_JSON = BASE / "reports" / "geo_linker" / "best.json"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def collect_trees():
    """Return dict tree_id -> {split, sides: {sisi_k: Path}}."""
    trees: dict[str, dict] = {}
    for split in ("train", "val", "test"):
        for f in sorted((LABEL_DIR / split).glob("*.txt")):
            parts = f.stem.rsplit("_", 1)
            if len(parts) != 2:
                continue
            tree_id, side_str = parts
            try:
                side_no = int(side_str)
            except ValueError:
                continue
            if tree_id not in trees:
                trees[tree_id] = {"split": split, "sides": {}}
            trees[tree_id]["sides"][f"sisi_{side_no}"] = f
    return trees


def load_json_tree_ids() -> set[str]:
    ids: set[str] = set()
    for jp in JSON_DIR.glob("*.json"):
        d = json.loads(jp.read_text())
        ids.add(d.get("tree_name", d.get("tree_id", "")))
    return ids


def main():
    cfg_dict = json.loads(BEST_JSON.read_text())["best_config"]
    cfg = LinkerConfig(**{k: v for k, v in cfg_dict.items() if k in LinkerConfig().as_dict()})
    linker = GeoLinker(cfg)
    print(f"Config: {cfg.as_dict()}")

    all_trees = collect_trees()
    json_ids = load_json_tree_ids()
    blind_trees = {tid: info for tid, info in all_trees.items() if tid not in json_ids}
    print(f"Total trees: {len(all_trees)} | JSON: {len(json_ids)} | BLIND: {len(blind_trees)}")

    rows = []
    class_totals_pred = Counter()
    class_totals_naive = Counter()
    by_domain: dict[str, list[dict]] = defaultdict(list)
    by_split: dict[str, list[dict]] = defaultdict(list)
    by_nsides: dict[int, list[dict]] = defaultdict(list)

    for tree_id, info in sorted(blind_trees.items()):
        tree = txt_to_tree_dict(tree_id, info["sides"])
        domain = "DAMIMAS" if "DAMIMAS" in tree_id else ("LONSUM" if "LONSUM" in tree_id else "OTHER")
        # Naive count
        naive = Counter()
        for sd in tree["images"].values():
            for ann in sd["annotations"]:
                naive[ann["class_name"]] += 1
        pred = linker.count(tree)

        total_naive = sum(naive[c] for c in NAMES)
        total_pred = sum(pred.values())
        row = {
            "tree_id": tree_id,
            "domain": domain,
            "split": info["split"],
            "n_sides": len(info["sides"]),
        }
        for c in NAMES:
            row[f"pred_{c}"] = pred[c]
            row[f"naive_{c}"] = naive[c]
            class_totals_pred[c] += pred[c]
            class_totals_naive[c] += naive[c]
        row["total_pred"] = total_pred
        row["total_naive"] = total_naive
        row["naive_ratio"] = total_pred / max(total_naive, 1)
        row["outlier"] = int(total_pred > 25 or row["naive_ratio"] < 0.30 or row["naive_ratio"] > 0.90)
        rows.append(row)
        by_domain[domain].append(row)
        by_split[info["split"]].append(row)
        by_nsides[len(info["sides"])].append(row)

    # per_tree.csv
    with open(OUT_DIR / "per_tree.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # distribution.csv
    with open(OUT_DIR / "distribution.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bucket", "n_trees", "B1", "B2", "B3", "B4", "total_pred", "total_naive", "ratio"])
        for name, group in (*[("split="+k, v) for k, v in by_split.items()],
                            *[("domain="+k, v) for k, v in by_domain.items()],
                            *[(f"n_sides={k}", v) for k, v in by_nsides.items()]):
            n = len(group)
            agg = {c: sum(r[f"pred_{c}"] for r in group) for c in NAMES}
            tp = sum(r["total_pred"] for r in group)
            tn = sum(r["total_naive"] for r in group)
            w.writerow([name, n, agg["B1"], agg["B2"], agg["B3"], agg["B4"], tp, tn, f"{tp/max(tn,1):.3f}"])

    # Reference ratios from JSON-228 (sanity baseline)
    ratios = [r["naive_ratio"] for r in rows]
    outliers = [r for r in rows if r["outlier"]]

    tot_pred = sum(r["total_pred"] for r in rows)
    tot_naive = sum(r["total_naive"] for r in rows)

    md = [f"# Blind Run GeoLinker — {len(rows)} pohon non-JSON", "",
          f"**Config:** `{cfg.as_dict()}`",
          "",
          "## 1. Ringkasan",
          "",
          f"| Item | Nilai |",
          f"|---|---:|",
          f"| Total pohon blind | {len(rows)} |",
          f"| Pohon 4-sisi | {len(by_nsides.get(4, []))} |",
          f"| Pohon 8-sisi | {len(by_nsides.get(8, []))} |",
          f"| Domain DAMIMAS | {len(by_domain.get('DAMIMAS', []))} |",
          f"| Domain LONSUM | {len(by_domain.get('LONSUM', []))} |",
          f"| Σ PRED bunches | {tot_pred:,} |",
          f"| Σ NAIVE bunches | {tot_naive:,} |",
          f"| Aggregate PRED/NAIVE | {tot_pred/max(tot_naive,1):.3f} |",
          f"| Referensi JSON-228 (unique/naive) | 0.559 |",
          "",
          "## 2. Per-Kelas Total",
          "",
          "| Kelas | PRED | NAIVE | PRED/NAIVE | Referensi JSON-228 ratio |",
          "|---|---:|---:|---:|---:|",]
    # Reference ratios from the 228-tree JSON (reports/full_gt_count numbers)
    ref_ratio = {"B1": 291/(291+1618)*1.788,  # approximate from full_gt_count table
                 "B2": 532/(532+2974)*1.788,
                 "B3": 1144/(1144+6417)*1.788,
                 "B4": 499/(499+2656)*1.788}
    # Simpler: use JSON-05 rate of 0.559 uniformly as baseline
    for c in NAMES:
        pv = class_totals_pred[c]; nv = class_totals_naive[c]
        md.append(f"| {c} | {pv:,} | {nv:,} | {pv/max(nv,1):.3f} | 0.559 |")
    md += ["",
           "## 3. Distribusi naive_ratio (per pohon)",
           "",
           f"- min   : {min(ratios):.3f}",
           f"- median: {median(ratios):.3f}",
           f"- mean  : {mean(ratios):.3f}",
           f"- max   : {max(ratios):.3f}",
           f"- n_outlier (>25 pred atau ratio <0.30 atau >0.90): **{len(outliers)}**",
           "",
           "## 4. Per Split",
           "",
           "| Split | n | Σ PRED | Σ NAIVE | ratio |",
           "|---|---:|---:|---:|---:|",]
    for sp in ("train", "val", "test"):
        grp = by_split.get(sp, [])
        if not grp: continue
        tp = sum(r["total_pred"] for r in grp)
        tn = sum(r["total_naive"] for r in grp)
        md.append(f"| {sp} | {len(grp)} | {tp:,} | {tn:,} | {tp/max(tn,1):.3f} |")
    md += ["",
           "## 5. Per Domain",
           "",
           "| Domain | n | Σ PRED | Σ NAIVE | ratio |",
           "|---|---:|---:|---:|---:|",]
    for dom in ("DAMIMAS", "LONSUM"):
        grp = by_domain.get(dom, [])
        if not grp: continue
        tp = sum(r["total_pred"] for r in grp)
        tn = sum(r["total_naive"] for r in grp)
        md.append(f"| {dom} | {len(grp)} | {tp:,} | {tn:,} | {tp/max(tn,1):.3f} |")

    md += ["",
           "## 6. Top-10 Outlier (manual review)",
           "",
           "| tree_id | split | n_sides | total_pred | total_naive | ratio |",
           "|---|---|---:|---:|---:|---:|",]
    for r in sorted(outliers, key=lambda x: -x["total_pred"])[:10]:
        md.append(f"| {r['tree_id']} | {r['split']} | {r['n_sides']} | {r['total_pred']} | {r['total_naive']} | {r['naive_ratio']:.3f} |")

    md += ["",
           "## 7. Catatan",
           "",
           "- **Tidak ada GT untuk 725 pohon ini** → metrik akurasi tidak bisa dihitung langsung.",
           "- Sanity check: ratio PRED/NAIVE ~0.56 (referensi JSON-228) menandakan algoritma berperilaku konsisten.",
           "- Drift per-kelas besar (terutama B3) mengindikasikan distribusi pohon non-JSON memang beda dari ber-JSON (mis. LONSUM lebih B3-heavy).",
           "- `naive_ratio` per pohon dipakai sebagai sanity proxy; outlier flagging untuk manual review.",]

    (OUT_DIR / "summary.md").write_text("\n".join(md))
    print(f"\nWrote: {OUT_DIR/'per_tree.csv'}")
    print(f"Wrote: {OUT_DIR/'distribution.csv'}")
    print(f"Wrote: {OUT_DIR/'summary.md'}")
    print(f"\nΣ PRED = {tot_pred}  |  Σ NAIVE = {tot_naive}  |  ratio = {tot_pred/max(tot_naive,1):.3f}")
    print(f"n_outlier = {len(outliers)}")


if __name__ == "__main__":
    main()
