from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "Brand-New-Dataset-YOLO"
OUT = ROOT / "EDA_report"

JSON_DIR = DATASET / "json"
LABEL_DIR = DATASET / "labels"
IMAGE_DIR = DATASET / "images"
SPLIT_MANIFEST = DATASET / "split_manifest.csv"
PARQUET_PATH = DATASET / "data" / "ground_truth.parquet"

CLASSES = ["B1", "B2", "B3", "B4"]
CLASS_TO_ID = {"B1": "0", "B2": "1", "B3": "2", "B4": "3"}
ID_TO_CLASS = {"0": "B1", "1": "B2", "2": "B3", "3": "B4"}


def ensure_out() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(exist_ok=True)
    (OUT / "plots").mkdir(exist_ok=True)


def load_json_records() -> list[dict]:
    rows: list[dict] = []
    for p in sorted(JSON_DIR.glob("*.json")):
        with p.open("r", encoding="utf-8-sig") as f:
            rows.append(json.load(f))
    return rows


def class_counts_from_labels() -> Counter:
    counts: Counter = Counter()
    for txt in sorted(LABEL_DIR.glob("*.txt")):
        with txt.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                counts[s.split()[0]] += 1
    return counts


def build_tables(records: list[dict]) -> dict[str, pd.DataFrame]:
    tree_rows = []
    bunch_rows = []
    app_rows = []
    side_rows = []
    anno_rows = []
    link_rows = []
    mismatch_rows = []
    image_meta_rows = []

    for rec in records:
        tree_id = rec["tree_id"]
        split = rec.get("split", "")
        domain = tree_id.split("_")[0] if "_" in tree_id else "UNKNOWN"
        images = rec.get("images", {})
        bunches = rec.get("bunches", [])
        summary = rec.get("summary", {})
        by_class = summary.get("by_class", {})
        by_side = summary.get("by_side", {})
        links = rec.get("_confirmedLinks", [])

        side_names = sorted(images.keys(), key=lambda x: images[x].get("side_index", 0))
        n_sides = len(side_names)

        tree_rows.append(
            {
                "tree_id": tree_id,
                "split": split,
                "domain": domain,
                "n_sides": n_sides,
                "total_unique_bunches": int(summary.get("total_unique_bunches", 0)),
                "total_detections": int(summary.get("total_detections", 0)),
                "duplicates_linked": int(summary.get("duplicates_linked", 0)),
                "B1": int(by_class.get("B1", 0)),
                "B2": int(by_class.get("B2", 0)),
                "B3": int(by_class.get("B3", 0)),
                "B4": int(by_class.get("B4", 0)),
                "other": int(by_class.get("other", 0)),
                "n_links": len(links),
            }
        )

        for side_name in side_names:
            side_obj = images[side_name]
            side_rows.append(
                {
                    "tree_id": tree_id,
                    "split": split,
                    "domain": domain,
                    "side": side_name,
                    "side_index": int(side_obj.get("side_index", -1)),
                    "width": int(side_obj.get("width", 0)),
                    "height": int(side_obj.get("height", 0)),
                    "bbox_count_field": int(side_obj.get("bbox_count", 0)),
                    "bbox_count_from_annotations": len(side_obj.get("annotations", [])),
                    "summary_by_side_count": int(by_side.get(side_name, 0)),
                }
            )
            image_meta_rows.append(
                {
                    "tree_id": tree_id,
                    "split": split,
                    "domain": domain,
                    "side": side_name,
                    "filename": side_obj.get("filename", ""),
                    "label_file": side_obj.get("label_file", ""),
                    "width": int(side_obj.get("width", 0)),
                    "height": int(side_obj.get("height", 0)),
                }
            )
            for ann in side_obj.get("annotations", []):
                x, y, w, h = ann.get("bbox_yolo", [0, 0, 0, 0])
                anno_rows.append(
                    {
                        "tree_id": tree_id,
                        "split": split,
                        "domain": domain,
                        "side": side_name,
                        "side_index": int(side_obj.get("side_index", -1)),
                        "box_index": int(ann.get("box_index", -1)),
                        "class_id": int(ann.get("class_id", -1)),
                        "class_name": ann.get("class_name", ""),
                        "x_center": float(x),
                        "y_center": float(y),
                        "w_norm": float(w),
                        "h_norm": float(h),
                        "area_norm": float(w) * float(h),
                    }
                )

        for b in bunches:
            appearances = b.get("appearances", [])
            side_idx = [a.get("side_index", -1) for a in appearances]
            unique_sides = len(set(side_idx))
            app_count = int(b.get("appearance_count", len(appearances)))
            side_dup = len(appearances) - unique_sides
            class_name = b.get("class", "")

            bunch_rows.append(
                {
                    "tree_id": tree_id,
                    "split": split,
                    "domain": domain,
                    "bunch_id": int(b.get("bunch_id", -1)),
                    "class": class_name,
                    "class_mismatch": bool(b.get("class_mismatch", False)),
                    "appearance_count": app_count,
                    "appearance_len": len(appearances),
                    "unique_side_count": unique_sides,
                    "same_side_duplicate_count": side_dup,
                    "tree_n_sides": n_sides,
                }
            )
            if app_count != unique_sides:
                mismatch_rows.append(
                    {
                        "tree_id": tree_id,
                        "split": split,
                        "domain": domain,
                        "bunch_id": int(b.get("bunch_id", -1)),
                        "class": class_name,
                        "appearance_count": app_count,
                        "unique_side_count": unique_sides,
                        "same_side_duplicate_count": side_dup,
                        "tree_n_sides": n_sides,
                    }
                )
            for a in appearances:
                app_rows.append(
                    {
                        "tree_id": tree_id,
                        "split": split,
                        "domain": domain,
                        "bunch_id": int(b.get("bunch_id", -1)),
                        "class": class_name,
                        "side": a.get("side", ""),
                        "side_index": int(a.get("side_index", -1)),
                        "box_index": int(a.get("box_index", -1)),
                    }
                )

        for lk in links:
            link_rows.append(
                {
                    "tree_id": tree_id,
                    "split": split,
                    "domain": domain,
                    "link_id": lk.get("linkId", ""),
                    "sideA": int(lk.get("sideA", -1)),
                    "bboxIdA": lk.get("bboxIdA", ""),
                    "sideB": int(lk.get("sideB", -1)),
                    "bboxIdB": lk.get("bboxIdB", ""),
                    "is_loop_across_sides": int(lk.get("sideA", -1) == lk.get("sideB", -1)),
                }
            )

    return {
        "trees": pd.DataFrame(tree_rows),
        "bunches": pd.DataFrame(bunch_rows),
        "appearances": pd.DataFrame(app_rows),
        "sides": pd.DataFrame(side_rows),
        "annotations": pd.DataFrame(anno_rows),
        "links": pd.DataFrame(link_rows),
        "mismatches": pd.DataFrame(mismatch_rows),
        "image_meta": pd.DataFrame(image_meta_rows),
    }


def write_tables(tables: dict[str, pd.DataFrame]) -> None:
    for name, df in tables.items():
        df.to_csv(OUT / "tables" / f"{name}.csv", index=False, encoding="utf-8")


def save_plot(fig_name: str) -> None:
    plt.tight_layout()
    plt.savefig(OUT / "plots" / fig_name, dpi=160, bbox_inches="tight")
    plt.close()


def make_plots(tables: dict[str, pd.DataFrame], label_class_counts: Counter) -> None:
    trees = tables["trees"]
    bunches = tables["bunches"]
    ann = tables["annotations"]
    sides = tables["sides"]

    plt.figure(figsize=(8, 5))
    vc = trees["n_sides"].value_counts().sort_index()
    plt.bar(vc.index.astype(str), vc.values)
    plt.xlabel("Number of sides per tree")
    plt.ylabel("Tree count")
    plt.title("Tree-side distribution")
    save_plot("tree_side_distribution.png")

    plt.figure(figsize=(8, 5))
    vc = bunches["appearance_count"].value_counts().sort_index()
    plt.bar(vc.index.astype(str), vc.values)
    for i, y in enumerate(vc.values):
        pct = 100.0 * y / len(bunches)
        plt.text(i, y, f"{y:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=8)
    plt.xlabel("Appearance count per unique bunch")
    plt.ylabel("Number of bunches")
    plt.title("Appearance-count distribution")
    save_plot("appearance_count_distribution.png")

    plt.figure(figsize=(8, 5))
    vc = bunches["unique_side_count"].value_counts().sort_index()
    plt.bar(vc.index.astype(str), vc.values)
    plt.xlabel("Unique side count per bunch")
    plt.ylabel("Number of bunches")
    plt.title("Unique-side-count distribution")
    save_plot("unique_side_count_distribution.png")

    plt.figure(figsize=(8, 5))
    dup = bunches["same_side_duplicate_count"].value_counts().sort_index()
    plt.bar(dup.index.astype(str), dup.values)
    plt.xlabel("Duplicate appearances within same side (per bunch)")
    plt.ylabel("Number of bunches")
    plt.title("Same-side duplicate distribution")
    save_plot("same_side_duplicate_distribution.png")

    plt.figure(figsize=(8, 5))
    by_cls = trees[CLASSES].sum()
    plt.bar(by_cls.index, by_cls.values)
    plt.xlabel("Class")
    plt.ylabel("Unique bunch count")
    plt.title("Unique bunches by class (from JSON summary)")
    save_plot("class_distribution_unique_bunches.png")

    plt.figure(figsize=(8, 5))
    order = sorted(label_class_counts.keys(), key=lambda z: int(z) if z.isdigit() else 99)
    x = [ID_TO_CLASS.get(k, k) for k in order]
    y = [label_class_counts[k] for k in order]
    plt.bar(x, y)
    plt.xlabel("Class")
    plt.ylabel("Detection count")
    plt.title("Detection distribution from YOLO labels")
    save_plot("class_distribution_detections_from_labels.png")

    plt.figure(figsize=(8, 5))
    plt.hist(ann["area_norm"], bins=60)
    plt.xlabel("Normalized bbox area")
    plt.ylabel("Frequency")
    plt.title("BBox area distribution")
    save_plot("bbox_area_distribution.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(ann["x_center"], bins=50)
    axes[0].set_title("x_center")
    axes[1].hist(ann["y_center"], bins=50)
    axes[1].set_title("y_center")
    save_plot("bbox_center_distribution.png")

    plt.figure(figsize=(8, 5))
    plt.scatter(ann["w_norm"], ann["h_norm"], s=2, alpha=0.25)
    plt.xlabel("w_norm")
    plt.ylabel("h_norm")
    plt.title("BBox width-height scatter")
    save_plot("bbox_wh_scatter.png")

    plt.figure(figsize=(8, 5))
    split_totals = trees.groupby("split")[CLASSES].sum().sum(axis=1).sort_values(ascending=False)
    plt.bar(split_totals.index, split_totals.values)
    plt.xlabel("Split")
    plt.ylabel("Unique bunch count")
    plt.title("Unique bunch totals by split")
    save_plot("split_unique_bunch_totals.png")

    plt.figure(figsize=(8, 5))
    side_bbox = sides.groupby("side_index")["bbox_count_from_annotations"].sum().sort_index()
    plt.bar(side_bbox.index.astype(str), side_bbox.values)
    plt.xlabel("Side index")
    plt.ylabel("Total detections")
    plt.title("Total detections by side index")
    save_plot("detections_by_side_index.png")

    for cls in sorted(ann["class_name"].dropna().unique()):
        sub = ann[ann["class_name"] == cls]
        if sub.empty:
            continue
        plt.figure(figsize=(6, 6))
        plt.hexbin(sub["x_center"], sub["y_center"], gridsize=35, cmap="viridis", mincnt=1)
        plt.colorbar(label="Count")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel("x_center")
        plt.ylabel("y_center")
        plt.title(f"Spatial heatmap - {cls}")
        save_plot(f"spatial_heatmap_{cls}.png")


def run_advanced_eda(records: list[dict], tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    integrity_rows = []
    graph_rows = []
    appearance_gt_side_rows = []

    for rec in records:
        tree_id = rec["tree_id"]
        split = rec.get("split", "")
        domain = tree_id.split("_")[0] if "_" in tree_id else "UNKNOWN"
        images = rec.get("images", {})
        summary = rec.get("summary", {})
        bunches = rec.get("bunches", [])
        links = rec.get("_confirmedLinks", [])

        for side_name, side_obj in images.items():
            img_name = side_obj.get("filename", "")
            lbl_name = side_obj.get("label_file", "")
            img_path = IMAGE_DIR / img_name
            lbl_path = LABEL_DIR / lbl_name

            json_ann = side_obj.get("annotations", [])
            json_count = len(json_ann)
            bbox_field = int(side_obj.get("bbox_count", 0))
            summary_side = int(summary.get("by_side", {}).get(side_name, 0))

            lbl_count = 0
            parse_error = 0
            label_cls = Counter()
            if lbl_path.exists():
                with lbl_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        s = line.strip()
                        if not s:
                            continue
                        parts = s.split()
                        if len(parts) < 5:
                            parse_error += 1
                            continue
                        lbl_count += 1
                        label_cls[parts[0]] += 1

            json_cls = Counter()
            for a in json_ann:
                json_cls[CLASS_TO_ID.get(a.get("class_name", ""), "other")] += 1

            integrity_rows.append(
                {
                    "tree_id": tree_id,
                    "split": split,
                    "domain": domain,
                    "side": side_name,
                    "image_file": img_name,
                    "label_file": lbl_name,
                    "image_exists": int(img_path.exists()),
                    "label_exists": int(lbl_path.exists()),
                    "json_annotation_count": json_count,
                    "json_bbox_count_field": bbox_field,
                    "json_summary_by_side_count": summary_side,
                    "label_line_count": lbl_count,
                    "label_parse_error_lines": parse_error,
                    "json_vs_label_count_diff": json_count - lbl_count,
                    "json_vs_bbox_field_diff": json_count - bbox_field,
                    "json_vs_summary_side_diff": json_count - summary_side,
                    "json_cls0_B1": json_cls.get("0", 0),
                    "json_cls1_B2": json_cls.get("1", 0),
                    "json_cls2_B3": json_cls.get("2", 0),
                    "json_cls3_B4": json_cls.get("3", 0),
                    "label_cls0_B1": label_cls.get("0", 0),
                    "label_cls1_B2": label_cls.get("1", 0),
                    "label_cls2_B3": label_cls.get("2", 0),
                    "label_cls3_B4": label_cls.get("3", 0),
                }
            )

        nodes = set()
        edges = set()
        for lk in links:
            a = (int(lk.get("sideA", -1)), str(lk.get("bboxIdA", "")))
            b = (int(lk.get("sideB", -1)), str(lk.get("bboxIdB", "")))
            nodes.add(a)
            nodes.add(b)
            edges.add(tuple(sorted((a, b))))

        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        seen = set()
        comp_sizes = []
        for n in nodes:
            if n in seen:
                continue
            stack = [n]
            seen.add(n)
            c = 0
            while stack:
                cur = stack.pop()
                c += 1
                for nxt in adj[cur]:
                    if nxt not in seen:
                        seen.add(nxt)
                        stack.append(nxt)
            comp_sizes.append(c)

        n_nodes = len(nodes)
        n_edges = len(edges)
        n_comp = len(comp_sizes)
        cycle_rank = n_edges - n_nodes + n_comp if n_nodes > 0 else 0
        max_deg = max((len(adj[n]) for n in adj), default=0)
        graph_rows.append(
            {
                "tree_id": tree_id,
                "split": split,
                "domain": domain,
                "n_nodes": n_nodes,
                "n_edges": n_edges,
                "n_components": n_comp,
                "largest_component_size": max(comp_sizes) if comp_sizes else 0,
                "cycle_rank": cycle_rank,
                "max_degree": max_deg,
            }
        )

        tree_n_sides = len(images)
        for b in bunches:
            app_count = int(b.get("appearance_count", 0))
            if app_count > tree_n_sides:
                appearance_gt_side_rows.append(
                    {
                        "tree_id": tree_id,
                        "split": split,
                        "domain": domain,
                        "tree_n_sides": tree_n_sides,
                        "bunch_id": int(b.get("bunch_id", -1)),
                        "class": b.get("class", ""),
                        "appearance_count": app_count,
                        "unique_side_count": len(set(a.get("side_index", -1) for a in b.get("appearances", []))),
                        "appearances_json": json.dumps(b.get("appearances", []), ensure_ascii=False),
                    }
                )

    return {
        "integrity_side_level": pd.DataFrame(integrity_rows),
        "link_graph_tree_level": pd.DataFrame(graph_rows),
        "appearance_gt_tree_sides_cases": pd.DataFrame(appearance_gt_side_rows),
    }


def write_advanced_tables(advanced: dict[str, pd.DataFrame]) -> None:
    for name, df in advanced.items():
        df.to_csv(OUT / "tables" / f"{name}.csv", index=False, encoding="utf-8")


def write_advanced_stats(tables: dict[str, pd.DataFrame], advanced: dict[str, pd.DataFrame], split_manifest_df: pd.DataFrame) -> None:
    trees = tables["trees"]
    ann = tables["annotations"]
    integrity = advanced["integrity_side_level"]
    graph = advanced["link_graph_tree_level"]

    drift_rows = []
    for cls in sorted(ann["class_name"].dropna().unique()):
        a = ann[(ann["domain"] == "DAMIMAS") & (ann["class_name"] == cls)]["area_norm"].values
        b = ann[(ann["domain"] == "LONSUM") & (ann["class_name"] == cls)]["area_norm"].values
        if len(a) > 0 and len(b) > 0:
            ks = ks_2samp(a, b)
            drift_rows.append({"test": "KS_area_norm_domain", "class": cls, "statistic": ks.statistic, "pvalue": ks.pvalue})

    cls_domain = trees.groupby("domain")[CLASSES].sum()
    if len(cls_domain) >= 2:
        chi2, p, _, _ = chi2_contingency(cls_domain.values)
        drift_rows.append({"test": "ChiSquare_class_mix_domain", "class": "ALL", "statistic": chi2, "pvalue": p})

    cls_split = trees.groupby("split")[CLASSES].sum()
    if len(cls_split) >= 2:
        chi2, p, _, _ = chi2_contingency(cls_split.values)
        drift_rows.append({"test": "ChiSquare_class_mix_split", "class": "ALL", "statistic": chi2, "pvalue": p})

    pd.DataFrame(drift_rows).to_csv(OUT / "tables" / "statistical_drift_tests.csv", index=False, encoding="utf-8")

    quality = [
        {"metric": "image_exists_all", "value": int((integrity["image_exists"] == 1).all()), "threshold": 1},
        {"metric": "label_exists_all", "value": int((integrity["label_exists"] == 1).all()), "threshold": 1},
        {"metric": "json_label_count_exact_match_rate", "value": float((integrity["json_vs_label_count_diff"] == 0).mean()), "threshold": 0.999},
        {"metric": "json_bbox_field_exact_match_rate", "value": float((integrity["json_vs_bbox_field_diff"] == 0).mean()), "threshold": 1.0},
        {"metric": "json_summary_side_exact_match_rate", "value": float((integrity["json_vs_summary_side_diff"] == 0).mean()), "threshold": 1.0},
        {"metric": "link_graph_cycle_rank_gt0_rate", "value": float((graph["cycle_rank"] > 0).mean()), "threshold": 0.0},
    ]
    pd.DataFrame(quality).to_csv(OUT / "tables" / "data_quality_scorecard.csv", index=False, encoding="utf-8")

    out = trees.assign(
        det_per_unique=np.where(trees["total_unique_bunches"] > 0, trees["total_detections"] / trees["total_unique_bunches"], np.nan)
    )
    mu = out["det_per_unique"].mean()
    sd = out["det_per_unique"].std(ddof=0)
    out["z_det_per_unique"] = (out["det_per_unique"] - mu) / sd if sd > 0 else 0.0
    out["is_outlier_abs_z_gt_3"] = (out["z_det_per_unique"].abs() > 3).astype(int)
    out.sort_values("z_det_per_unique", ascending=False).to_csv(OUT / "tables" / "tree_outlier_scores.csv", index=False, encoding="utf-8")

    split_col = "split" if "split" in split_manifest_df.columns else ("split_name" if "split_name" in split_manifest_df.columns else None)
    if split_col is not None:
        split_manifest_df.groupby(split_col).size().rename("rows").reset_index().to_csv(
            OUT / "tables" / "split_manifest_split_counts.csv", index=False, encoding="utf-8"
        )


def build_summary_md(
    tables: dict[str, pd.DataFrame],
    advanced: dict[str, pd.DataFrame],
    label_class_counts: Counter,
    split_manifest_df: pd.DataFrame,
    parquet_df: pd.DataFrame | None,
) -> None:
    trees = tables["trees"]
    bunches = tables["bunches"]
    annotations = tables["annotations"]
    mismatches = tables["mismatches"]
    links = tables["links"]
    integrity = advanced["integrity_side_level"]
    graph = advanced["link_graph_tree_level"]
    app_gt_side = advanced["appearance_gt_tree_sides_cases"]

    side_dist = trees["n_sides"].value_counts().sort_index().to_dict()
    app_dist = bunches["appearance_count"].value_counts().sort_index().to_dict()
    unique_side_dist = bunches["unique_side_count"].value_counts().sort_index().to_dict()
    dup_dist = bunches["same_side_duplicate_count"].value_counts().sort_index().to_dict()

    app_gt4 = int((bunches["appearance_count"] > 4).sum())
    app_gt_tree_sides = int((bunches["appearance_count"] > bunches["tree_n_sides"]).sum())
    mismatch_examples = mismatches.sort_values(["same_side_duplicate_count", "appearance_count"], ascending=False).head(20)

    per_tree_density = trees.assign(
        det_per_unique=np.where(trees["total_unique_bunches"] > 0, trees["total_detections"] / trees["total_unique_bunches"], np.nan)
    )
    top_dense = per_tree_density.sort_values("det_per_unique", ascending=False).head(15)

    lines = []
    lines.append("# EDA Report - Brand-New-Dataset-YOLO")
    lines.append("")
    lines.append("## Scope")
    lines.append("- Source: `Brand-New-Dataset-YOLO/`")
    lines.append("- JSON GT files analyzed from `json/*.json`")
    lines.append("- Label detections analyzed from `labels/*.txt`")
    lines.append("- Split metadata from `split_manifest.csv`")
    lines.append("- Optional parquet read from `data/ground_truth.parquet`")
    lines.append("")
    lines.append("## Global Counts")
    lines.append(f"- Trees (JSON): **{len(trees):,}**")
    lines.append(f"- Unique bunches: **{len(bunches):,}**")
    lines.append(f"- Annotation rows (YOLO-like entries in JSON images): **{len(annotations):,}**")
    lines.append(f"- Confirmed links: **{len(links):,}**")
    lines.append("")
    lines.append("## Side Distribution (Trees)")
    for k, v in side_dist.items():
        lines.append(f"- {k} sides: {v:,} trees")
    lines.append("")
    lines.append("## Appearance Distribution (Unique Bunches)")
    for k, v in app_dist.items():
        pct = 100 * v / len(bunches)
        lines.append(f"- appearance_count={k}: {v:,} ({pct:.1f}%)")
    lines.append("")
    lines.append("## Unique Side Count Distribution (Unique Bunches)")
    for k, v in unique_side_dist.items():
        pct = 100 * v / len(bunches)
        lines.append(f"- unique_side_count={k}: {v:,} ({pct:.1f}%)")
    lines.append("")
    lines.append("## Same-side Duplicate Distribution")
    for k, v in dup_dist.items():
        pct = 100 * v / len(bunches)
        lines.append(f"- same_side_duplicate_count={k}: {v:,} ({pct:.2f}%)")
    lines.append("")
    lines.append("## Key Anomaly Counters")
    lines.append(f"- Bunches with `appearance_count > 4`: **{app_gt4:,}**")
    lines.append(f"- Bunches with `appearance_count > tree_n_sides`: **{app_gt_tree_sides:,}**")
    lines.append(f"- Rows in `tables/mismatches.csv`: **{len(mismatches):,}**")
    lines.append(f"- Rows in `tables/appearance_gt_tree_sides_cases.csv`: **{len(app_gt_side):,}**")
    lines.append("")
    lines.append("## Integrity Audit (JSON/TXT/Image)")
    lines.append(f"- Side rows audited: **{len(integrity):,}**")
    lines.append(f"- Missing images: **{int((integrity['image_exists'] == 0).sum()):,}**")
    lines.append(f"- Missing labels: **{int((integrity['label_exists'] == 0).sum()):,}**")
    lines.append(f"- JSON vs label count exact match: **{100*(integrity['json_vs_label_count_diff'] == 0).mean():.2f}%**")
    lines.append(f"- JSON vs bbox_count exact match: **{100*(integrity['json_vs_bbox_field_diff'] == 0).mean():.2f}%**")
    lines.append(f"- JSON vs summary.by_side exact match: **{100*(integrity['json_vs_summary_side_diff'] == 0).mean():.2f}%**")
    lines.append("")
    lines.append("## Link-Graph Diagnostics")
    lines.append(f"- Trees with cycle_rank > 0: **{int((graph['cycle_rank'] > 0).sum()):,}**")
    lines.append(f"- Max cycle_rank: **{int(graph['cycle_rank'].max()) if len(graph) else 0}**")
    lines.append(f"- Max graph degree: **{int(graph['max_degree'].max()) if len(graph) else 0}**")
    lines.append("")
    lines.append("## Class Distribution")
    by_class = trees[CLASSES].sum()
    for c in CLASSES:
        lines.append(f"- JSON unique bunch {c}: {int(by_class[c]):,}")
    lines.append("")
    lines.append("### Detection Distribution from labels/*.txt")
    for raw_cls in sorted(label_class_counts.keys(), key=lambda z: int(z) if z.isdigit() else 99):
        lines.append(f"- Label class {raw_cls} ({ID_TO_CLASS.get(raw_cls, '?')}): {label_class_counts[raw_cls]:,}")
    lines.append("")
    lines.append("## Split Summary (from JSON)")
    split_sum = trees.groupby("split")[CLASSES + ["total_unique_bunches", "total_detections"]].sum()
    lines.append(split_sum.to_markdown())
    lines.append("")
    lines.append("## Top Trees by Detection-per-Unique-Bunch Ratio")
    lines.append(
        top_dense[["tree_id", "split", "n_sides", "total_detections", "total_unique_bunches", "det_per_unique"]].to_markdown(index=False)
    )
    lines.append("")
    lines.append("## Sample Mismatch Cases (same bunch repeated in same side)")
    if len(mismatch_examples) == 0:
        lines.append("- No mismatch rows.")
    else:
        lines.append(mismatch_examples.to_markdown(index=False))
    lines.append("")
    lines.append("## split_manifest.csv quick checks")
    lines.append(f"- Rows in split_manifest.csv: **{len(split_manifest_df):,}**")
    lines.append(f"- Unique tree_id in split_manifest.csv: **{split_manifest_df['tree_id'].nunique():,}**")
    lines.append("")

    if parquet_df is None:
        lines.append("## ground_truth.parquet")
        lines.append("- Could not load parquet file.")
        lines.append("")
    else:
        lines.append("## ground_truth.parquet")
        lines.append(f"- Rows: **{len(parquet_df):,}**")
        lines.append(f"- Columns ({len(parquet_df.columns)}): `{', '.join(parquet_df.columns)}`")
        lines.append("")

    lines.append("## Outputs")
    lines.append("- Tables: `EDA_report/tables/*.csv`")
    lines.append("- Plots: `EDA_report/plots/*.png`")
    lines.append("- This summary: `EDA_report/SUMMARY.md`")
    lines.append("- Advanced stats: `statistical_drift_tests.csv`, `data_quality_scorecard.csv`, `tree_outlier_scores.csv`")
    lines.append("")
    (OUT / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_out()
    records = load_json_records()
    tables = build_tables(records)
    write_tables(tables)

    label_class_counts = class_counts_from_labels()
    split_manifest_df = pd.read_csv(SPLIT_MANIFEST, encoding="utf-8")
    split_manifest_df.to_csv(OUT / "tables" / "split_manifest_snapshot.csv", index=False, encoding="utf-8")

    parquet_df = None
    if PARQUET_PATH.exists():
        parquet_df = pd.read_parquet(PARQUET_PATH)
        parquet_df.to_csv(OUT / "tables" / "ground_truth_parquet_snapshot.csv", index=False, encoding="utf-8")

    advanced = run_advanced_eda(records, tables)
    write_advanced_tables(advanced)
    write_advanced_stats(tables, advanced, split_manifest_df)
    make_plots(tables, label_class_counts)
    build_summary_md(tables, advanced, label_class_counts, split_manifest_df, parquet_df)
    print(f"EDA complete. Output: {OUT}")


if __name__ == "__main__":
    main()
