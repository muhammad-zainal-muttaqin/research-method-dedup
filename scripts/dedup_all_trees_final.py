"""
Final dedup comparison:
- 228 JSON trees: use JSON annotations + GT for accuracy eval (same as v1/v2/v3)
- 725 non-JSON trees: use TXT labels, run dedup methods, output counts (no GT)
"""

import json
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
JSON_DIR = BASE / "json"
LABEL_DIRS = [BASE / "dataset" / "labels" / s for s in ["train", "val", "test"]]
OUT_DIR = BASE / "reports" / "dedup_all_trees_final"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NAMES = ["B1", "B2", "B3", "B4"]
CLASS_MAP = {"B1": 0, "B2": 1, "B3": 2, "B4": 3}
INV_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}

# Learned thresholds from v3 (trained on _confirmedLinks)
LEARNED = json.loads((BASE / "reports" / "dedup_research_v3" / "learned_thresholds.json").read_text(encoding="utf-8"))


# ═══════════════════════ JSON loading ═══════════════════════

def load_json_tree(json_path):
    data = json.loads(json_path.read_text(encoding="utf-8"))
    tree_id = data.get("tree_name", data.get("tree_id", json_path.stem))
    gt = data["summary"]["by_class"]
    gt_counts = {c: gt.get(c, 0) for c in NAMES}

    all_detections = []
    for side, side_data in data["images"].items():
        si = side_data.get("side_index", int(side.replace("sisi_", "")) - 1)
        for ann in side_data.get("annotations", []):
            if "bbox_yolo" in ann:
                cx, cy, w, h = ann["bbox_yolo"]
                all_detections.append({
                    "class": ann["class_name"],
                    "y_norm": cy,
                    "x_norm": cx,
                    "area_norm": w * h,
                    "aspect_ratio": w / h if h > 0 else 1.0,
                    "side": side,
                    "side_index": si,
                })
    return tree_id, all_detections, gt_counts, data.get("split", "unknown")


# ═══════════════════════ TXT loading ═══════════════════════

def load_txt_trees():
    """Parse all YOLO TXT labels into tree_id -> detections dict."""
    trees = defaultdict(list)
    for lbl_dir in LABEL_DIRS:
        for txt_path in lbl_dir.glob("*.txt"):
            stem = txt_path.stem
            parts = stem.rsplit("_", 1)
            tree_id = parts[0]
            side_num = int(parts[1])
            side_name = f"sisi_{side_num}"
            side_index = side_num - 1

            with open(txt_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    cls_id, cx, cy, w, h = map(float, line.split())
                    cname = INV_CLASS_MAP.get(int(cls_id), f"C{int(cls_id)}")
                    if cname not in NAMES:
                        continue
                    trees[tree_id].append({
                        "class": cname,
                        "y_norm": cy,
                        "x_norm": cx,
                        "area_norm": w * h,
                        "aspect_ratio": w / h if h > 0 else 1.0,
                        "side": side_name,
                        "side_index": side_index,
                    })
    return trees


# ═══════════════════════ Methods ═══════════════════════

def naive_count(dets):
    counts = Counter(d["class"] for d in dets)
    return {c: counts.get(c, 0) for c in NAMES}


def corrected_naive(dets):
    factors = {"B1": 1.986, "B2": 1.786, "B3": 1.795, "B4": 1.655}
    naive = naive_count(dets)
    return {c: max(0, round(naive[c] / factors[c])) for c in NAMES}


def feature_cluster_count(dets, eps=0.12, min_samples=1):
    counts = {}
    for c in NAMES:
        feats = [[d["y_norm"], np.sqrt(d["area_norm"])] for d in dets if d["class"] == c]
        if not feats:
            counts[c] = 0
            continue
        feats = np.array(feats)
        if len(feats) == 1:
            counts[c] = 1
            continue
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(feats)
        counts[c] = len(set(clustering.labels_))
    return counts


def visibility_count(dets, alpha=1.0, sigma=0.3):
    counts = {}
    for c in NAMES:
        cdets = [d for d in dets if d["class"] == c]
        if not cdets:
            counts[c] = 0
            continue
        total = sum(1.0 / (1.0 + alpha * np.exp(-((d["x_norm"] - 0.5)**2) / (2.0 * sigma**2))) for d in cdets)
        counts[c] = max(0, int(round(total)))
    return counts


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1


def adjacent(si, sj):
    return abs(si - sj) == 1 or {si, sj} == {0, 3}


def learned_graph_count(dets):
    counts = {}
    for c in NAMES:
        cdets = [d for d in dets if d["class"] == c]
        n = len(cdets)
        if n == 0:
            counts[c] = 0
            continue
        if n == 1:
            counts[c] = 1
            continue
        uf = UnionFind(n)
        ty = LEARNED["tol_y"].get(c, 0.10)
        ta = LEARNED["tol_area"].get(c, 0.06)
        for i in range(n):
            for j in range(i + 1, n):
                if cdets[i]["side_index"] == cdets[j]["side_index"]:
                    continue
                si, sj = cdets[i]["side_index"], cdets[j]["side_index"]
                if not adjacent(si, sj):
                    continue
                if abs(cdets[i]["y_norm"] - cdets[j]["y_norm"]) < ty:
                    if abs(np.sqrt(cdets[i]["area_norm"]) - np.sqrt(cdets[j]["area_norm"])) < ta:
                        uf.union(i, j)
        counts[c] = len({uf.find(i) for i in range(n)})
    return counts


def hungarian_match_count(dets, cost_thresh=1.0):
    counts = {}
    for c in NAMES:
        cdets = [d for d in dets if d["class"] == c]
        n = len(cdets)
        if n == 0:
            counts[c] = 0
            continue
        if n == 1:
            counts[c] = 1
            continue
        ty = LEARNED["tol_y"].get(c, 0.10)
        ta = LEARNED["tol_area"].get(c, 0.06)
        ty, ta = max(ty, 1e-6), max(ta, 1e-6)
        uf = UnionFind(n)
        for si in range(4):
            sj = (si + 1) % 4
            idx_i = [k for k, d in enumerate(cdets) if d["side_index"] == si]
            idx_j = [k for k, d in enumerate(cdets) if d["side_index"] == sj]
            if not idx_i or not idx_j:
                continue
            cost = np.zeros((len(idx_i), len(idx_j)))
            for a, ii in enumerate(idx_i):
                for b, jj in enumerate(idx_j):
                    dy = abs(cdets[ii]["y_norm"] - cdets[jj]["y_norm"]) / ty
                    da = abs(np.sqrt(cdets[ii]["area_norm"]) - np.sqrt(cdets[jj]["area_norm"])) / ta
                    cost[a, b] = np.sqrt(dy * dy + da * da)
            row_ind, col_ind = linear_sum_assignment(cost)
            for a, b in zip(row_ind, col_ind):
                if cost[a, b] < cost_thresh:
                    uf.union(idx_i[a], idx_j[b])
        counts[c] = len({uf.find(i) for i in range(n)})
    return counts


def cascade_match_count(dets):
    counts = {}
    for c in NAMES:
        cdets = [d for d in dets if d["class"] == c]
        n = len(cdets)
        if n == 0:
            counts[c] = 0
            continue
        if n == 1:
            counts[c] = 1
            continue
        ty = LEARNED["tol_y"].get(c, 0.10)
        ta = LEARNED["tol_area"].get(c, 0.06)
        ty, ta = max(ty, 1e-6), max(ta, 1e-6)

        clusters = []
        def find_best(d):
            best_k, best_cost = -1, float('inf')
            for k, cl in enumerate(clusters):
                dy = abs(d["y_norm"] - cl["mean_cy"]) / ty
                da = abs(np.sqrt(d["area_norm"]) - cl["mean_area"]) / ta
                cost = np.sqrt(dy * dy + da * da)
                if cost < best_cost:
                    best_cost, best_k = cost, k
            return best_k, best_cost

        for d in cdets:
            d["_cidx"] = len(cdets)  # dummy, not used
        for d in cdets:
            del d["_cidx"]

        # Re-assign local indices
        for ii, d in enumerate(cdets):
            d["_cidx"] = ii

        for side_idx in range(4):
            for d in [x for x in cdets if x["side_index"] == side_idx]:
                bk, bc = find_best(d)
                if bk >= 0 and bc < 1.0:
                    cl = clusters[bk]
                    cl["indices"].add(d["_cidx"])
                    n_prev = len(cl["indices"]) - 1
                    cl["mean_cy"] = (cl["mean_cy"] * n_prev + d["y_norm"]) / (n_prev + 1)
                    cl["mean_area"] = (cl["mean_area"] * n_prev + np.sqrt(d["area_norm"])) / (n_prev + 1)
                else:
                    clusters.append({
                        "indices": {d["_cidx"]},
                        "mean_cy": d["y_norm"],
                        "mean_area": np.sqrt(d["area_norm"]),
                    })

        # merge pass
        changed = True
        while changed:
            changed = False
            new_clusters = []
            merged = set()
            for i in range(len(clusters)):
                if i in merged:
                    continue
                ci = clusters[i]
                for j in range(i + 1, len(clusters)):
                    if j in merged:
                        continue
                    cj = clusters[j]
                    si = {cdets[idx]["side_index"] for idx in ci["indices"]}
                    sj = {cdets[idx]["side_index"] for idx in cj["indices"]}
                    if not any(adjacent(a, b) for a in si for b in sj):
                        continue
                    dy = abs(ci["mean_cy"] - cj["mean_cy"]) / ty
                    da = abs(ci["mean_area"] - cj["mean_area"]) / ta
                    if np.sqrt(dy * dy + da * da) < 1.0:
                        ci["indices"].update(cj["indices"])
                        n_prev = len(ci["indices"]) - len(cj["indices"])
                        n_new = len(ci["indices"])
                        ci["mean_cy"] = (ci["mean_cy"] * n_prev + cj["mean_cy"] * len(cj["indices"])) / n_new
                        ci["mean_area"] = (ci["mean_area"] * n_prev + cj["mean_area"] * len(cj["indices"])) / n_new
                        merged.add(j)
                        changed = True
                new_clusters.append(ci)
                merged.add(i)
            clusters = new_clusters

        counts[c] = len(clusters)
    return counts


# ═══════════════════════ Evaluation ═══════════════════════

def eval_preds(pred, gt):
    err = {c: abs(pred.get(c, 0) - gt.get(c, 0)) for c in NAMES}
    mae = np.mean(list(err.values()))
    within1 = all(e <= 1 for e in err.values())
    return mae, within1, sum(err.values())


# ═══════════════════════ Main ═══════════════════════

def main():
    print("Loading JSON trees (228)...")
    json_trees = {}
    for jp in sorted(JSON_DIR.glob("*.json")):
        tree_id, dets, gt, split = load_json_tree(jp)
        json_trees[tree_id] = {"dets": dets, "gt": gt, "split": split}

    print("Loading TXT trees (all)...")
    txt_trees = load_txt_trees()
    print(f"  Trees from TXT: {len(txt_trees)}")

    methods = {
        "naive": naive_count,
        "corrected": corrected_naive,
        "feature_cluster": feature_cluster_count,
        "visibility": visibility_count,
        "learned_graph": learned_graph_count,
        "hungarian_match": hungarian_match_count,
        "cascade_match": cascade_match_count,
    }

    # ── Part A: Evaluate on 228 JSON trees using JSON data ──
    print("\nEvaluating methods on 228 JSON trees (using JSON annotations)...")
    json_results = []
    for tree_id, data in sorted(json_trees.items()):
        dets = data["dets"]
        gt = data["gt"]
        for mname, mfunc in methods.items():
            pred = mfunc(dets)
            mae, within1, err_sum = eval_preds(pred, gt)
            json_results.append({
                "tree_id": tree_id, "split": data["split"], "method": mname,
                "MAE": mae, "within_1": within1, "err_sum": err_sum,
                **{f"gt_{c}": gt[c] for c in NAMES},
                **{f"pred_{c}": pred.get(c, 0) for c in NAMES},
            })

    json_df = pd.DataFrame(json_results)
    json_summary = []
    for mname in methods:
        sub = json_df[json_df["method"] == mname]
        json_summary.append({
            "method": mname,
            "mean_MAE": round(sub["MAE"].mean(), 4),
            "acc_within_1": round(sub["within_1"].mean() * 100, 2),
            "mean_total_err": round(sub["err_sum"].mean(), 2),
            "score": round(sub["within_1"].mean() * 100 - sub["MAE"].mean() * 10, 2),
        })
    json_summary_df = pd.DataFrame(json_summary).sort_values("score", ascending=False)
    json_summary_df.to_csv(OUT_DIR / "json_228_accuracy.csv", index=False)
    print(json_summary_df.to_string(index=False))

    # ── Part B: Run on 725 non-JSON trees using TXT data ──
    nonjson_ids = sorted(set(txt_trees.keys()) - set(json_trees.keys()))
    print(f"\nRunning on {len(nonjson_ids)} non-JSON trees (TXT only)...")
    nonjson_rows = []
    for tree_id in nonjson_ids:
        dets = txt_trees[tree_id]
        row = {"tree_id": tree_id, "n_dets": len(dets), "n_sides": len(set(d["side_index"] for d in dets))}
        for mname, mfunc in methods.items():
            pred = mfunc(dets)
            for c in NAMES:
                row[f"{mname}_{c}"] = pred.get(c, 0)
            row[f"{mname}_total"] = sum(pred.values())
        nonjson_rows.append(row)

    nonjson_df = pd.DataFrame(nonjson_rows)
    nonjson_df.to_csv(OUT_DIR / "nonjson_725_counts.csv", index=False)

    # Aggregate
    agg_rows = []
    for mname in methods:
        totals = {c: int(nonjson_df[f"{mname}_{c}"].sum()) for c in NAMES}
        agg_rows.append({
            "method": mname,
            **{c: totals[c] for c in NAMES},
            "total": sum(totals.values()),
        })
    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv(OUT_DIR / "nonjson_725_summary.csv", index=False)

    # Dedup ratios
    nonjson_df["naive_total"] = nonjson_df[[f"naive_{c}" for c in NAMES]].sum(axis=1)
    ratio_rows = []
    for mname in methods:
        if mname == "naive":
            continue
        ratio = nonjson_df[f"{mname}_total"] / nonjson_df["naive_total"]
        ratio_rows.append({
            "method": mname,
            "mean_ratio": round(ratio.mean(), 4),
            "median_ratio": round(ratio.median(), 4),
            "std_ratio": round(ratio.std(), 4),
        })
    ratio_df = pd.DataFrame(ratio_rows).sort_values("mean_ratio", ascending=False)
    ratio_df.to_csv(OUT_DIR / "nonjson_725_ratios.csv", index=False)

    print("\n" + "=" * 60)
    print("NON-JSON TOTAL COUNTS BY METHOD")
    print("=" * 60)
    print(agg_df.to_string(index=False))

    print("\n" + "=" * 60)
    print("NON-JSON DEDUP RATIO (dedup_total / naive_total)")
    print("=" * 60)
    print(ratio_df.to_string(index=False))

    # ── Part C: Per-class comparison for non-JSON ──
    print("\n" + "=" * 60)
    print("NON-JSON PER-CLASS COUNTS")
    print("=" * 60)
    for c in NAMES:
        print(f"\nClass {c}:")
        sub = agg_df[["method", c]].sort_values(c, ascending=False)
        print(sub.to_string(index=False))

    print(f"\nAll outputs saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
