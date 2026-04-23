"""
Apply v1/v2/v3 dedup methods to ALL trees (953) using YOLO TXT labels.
For 228 JSON trees: evaluate accuracy vs GT.
For 725 non-JSON trees: output dedup counts for comparison.

Methods run (no-GT required):
  naive, corrected, feature_cluster, visibility,
  learned_graph, hungarian_match, cascade_match
"""

import json
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
JSON_DIR = BASE / "json"
LABEL_DIRS = [BASE / "dataset" / "labels" / s for s in ["train", "val", "test"]]
OUT_DIR = BASE / "reports" / "nonjson_dedup_compare"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NAMES = ["B1", "B2", "B3", "B4"]
CLASS_MAP = {"B1": 0, "B2": 1, "B3": 2, "B4": 3}
INV_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}
SIDE_NAMES = ["sisi_1", "sisi_2", "sisi_3", "sisi_4"]

# ── Load learned thresholds from v3 ──
LEARNED = json.loads((BASE / "reports" / "dedup_research_v3" / "learned_thresholds.json").read_text(encoding="utf-8"))


def parse_txt_labels():
    """Parse all YOLO TXT files into tree_id -> list of detections."""
    trees = defaultdict(list)
    for lbl_dir in LABEL_DIRS:
        for txt_path in lbl_dir.glob("*.txt"):
            stem = txt_path.stem  # e.g. DAMIMAS_A21B_0001_1
            parts = stem.rsplit("_", 1)
            tree_id = parts[0]
            side_num = int(parts[1])  # 1-8
            side_name = f"sisi_{side_num}"
            side_index = side_num - 1

            with open(txt_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    cls_id, cx, cy, w, h = map(float, line.split())
                    cls_id = int(cls_id)
                    cname = INV_CLASS_MAP.get(cls_id, f"C{cls_id}")
                    if cname not in NAMES:
                        continue
                    trees[tree_id].append({
                        "class": cname,
                        "bbox_yolo": [cx, cy, w, h],
                        "y_norm": cy,
                        "x_norm": cx,
                        "area_norm": w * h,
                        "aspect_ratio": w / h if h > 0 else 1.0,
                        "side": side_name,
                        "side_index": side_index,
                    })
    return trees


def load_json_gt():
    """Load GT for 228 JSON trees."""
    gt = {}
    for jp in JSON_DIR.glob("*.json"):
        data = json.loads(jp.read_text(encoding="utf-8"))
        tree_id = data.get("tree_name", data.get("tree_id", jp.stem))
        split = data.get("split", "unknown")
        gt[tree_id] = {
            "split": split,
            "by_class": {c: data["summary"]["by_class"].get(c, 0) for c in NAMES},
        }
    return gt


# ── Methods ──

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
        feats = []
        for d in dets:
            if d["class"] == c:
                feats.append([d["y_norm"], np.sqrt(d["area_norm"])])
        if not feats:
            counts[c] = 0
            continue
        feats = np.array(feats)
        if len(feats) == 1:
            counts[c] = 1
            continue
        from sklearn.cluster import DBSCAN
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
        total = 0.0
        for d in cdets:
            cx = d["x_norm"]
            vis = 1.0 + alpha * np.exp(-((cx - 0.5) ** 2) / (2.0 * sigma ** 2))
            total += 1.0 / vis
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
    learned = LEARNED
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
        ty = learned["tol_y"].get(c, 0.10)
        ta = learned["tol_area"].get(c, 0.06)
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
        comps = {uf.find(i) for i in range(n)}
        counts[c] = len(comps)
    return counts


def hungarian_match_count(dets, cost_thresh=1.0):
    learned = LEARNED
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
        ty = learned["tol_y"].get(c, 0.10)
        ta = learned["tol_area"].get(c, 0.06)
        if ty <= 0:
            ty = 1e-6
        if ta <= 0:
            ta = 1e-6

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
        comps = {uf.find(i) for i in range(n)}
        counts[c] = len(comps)
    return counts


def cascade_match_count(dets):
    learned = LEARNED
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
        ty = learned["tol_y"].get(c, 0.10)
        ta = learned["tol_area"].get(c, 0.06)
        if ty <= 0:
            ty = 1e-6
        if ta <= 0:
            ta = 1e-6

        clusters = []

        def find_best_cluster(d):
            best_k = -1
            best_cost = float('inf')
            for k, cl in enumerate(clusters):
                dy = abs(d["y_norm"] - cl["mean_cy"]) / ty
                da = abs(np.sqrt(d["area_norm"]) - cl["mean_area"]) / ta
                cost = np.sqrt(dy * dy + da * da)
                if cost < best_cost:
                    best_cost = cost
                    best_k = k
            return best_k, best_cost

        for ii, d in enumerate(cdets):
            d["_cidx"] = ii

        for side_idx in range(4):
            side_dets = [d for d in cdets if d["side_index"] == side_idx]
            for d in side_dets:
                best_k, best_cost = find_best_cluster(d)
                if best_k >= 0 and best_cost < 1.0:
                    cl = clusters[best_k]
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
                    sides_i = {cdets[idx]["side_index"] for idx in ci["indices"]}
                    sides_j = {cdets[idx]["side_index"] for idx in cj["indices"]}
                    has_adj = any(adjacent(a, b) for a in sides_i for b in sides_j)
                    if not has_adj:
                        continue
                    dy = abs(ci["mean_cy"] - cj["mean_cy"]) / ty
                    da = abs(ci["mean_area"] - cj["mean_area"]) / ta
                    cost = np.sqrt(dy * dy + da * da)
                    if cost < 1.0:
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


# ── Evaluation helpers ──

def compute_metrics(pred, gt):
    err = {c: abs(pred.get(c, 0) - gt.get(c, 0)) for c in NAMES}
    mae = np.mean(list(err.values()))
    within_1 = all(e <= 1 for e in err.values())
    return mae, within_1, sum(err.values())


def main():
    print("Parsing TXT labels for all trees...")
    tree_dets = parse_txt_labels()
    print(f"Total trees from TXT: {len(tree_dets)}")

    print("Loading JSON GT...")
    json_gt = load_json_gt()
    print(f"Trees with JSON GT: {len(json_gt)}")

    methods = {
        "naive": naive_count,
        "corrected": corrected_naive,
        "feature_cluster": feature_cluster_count,
        "visibility": visibility_count,
        "learned_graph": learned_graph_count,
        "hungarian_match": hungarian_match_count,
        "cascade_match": cascade_match_count,
    }

    rows = []
    json_eval = {m: {"maes": [], "within1": [], "err_sums": []} for m in methods}

    for tree_id, dets in sorted(tree_dets.items()):
        has_json = tree_id in json_gt
        split = json_gt[tree_id]["split"] if has_json else "nonjson"

        row = {
            "tree_id": tree_id,
            "split": split,
            "has_json": has_json,
            "n_dets": len(dets),
            "n_sides": len(set(d["side_index"] for d in dets)),
        }

        for mname, mfunc in methods.items():
            pred = mfunc(dets)
            for c in NAMES:
                row[f"{mname}_{c}"] = pred.get(c, 0)
            row[f"{mname}_total"] = sum(pred.values())

            if has_json:
                gt = json_gt[tree_id]["by_class"]
                mae, within1, err_sum = compute_metrics(pred, gt)
                json_eval[mname]["maes"].append(mae)
                json_eval[mname]["within1"].append(within1)
                json_eval[mname]["err_sums"].append(err_sum)

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "all_trees_dedup_counts.csv", index=False)

    # ── JSON evaluation summary ──
    eval_rows = []
    for mname, vals in json_eval.items():
        if not vals["maes"]:
            continue
        eval_rows.append({
            "method": mname,
            "n_trees": len(vals["maes"]),
            "mean_MAE": round(np.mean(vals["maes"]), 4),
            "acc_within_1": round(np.mean(vals["within1"]) * 100, 2),
            "mean_total_error": round(np.mean(vals["err_sums"]), 2),
            "score": round(np.mean(vals["within1"]) * 100 - np.mean(vals["maes"]) * 10, 2),
        })

    eval_df = pd.DataFrame(eval_rows).sort_values("score", ascending=False)
    eval_df.to_csv(OUT_DIR / "json_accuracy_validation.csv", index=False)

    # ── Non-JSON summary: compare methods ──
    nonjson_df = df[~df["has_json"]].copy()
    summary_rows = []
    for mname in methods:
        totals = {c: int(nonjson_df[f"{mname}_{c}"].sum()) for c in NAMES}
        summary_rows.append({
            "method": mname,
            "B1": totals["B1"],
            "B2": totals["B2"],
            "B3": totals["B3"],
            "B4": totals["B4"],
            "total": sum(totals.values()),
            "trees": len(nonjson_df),
        })

    nonjson_summary = pd.DataFrame(summary_rows)
    nonjson_summary.to_csv(OUT_DIR / "nonjson_counts_by_method.csv", index=False)

    # ── Per-tree dedup ratio for non-JSON ──
    nonjson_df["naive_total"] = nonjson_df["naive_B1"] + nonjson_df["naive_B2"] + nonjson_df["naive_B3"] + nonjson_df["naive_B4"]
    for mname in methods:
        if mname == "naive":
            continue
        nonjson_df[f"{mname}_ratio"] = nonjson_df[f"{mname}_total"] / nonjson_df["naive_total"]

    ratio_rows = []
    for mname in methods:
        if mname == "naive":
            continue
        ratio_rows.append({
            "method": mname,
            "mean_ratio_to_naive": round(nonjson_df[f"{mname}_ratio"].mean(), 4),
            "median_ratio": round(nonjson_df[f"{mname}_ratio"].median(), 4),
            "std_ratio": round(nonjson_df[f"{mname}_ratio"].std(), 4),
        })

    ratio_df = pd.DataFrame(ratio_rows).sort_values("mean_ratio_to_naive", ascending=False)
    ratio_df.to_csv(OUT_DIR / "nonjson_dedup_ratios.csv", index=False)

    # ── Print summary ──
    print("\n" + "=" * 60)
    print("ACCURACY ON 228 JSON TREES (validation)")
    print("=" * 60)
    print(eval_df.to_string(index=False))

    print("\n" + "=" * 60)
    print("NON-JSON (725 trees) — TOTAL COUNTS BY METHOD")
    print("=" * 60)
    print(nonjson_summary.to_string(index=False))

    print("\n" + "=" * 60)
    print("NON-JSON — DEDUP RATIO (dedup / naive)")
    print("=" * 60)
    print(ratio_df.to_string(index=False))

    print(f"\nOutputs saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
