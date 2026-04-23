"""
Final dedup comparison:
- 228 JSON trees: use JSON annotations + GT for accuracy eval (same as v1/v2/v3)
- 725 non-JSON trees: use TXT labels, run dedup methods, output counts (no GT)
- Includes benchmark leaders from v5/v6/v7/v8 for one-pass comparison.
"""

import ast
import json
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN
from sklearn.linear_model import Ridge

from dedup_research_v6 import (
    load_v5_reference_params as load_v6_selector_params,
    selector_v6 as v6_selector_count,
)
from dedup_research_v7 import (
    stacking_bracketed as stacking_bracketed_v7_count,
    stacking_density_corrected as stacking_density_v7_count,
)
from dedup_research_v8 import (
    entropy_modulated as entropy_modulated_v8_count,
    v8_entropy_stacking as v8_entropy_stacking_count,
)

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


Y_PRIOR = None


def compute_y_prior(tree_data):
    global Y_PRIOR
    y_vals = {c: [] for c in NAMES}
    for _, dets, _, _ in tree_data:
        for d in dets:
            y_vals[d["class"]].append(d["y_norm"])

    Y_PRIOR = {}
    for c in NAMES:
        vals = np.array(y_vals[c]) if y_vals[c] else np.array([0.5])
        Y_PRIOR[c] = {
            "y_mean": float(np.mean(vals)),
            "y_std": float(np.std(vals)) if len(vals) > 1 else 0.1,
            "y_min": float(np.min(vals)),
            "y_max": float(np.max(vals)),
        }
    return Y_PRIOR


def adaptive_visibility_count(dets, base_alpha=1.0, base_sigma=0.3,
                               alpha_min=0.5, alpha_max=1.6,
                               sigma_min=0.12, sigma_max=0.55):
    n_total = len(dets)
    y_vals = [d["y_norm"] for d in dets]
    y_span = max(y_vals) - min(y_vals) if y_vals else 0.5

    density = n_total / 12.0
    alpha = base_alpha * (1.35 - 0.35 * min(density, 1.6))
    sigma = base_sigma * (0.55 + 0.45 * min(density, 1.6))

    if y_span > 0.7:
        sigma *= 0.88
        alpha *= 1.08
    elif y_span < 0.3:
        sigma *= 1.18
        alpha *= 0.92

    alpha = float(np.clip(alpha, alpha_min, alpha_max))
    sigma = float(np.clip(sigma, sigma_min, sigma_max))
    return visibility_count(dets, alpha=alpha, sigma=sigma)


def adaptive_corrected_count(dets):
    n_total = len(dets)
    base_factors = {"B1": 1.986, "B2": 1.786, "B3": 1.795, "B4": 1.655}
    dup_rate = 2.05 - 0.014 * n_total
    dup_rate = np.clip(dup_rate, 1.45, 2.10)
    scale = float(dup_rate / 1.79)
    factors = {c: base_factors[c] * scale for c in NAMES}
    naive = naive_count(dets)
    return {c: max(0, round(naive[c] / factors[c])) for c in NAMES}


def density_scaled_visibility(dets):
    n_total = len(dets)
    vis = visibility_count(dets)
    density_boost = 1.0 + 0.025 * (n_total - 12) / 12.0
    density_boost = np.clip(density_boost, 0.92, 1.15)
    return {c: max(0, int(round(vis[c] * density_boost))) for c in NAMES}


def class_aware_visibility_count(dets, alpha_B1B4=1.0, alpha_B2B3=0.65,
                                  sigma_B1B4=0.35, sigma_B2B3=0.45):
    counts = {}
    for c in NAMES:
        cdets = [d for d in dets if d["class"] == c]
        if not cdets:
            counts[c] = 0
            continue
        alpha = alpha_B2B3 if c in ("B2", "B3") else alpha_B1B4
        sigma = sigma_B2B3 if c in ("B2", "B3") else sigma_B1B4
        total = sum(1.0 / (1.0 + alpha * np.exp(-((d["x_norm"] - 0.5) ** 2) / (2.0 * sigma ** 2))) for d in cdets)
        counts[c] = max(0, int(round(total)))
    return counts


def side_coverage_count(dets):
    vis = visibility_count(dets)
    naive = naive_count(dets)
    counts = {}
    for c in NAMES:
        cdets = [d for d in dets if d["class"] == c]
        if not cdets:
            counts[c] = 0
            continue
        per_side = Counter(d["side_index"] for d in cdets)
        max_per_side = max(per_side.values()) if per_side else 0
        pred = vis[c]
        pred = max(pred, max_per_side)
        pred = min(pred, naive[c])
        counts[c] = pred
    return counts


def hybrid_visibility_corrected(dets, vis_weight=0.6):
    vis = visibility_count(dets)
    corr = adaptive_corrected_count(dets)
    return {c: max(0, int(round(vis_weight * vis[c] + (1 - vis_weight) * corr[c]))) for c in NAMES}


def ordinal_prior_count(dets, flip_threshold_sigma=2.0):
    global Y_PRIOR
    if Y_PRIOR is None:
        raise RuntimeError("Y_PRIOR not computed.")

    counts = visibility_count(dets)
    flagged = {c: 0 for c in NAMES}
    for d in dets:
        c = d["class"]
        if c not in Y_PRIOR:
            continue
        y = d["y_norm"]
        prior = Y_PRIOR[c]
        if abs(y - prior["y_mean"]) > flip_threshold_sigma * prior["y_std"]:
            flagged[c] += 1

    b2_flag_rate = flagged["B2"] / max(counts["B2"], 1)
    b3_flag_rate = flagged["B3"] / max(counts["B3"], 1)

    if b2_flag_rate > b3_flag_rate + 0.15 and counts["B2"] > 0:
        n_swap = max(1, int(round(0.35 * counts["B2"] * b2_flag_rate)))
        counts["B2"] = max(0, counts["B2"] - n_swap)
        counts["B3"] = counts["B3"] + n_swap
    elif b3_flag_rate > b2_flag_rate + 0.15 and counts["B3"] > 0:
        n_swap = max(1, int(round(0.35 * counts["B3"] * b3_flag_rate)))
        counts["B3"] = max(0, counts["B3"] - n_swap)
        counts["B2"] = counts["B2"] + n_swap

    return counts


def relaxed_matching_count(dets, y_thresh=0.15, area_thresh=0.12, cx_thresh=0.35):
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
        for i in range(n):
            for j in range(i + 1, n):
                di, dj = cdets[i], cdets[j]
                if di["side_index"] == dj["side_index"]:
                    continue
                if abs(di["y_norm"] - dj["y_norm"]) > y_thresh:
                    continue
                da = np.sqrt(di["area_norm"])
                db = np.sqrt(dj["area_norm"])
                if abs(da - db) > area_thresh:
                    continue
                if abs(di["x_norm"] - dj["x_norm"]) > cx_thresh:
                    continue
                uf.union(i, j)
        counts[c] = len({uf.find(i) for i in range(n)})
    return counts


def median_ensemble(preds_list):
    result = {}
    for c in NAMES:
        vals = [p[c] for p in preds_list]
        result[c] = int(round(np.median(vals)))
    return result


def trimmed_mean_ensemble(preds_list):
    result = {}
    for c in NAMES:
        vals = sorted([p[c] for p in preds_list])
        if len(vals) <= 2:
            result[c] = int(round(np.mean(vals)))
        else:
            result[c] = int(round(np.mean(vals[1:-1])))
    return result


def load_v5_reference_params():
    ref_path = BASE / "reports" / "dedup_research_v5" / "method_comparison_v5.csv"
    df = pd.read_csv(ref_path)

    def row(method):
        r = df[df["method"] == method].iloc[0].to_dict()
        return r

    ens_row = row("best_ensemble_grid")
    subset = ast.literal_eval(ens_row["subset"])
    return {
        "best_visibility": row("best_visibility_grid"),
        "best_class_aware": row("best_class_aware_grid"),
        "best_relaxed": row("best_relaxed_grid"),
        "best_ensemble_subset": subset,
        "best_ensemble_agg": ens_row["agg"],
    }


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

    refs = load_v5_reference_params()
    v6_selector_params = load_v6_selector_params()
    compute_y_prior(list((tid, data["dets"], data["gt"], data["split"]) for tid, data in json_trees.items()))

    best_vis = refs["best_visibility"]
    best_cav = refs["best_class_aware"]
    best_rel = refs["best_relaxed"]
    best_ensemble_subset = tuple(refs["best_ensemble_subset"])
    best_ensemble_agg = refs["best_ensemble_agg"]

    direct_methods = {
        "naive": naive_count,
        "corrected": corrected_naive,
        "visibility": visibility_count,
        "adaptive_visibility": adaptive_visibility_count,
        "adaptive_corrected": adaptive_corrected_count,
        "v6_selector": lambda dets, params=v6_selector_params: v6_selector_count(dets, params),
        "stacking_density_v7": stacking_density_v7_count,
        "stacking_bracketed_v7": stacking_bracketed_v7_count,
        "entropy_modulated_v8": entropy_modulated_v8_count,
        "v8_entropy_stacking": v8_entropy_stacking_count,
        "density_scaled_vis": density_scaled_visibility,
        "class_aware_vis": class_aware_visibility_count,
        "side_coverage": side_coverage_count,
        "hybrid_vis_corr": hybrid_visibility_corrected,
        "ordinal_prior": ordinal_prior_count,
        "relaxed_match": relaxed_matching_count,
        "best_visibility_grid": lambda dets, a=float(best_vis["alpha"]), s=float(best_vis["sigma"]): visibility_count(dets, alpha=a, sigma=s),
        "best_class_aware_grid": lambda dets,
                                         a14=float(best_cav["alpha_B1B4"]),
                                         a23=float(best_cav["alpha_B2B3"]),
                                         s14=float(best_cav["sigma_B1B4"]),
                                         s23=float(best_cav["sigma_B2B3"]): class_aware_visibility_count(dets, a14, a23, s14, s23),
        "best_relaxed_grid": lambda dets,
                                      yt=float(best_rel["y_thresh"]),
                                      at=float(best_rel["area_thresh"]),
                                      ct=float(best_rel["cx_thresh"]): relaxed_matching_count(dets, yt, at, ct),
    }

    ensemble_methods = {
        "best_ensemble_grid": None,
        "naive_mean_ensemble": None,
    }

    methods = {**direct_methods, **ensemble_methods}

    # ── Part A: Evaluate on 228 JSON trees using JSON data ──
    print("\nEvaluating methods on 228 JSON trees (using JSON annotations)...")
    json_results = []
    for tree_id, data in sorted(json_trees.items()):
        dets = data["dets"]
        gt = data["gt"]
        pred_cache = {}
        for mname, mfunc in direct_methods.items():
            pred = mfunc(dets)
            pred_cache[mname] = pred
            mae, within1, err_sum = eval_preds(pred, gt)
            json_results.append({
                "tree_id": tree_id, "split": data["split"], "method": mname,
                "MAE": mae, "within_1": within1, "err_sum": err_sum,
                **{f"gt_{c}": gt[c] for c in NAMES},
                **{f"pred_{c}": pred.get(c, 0) for c in NAMES},
            })

        ensemble_input = [pred_cache[name] for name in best_ensemble_subset]
        if best_ensemble_agg == "median":
            pred = median_ensemble(ensemble_input)
        else:
            pred = trimmed_mean_ensemble(ensemble_input)
        mae, within1, err_sum = eval_preds(pred, gt)
        json_results.append({
            "tree_id": tree_id, "split": data["split"], "method": "best_ensemble_grid",
            "MAE": mae, "within_1": within1, "err_sum": err_sum,
            **{f"gt_{c}": gt[c] for c in NAMES},
            **{f"pred_{c}": pred.get(c, 0) for c in NAMES},
        })

        mean_pred = {}
        heuristic_pool = [
            "visibility",
            "adaptive_visibility",
            "adaptive_corrected",
            "corrected",
            "density_scaled_vis",
            "class_aware_vis",
            "side_coverage",
            "hybrid_vis_corr",
            "ordinal_prior",
            "relaxed_match",
        ]
        for c in NAMES:
            mean_pred[c] = int(round(np.mean([pred_cache[name][c] for name in heuristic_pool])))
        mae, within1, err_sum = eval_preds(mean_pred, gt)
        json_results.append({
            "tree_id": tree_id, "split": data["split"], "method": "naive_mean_ensemble",
            "MAE": mae, "within_1": within1, "err_sum": err_sum,
            **{f"gt_{c}": gt[c] for c in NAMES},
            **{f"pred_{c}": mean_pred.get(c, 0) for c in NAMES},
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
    nonjson_preds = {name: [] for name in direct_methods}
    for tree_id in nonjson_ids:
        dets = txt_trees[tree_id]
        row = {"tree_id": tree_id, "n_dets": len(dets), "n_sides": len(set(d["side_index"] for d in dets))}
        for mname, mfunc in direct_methods.items():
            pred = mfunc(dets)
            nonjson_preds[mname].append(pred)
            for c in NAMES:
                row[f"{mname}_{c}"] = pred.get(c, 0)
            row[f"{mname}_total"] = sum(pred.values())
        nonjson_rows.append(row)

    heuristic_pool = [
        "visibility",
        "adaptive_visibility",
        "adaptive_corrected",
        "corrected",
        "density_scaled_vis",
        "class_aware_vis",
        "side_coverage",
        "hybrid_vis_corr",
        "ordinal_prior",
        "relaxed_match",
    ]
    best_ensemble_preds = []
    naive_mean_preds = []
    for i in range(len(nonjson_rows)):
        ensemble_input = [nonjson_preds[name][i] for name in best_ensemble_subset]
        if best_ensemble_agg == "median":
            best_ensemble_preds.append(median_ensemble(ensemble_input))
        else:
            best_ensemble_preds.append(trimmed_mean_ensemble(ensemble_input))

        mean_pred = {}
        for c in NAMES:
            mean_pred[c] = int(round(np.mean([nonjson_preds[name][i][c] for name in heuristic_pool])))
        naive_mean_preds.append(mean_pred)

    for row, pred in zip(nonjson_rows, best_ensemble_preds):
        for c in NAMES:
            row[f"best_ensemble_grid_{c}"] = pred.get(c, 0)
        row["best_ensemble_grid_total"] = sum(pred.values())

    for row, pred in zip(nonjson_rows, naive_mean_preds):
        for c in NAMES:
            row[f"naive_mean_ensemble_{c}"] = pred.get(c, 0)
        row["naive_mean_ensemble_total"] = sum(pred.values())

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
