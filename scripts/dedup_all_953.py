"""
Run ALL dedup methods on all 953 trees (228 JSON + 725 TXT).
For JSON trees: also compute accuracy vs GT.
For all trees: output raw counts per method.
No training, no embeddings — heuristic only.
"""

from __future__ import annotations

import json
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
JSON_DIR = BASE / "json"
LABEL_DIRS = [BASE / "dataset" / "labels" / s for s in ["train", "val", "test"]]
OUT_DIR = BASE / "reports" / "dedup_all_953"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NAMES = ["B1", "B2", "B3", "B4"]
INV_CLASS_MAP = {0: "B1", 1: "B2", 2: "B3", 3: "B4"}


# ─── loaders ────────────────────────────────────────────────

def _parse_det(ann, side, si):
    cx, cy, w, h = ann["bbox_yolo"]
    return {
        "class": ann["class_name"],
        "y_norm": cy, "x_norm": cx,
        "area_norm": w * h,
        "aspect_ratio": w / h if h > 0 else 1.0,
        "side": side, "side_index": si,
    }


def load_json_trees() -> Dict[str, dict]:
    trees = {}
    for jp in sorted(JSON_DIR.glob("*.json")):
        data = json.loads(jp.read_text(encoding="utf-8"))
        tree_id = data.get("tree_name", data.get("tree_id", jp.stem))
        gt = {c: data["summary"]["by_class"].get(c, 0) for c in NAMES}
        dets = []
        for side, sd in data["images"].items():
            si = sd.get("side_index", int(side.replace("sisi_", "")) - 1)
            for ann in sd.get("annotations", []):
                if "bbox_yolo" in ann:
                    dets.append(_parse_det(ann, side, si))
        trees[tree_id] = {"dets": dets, "gt": gt, "split": data.get("split", "unknown"), "source": "json"}
    return trees


def load_txt_trees() -> Dict[str, dict]:
    raw: Dict[str, list] = defaultdict(list)
    for lbl_dir in LABEL_DIRS:
        for txt in lbl_dir.glob("*.txt"):
            parts = txt.stem.rsplit("_", 1)
            tree_id, side_num = parts[0], int(parts[1])
            side_name = f"sisi_{side_num}"
            si = side_num - 1
            for line in txt.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                cls_id, cx, cy, w, h = map(float, line.split())
                cname = INV_CLASS_MAP.get(int(cls_id))
                if cname is None:
                    continue
                raw[tree_id].append({
                    "class": cname,
                    "y_norm": cy, "x_norm": cx,
                    "area_norm": w * h,
                    "aspect_ratio": w / h if h > 0 else 1.0,
                    "side": side_name, "side_index": si,
                })
    return {tid: {"dets": dets, "gt": None, "split": None, "source": "txt"}
            for tid, dets in raw.items()}


# ─── methods ────────────────────────────────────────────────

def naive_count(dets):
    c = Counter(d["class"] for d in dets)
    return {cl: c.get(cl, 0) for cl in NAMES}


def corrected_naive(dets):
    factors = {"B1": 1.986, "B2": 1.786, "B3": 1.795, "B4": 1.655}
    n = naive_count(dets)
    return {c: max(0, round(n[c] / factors[c])) for c in NAMES}


def adaptive_corrected(dets):
    n_total = len(dets)
    base = {"B1": 1.986, "B2": 1.786, "B3": 1.795, "B4": 1.655}
    dup_rate = float(np.clip(2.05 - 0.014 * n_total, 1.45, 2.10))
    scale = dup_rate / 1.79
    n = naive_count(dets)
    return {c: max(0, round(n[c] / (base[c] * scale))) for c in NAMES}


def visibility_count(dets, alpha=1.0, sigma=0.3):
    out = {}
    for c in NAMES:
        cd = [d for d in dets if d["class"] == c]
        if not cd:
            out[c] = 0
            continue
        total = sum(
            1.0 / (1.0 + alpha * np.exp(-((d["x_norm"] - 0.5) ** 2) / (2.0 * sigma ** 2)))
            for d in cd
        )
        out[c] = max(0, int(round(total)))
    return out


def best_visibility_grid(dets):
    return visibility_count(dets, alpha=0.8, sigma=0.35)


def adaptive_visibility(dets):
    n_total = len(dets)
    y_vals = [d["y_norm"] for d in dets]
    y_span = (max(y_vals) - min(y_vals)) if y_vals else 0.5
    density = n_total / 12.0
    alpha = 1.0 * (1.35 - 0.35 * min(density, 1.6))
    sigma = 0.3 * (0.55 + 0.45 * min(density, 1.6))
    if y_span > 0.7:
        sigma *= 0.88; alpha *= 1.08
    elif y_span < 0.3:
        sigma *= 1.18; alpha *= 0.92
    return visibility_count(dets, alpha=float(np.clip(alpha, 0.5, 1.6)), sigma=float(np.clip(sigma, 0.12, 0.55)))


def class_aware_vis(dets):
    out = {}
    for c in NAMES:
        cd = [d for d in dets if d["class"] == c]
        if not cd:
            out[c] = 0
            continue
        alpha = 0.65 if c in ("B2", "B3") else 1.0
        sigma = 0.45 if c in ("B2", "B3") else 0.35
        out[c] = max(0, int(round(sum(
            1.0 / (1.0 + alpha * np.exp(-((d["x_norm"] - 0.5) ** 2) / (2.0 * sigma ** 2)))
            for d in cd
        ))))
    return out


def density_scaled_vis(dets):
    n_total = len(dets)
    vis = visibility_count(dets)
    boost = float(np.clip(1.0 + 0.025 * (n_total - 12) / 12.0, 0.92, 1.15))
    return {c: max(0, int(round(vis[c] * boost))) for c in NAMES}


def side_coverage(dets):
    vis = visibility_count(dets)
    n = naive_count(dets)
    out = {}
    for c in NAMES:
        cd = [d for d in dets if d["class"] == c]
        if not cd:
            out[c] = 0
            continue
        max_per_side = max(Counter(d["side_index"] for d in cd).values())
        out[c] = min(max(vis[c], max_per_side), n[c])
    return out


def hybrid_vis_corr(dets, w=0.6):
    vis = visibility_count(dets)
    corr = adaptive_corrected(dets)
    return {c: max(0, int(round(w * vis[c] + (1 - w) * corr[c]))) for c in NAMES}


# ─── UnionFind for matching methods ──────────────────────────

class _UF:
    def __init__(self, n):
        self.p = list(range(n)); self.r = [0] * n
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry: return
        if self.r[rx] < self.r[ry]: rx, ry = ry, rx
        self.p[ry] = rx
        if self.r[rx] == self.r[ry]: self.r[rx] += 1


def relaxed_match(dets, y_thresh=0.15, area_thresh=0.12, cx_thresh=0.35):
    out = {}
    for c in NAMES:
        cd = [d for d in dets if d["class"] == c]
        n = len(cd)
        if n == 0: out[c] = 0; continue
        if n == 1: out[c] = 1; continue
        uf = _UF(n)
        for i in range(n):
            for j in range(i + 1, n):
                if cd[i]["side_index"] == cd[j]["side_index"]: continue
                if abs(cd[i]["y_norm"] - cd[j]["y_norm"]) > y_thresh: continue
                if abs(np.sqrt(cd[i]["area_norm"]) - np.sqrt(cd[j]["area_norm"])) > area_thresh: continue
                if abs(cd[i]["x_norm"] - cd[j]["x_norm"]) > cx_thresh: continue
                uf.union(i, j)
        out[c] = len({uf.find(i) for i in range(n)})
    return out


# ─── v6 selector (inline, no file dep for params) ────────────

def _v6_meta(dets):
    n = naive_count(dets)
    total = sum(n.values())
    meta = {"total_det": total}
    for c in NAMES:
        cd = [d for d in dets if d["class"] == c]
        sides = set(d["side_index"] for d in cd)
        meta[f"{c}_naive"] = n[c]
        meta[f"{c}_activesides"] = len(sides)
        meta[f"{c}_ratio"] = n[c] / len(sides) if sides else 0.0
        meta[f"{c}_maxside"] = max(Counter(d["side_index"] for d in cd).values()) if cd else 0
    return meta


def v6_selector(dets):
    import dedup_research_v6 as _v6
    # load params from CSV (cached globally below)
    return _v6.selector_v6(dets, _V6_PARAMS)


def _stacking_bracketed(dets):
    import dedup_research_v7 as _v7
    return _v7.stacking_bracketed(dets)


def _b2_b4_boosted(dets):
    import dedup_research_v8 as _v8
    return _v8.b2_b4_boosted(dets)


def _floor_anchored(dets):
    import dedup_research_v8 as _v8
    return _v8.floor_anchored(dets)


def _per_side_median(dets):
    import dedup_research_v8 as _v8
    return _v8.per_side_median(dets)


def _entropy_modulated(dets):
    import dedup_research_v8 as _v8
    return _v8.entropy_modulated(dets)


def _v8_entropy_stacking(dets):
    import dedup_research_v8 as _v8
    return _v8.v8_entropy_stacking(dets)


def _ordinal_b3(dets):
    import dedup_research_v7 as _v7
    return _v7.ordinal_modulated_b3(dets)


def _side_agreement(dets):
    import dedup_research_v8 as _v8
    return _v8.side_agreement_corrected(dets)


def _multi_consensus(dets):
    import dedup_research_v8 as _v8
    return _v8.multi_estimator_consensus(dets)


def _stacking_density(dets):
    import dedup_research_v7 as _v7
    return _v7.stacking_density_corrected(dets)


def v9_selector(dets):
    import dedup_research_v9 as _v9
    pred, _ = _v9.selector_v9_with_meta(dets, _V6_PARAMS)
    return pred


def v9_b2_median_v6(dets):
    import dedup_research_v9 as _v9
    return _v9.b2_median_v6_factory(_V6_PARAMS)(dets)


def v9_median_strong5(dets):
    import dedup_research_v9 as _v9
    return _v9.median_strong5_factory(_V6_PARAMS)(dets)


# ─── method registry ─────────────────────────────────────────

METHOD_GROUPS = {
    # statistical/divisor
    "naive": naive_count,
    "corrected": corrected_naive,
    "adaptive_corrected": adaptive_corrected,
    "density_scaled_vis": density_scaled_vis,
    "side_coverage": side_coverage,
    "hybrid_vis_corr": hybrid_vis_corr,
    # visibility/geometry
    "visibility": visibility_count,
    "best_visibility_grid": best_visibility_grid,
    "adaptive_visibility": adaptive_visibility,
    "class_aware_vis": class_aware_vis,
    # matching/graph (works ok on JSON; noisy on TXT)
    "relaxed_match": relaxed_match,
    # v6-v9 selectors
    "v6_selector": v6_selector,
    "v7_stacking_density": _stacking_density,
    "v7_stacking_bracketed": _stacking_bracketed,
    "v7_ordinal_b3": _ordinal_b3,
    "v8_entropy_modulated": _entropy_modulated,
    "v8_entropy_stacking": _v8_entropy_stacking,
    "v8_b2_b4_boosted": _b2_b4_boosted,
    "v8_floor_anchor_50": _floor_anchored,
    "v8_per_side_median": _per_side_median,
    "v8_side_agreement": _side_agreement,
    "v8_multi_consensus": _multi_consensus,
    "v9_selector": v9_selector,
    "v9_b2_median_v6": v9_b2_median_v6,
    "v9_median_strong5": v9_median_strong5,
}

_V6_PARAMS: dict = {}


def _load_v6_params():
    global _V6_PARAMS
    import dedup_research_v6 as _v6
    _V6_PARAMS = _v6.load_v5_reference_params()


# ─── eval helpers ────────────────────────────────────────────

def _within1(pred, gt):
    return all(abs(pred.get(c, 0) - gt.get(c, 0)) <= 1 for c in NAMES)


def _mae(pred, gt):
    return float(np.mean([abs(pred.get(c, 0) - gt.get(c, 0)) for c in NAMES]))


# ─── main ────────────────────────────────────────────────────

def main():
    print("Loading v6 params...")
    _load_v6_params()

    print("Loading JSON trees (228)...")
    json_trees = load_json_trees()

    print("Loading TXT trees...")
    txt_trees = load_txt_trees()

    # merge: prefer JSON data for shared tree_ids
    all_trees: Dict[str, dict] = {}
    all_trees.update(txt_trees)
    all_trees.update(json_trees)
    print(f"Total unique trees: {len(all_trees)}  (JSON={len(json_trees)}, TXT-only={len(all_trees)-len(json_trees)})")

    method_names = list(METHOD_GROUPS.keys())

    # ── per-tree predictions ──────────────────────────────────
    rows = []
    for tree_id, info in sorted(all_trees.items()):
        dets = info["dets"]
        row = {
            "tree_id": tree_id,
            "source": info["source"],
            "split": info["split"],
            "n_dets": len(dets),
            "n_sides": len(set(d["side_index"] for d in dets)),
        }
        for mname, mfunc in METHOD_GROUPS.items():
            try:
                pred = mfunc(dets)
            except Exception as e:
                pred = {c: -1 for c in NAMES}
                row[f"_err_{mname}"] = str(e)
            for c in NAMES:
                row[f"{mname}_{c}"] = pred.get(c, 0)
            row[f"{mname}_total"] = sum(pred.get(c, 0) for c in NAMES)
        rows.append(row)

    per_tree_df = pd.DataFrame(rows)
    per_tree_df.to_csv(OUT_DIR / "all_953_per_tree.csv", index=False)
    print(f"Per-tree CSV saved ({len(per_tree_df)} rows).")

    # ── accuracy on 228 JSON trees ────────────────────────────
    json_ids = set(json_trees.keys())
    json_df = per_tree_df[per_tree_df["tree_id"].isin(json_ids)].copy()

    acc_rows = []
    for mname in method_names:
        within1_list, mae_list, errsum_list = [], [], []
        for tree_id in json_df["tree_id"]:
            gt = json_trees[tree_id]["gt"]
            pred = {c: int(json_df.loc[json_df["tree_id"] == tree_id, f"{mname}_{c}"].iloc[0]) for c in NAMES}
            within1_list.append(_within1(pred, gt))
            mae_list.append(_mae(pred, gt))
            errsum_list.append(sum(abs(pred.get(c, 0) - gt.get(c, 0)) for c in NAMES))
        acc_rows.append({
            "method": mname,
            "acc_within1_pct": round(np.mean(within1_list) * 100, 2),
            "MAE": round(np.mean(mae_list), 4),
            "mean_total_err": round(np.mean(errsum_list), 4),
            "n_fail": int(sum(1 for x in within1_list if not x)),
        })

    acc_df = pd.DataFrame(acc_rows).sort_values("acc_within1_pct", ascending=False)
    acc_df.to_csv(OUT_DIR / "json_228_accuracy.csv", index=False)

    # ── aggregate counts (953 trees) ─────────────────────────
    agg_rows = []
    for mname in method_names:
        totals = {c: int(per_tree_df[f"{mname}_{c}"].sum()) for c in NAMES}
        grand = sum(totals.values())
        naive_grand = int(per_tree_df[[f"naive_{c}" for c in NAMES]].sum().sum())
        agg_rows.append({
            "method": mname,
            **{c: totals[c] for c in NAMES},
            "total": grand,
            "dedup_ratio_vs_naive": round(grand / naive_grand, 4) if naive_grand else None,
        })
    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv(OUT_DIR / "all_953_totals.csv", index=False)

    # ── per-method mean total per tree (953) ──────────────────
    mean_rows = []
    for mname in method_names:
        mean_rows.append({
            "method": mname,
            "mean_total_per_tree": round(per_tree_df[f"{mname}_total"].mean(), 3),
            "median_total_per_tree": round(per_tree_df[f"{mname}_total"].median(), 3),
            **{f"mean_{c}": round(per_tree_df[f"{mname}_{c}"].mean(), 3) for c in NAMES},
        })
    mean_df = pd.DataFrame(mean_rows)
    mean_df.to_csv(OUT_DIR / "all_953_mean_per_tree.csv", index=False)

    # ── print summaries ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("ACCURACY on 228 JSON trees (Acc ±1 per class per tree)")
    print("=" * 70)
    print(acc_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("AGGREGATE COUNTS — all 953 trees")
    print("=" * 70)
    print(agg_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("MEAN BUNCHES PER TREE — all 953 trees")
    print("=" * 70)
    print(mean_df.to_string(index=False))

    print(f"\nAll outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
