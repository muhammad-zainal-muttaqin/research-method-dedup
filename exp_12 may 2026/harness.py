"""
Experiment harness for 953-tree dedup research (iter 12 Mei 2026).

Goal: push Acc±1 ≥ 90%. Hard constraint: deterministic, parameter-free, no
training, no learned thresholds chosen on val/test split.

This module owns:
  - dataset loader (Brand-New-Dataset-YOLO/json/, full 953)
  - metric computation (mandatory CLAUDE.md set + failure cataloguing)
  - thin driver that runs registered methods and writes per-tree predictions
    + accuracy CSV + per-split breakdown

It re-uses estimators in algorithms/ as the candidate pool. New methods live
beside this file as `methods_*.py` and register themselves via REGISTRY.

NOTE on honesty: any parameter introduced here must be derived from a fixed
external snapshot (e.g. the 228-tree development set captured in
reports/dedup_research_v5/ — already used by M17/M01) — never tuned on
the 953-tree benchmark itself.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
JSON_DIR = BASE / "Brand-New-Dataset-YOLO" / "json"
OUT_DIR = Path(__file__).resolve().parent / "out"
OUT_DIR.mkdir(exist_ok=True)

NAMES: Tuple[str, ...] = ("B1", "B2", "B3", "B4")

PredictFn = Callable[[List[dict]], Dict[str, int]]


# ───────────────────────── data ─────────────────────────


@dataclass(frozen=True)
class Tree:
    tree_id: str
    split: str
    dets: List[dict]
    gt: Dict[str, int]


def _parse_det(ann: dict, side: str, side_index: int) -> dict:
    cx, cy, w, h = ann["bbox_yolo"]
    return {
        "class": ann["class_name"],
        "x_norm": float(cx),
        "y_norm": float(cy),
        "w_norm": float(w),
        "h_norm": float(h),
        "area_norm": float(w) * float(h),
        "aspect_ratio": (float(w) / float(h)) if h > 0 else 1.0,
        "side": side,
        "side_index": int(side_index),
    }


def load_trees() -> List[Tree]:
    trees: List[Tree] = []
    for jp in sorted(JSON_DIR.glob("*.json")):
        data = json.loads(jp.read_text(encoding="utf-8"))
        tree_id = data.get("tree_name", data.get("tree_id", jp.stem))
        gt = {c: int(data["summary"]["by_class"].get(c, 0)) for c in NAMES}
        dets: List[dict] = []
        for side, sd in data["images"].items():
            si = sd.get("side_index", int(side.replace("sisi_", "")) - 1)
            for ann in sd.get("annotations", []):
                if "bbox_yolo" in ann:
                    dets.append(_parse_det(ann, side, si))
        trees.append(Tree(tree_id=tree_id, split=data.get("split", "unknown"), dets=dets, gt=gt))
    return trees


# ─────────────────────── metrics ───────────────────────


def _within1(pred: Dict[str, int], gt: Dict[str, int]) -> bool:
    return all(abs(pred.get(c, 0) - gt.get(c, 0)) <= 1 for c in NAMES)


def _exact(pred: Dict[str, int], gt: Dict[str, int]) -> bool:
    return all(pred.get(c, 0) == gt.get(c, 0) for c in NAMES)


def _mae(pred: Dict[str, int], gt: Dict[str, int]) -> float:
    return float(np.mean([abs(pred.get(c, 0) - gt.get(c, 0)) for c in NAMES]))


def evaluate(method: str, predictions: Dict[str, Dict[str, int]], trees: List[Tree]) -> dict:
    """Compute the mandatory metric set for one method over the full 953-tree set."""
    rows = []
    for t in trees:
        p = predictions[t.tree_id]
        row = {
            "tree_id": t.tree_id,
            "split": t.split,
            "within1": _within1(p, t.gt),
            "exact_profile": _exact(p, t.gt),
            "total_pred": sum(p.get(c, 0) for c in NAMES),
            "total_gt": sum(t.gt.get(c, 0) for c in NAMES),
        }
        for c in NAMES:
            row[f"abs_err_{c}"] = abs(p.get(c, 0) - t.gt.get(c, 0))
            row[f"err_{c}"] = p.get(c, 0) - t.gt.get(c, 0)
        rows.append(row)
    df = pd.DataFrame(rows)
    per_class_mae = {f"MAE_{c}": float(df[f"abs_err_{c}"].mean()) for c in NAMES}
    per_class_bias = {f"bias_{c}": float(df[f"err_{c}"].mean()) for c in NAMES}
    total_mae = float((df["total_pred"] - df["total_gt"]).abs().mean())
    total_within1 = float((df["total_pred"] - df["total_gt"]).abs().le(1).mean()) * 100.0
    macro = float(np.mean(list(per_class_mae.values())))
    summary = {
        "method": method,
        "acc_within1_pct": float(df["within1"].mean() * 100.0),
        "exact_profile_acc_pct": float(df["exact_profile"].mean() * 100.0),
        "macro_class_MAE": macro,
        "total_count_MAE": total_mae,
        "total_count_within1_pct": total_within1,
        "n_fail": int((~df["within1"]).sum()),
        **per_class_mae,
        **per_class_bias,
    }
    return {"summary": summary, "per_tree": df}


def split_breakdown(per_tree_df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for split, sub in per_tree_df.groupby("split"):
        out.append({
            "split": split,
            "n": len(sub),
            "acc_within1_pct": float(sub["within1"].mean() * 100.0),
            "exact_profile_pct": float(sub["exact_profile"].mean() * 100.0),
            "macro_MAE": float(np.mean([sub[f"abs_err_{c}"].mean() for c in NAMES])),
        })
    return pd.DataFrame(out)


# ─────────────────────── driver ───────────────────────


def run(methods: Dict[str, PredictFn], trees: Iterable[Tree], tag: str) -> pd.DataFrame:
    """Run methods, write per_tree.csv + accuracy.csv + split.csv, return summary df."""
    trees = list(trees)
    summaries = []
    per_tree_master: Dict[str, dict] = {t.tree_id: {"tree_id": t.tree_id, "split": t.split} for t in trees}
    for name, fn in methods.items():
        preds: Dict[str, Dict[str, int]] = {}
        for t in trees:
            preds[t.tree_id] = fn(t.dets)
        result = evaluate(name, preds, trees)
        summaries.append(result["summary"])
        # split breakdown per method
        split_df = split_breakdown(result["per_tree"])
        split_df.insert(0, "method", name)
        split_csv = OUT_DIR / f"split_{tag}.csv"
        split_df.to_csv(split_csv, mode="a", header=not split_csv.exists(), index=False)
        # store per-tree preds
        for t in trees:
            for c in NAMES:
                per_tree_master[t.tree_id][f"{name}_{c}"] = preds[t.tree_id].get(c, 0)
            per_tree_master[t.tree_id][f"{name}_within1"] = bool(_within1(preds[t.tree_id], t.gt))

    per_tree_df = pd.DataFrame(per_tree_master.values())
    per_tree_df.to_csv(OUT_DIR / f"per_tree_{tag}.csv", index=False)

    summary_df = pd.DataFrame(summaries).sort_values("acc_within1_pct", ascending=False)
    summary_df.to_csv(OUT_DIR / f"accuracy_{tag}.csv", index=False)
    return summary_df


# ─────────────────────── helpers exposed to methods ───────────────────────


def naive_count(dets: List[dict]) -> Dict[str, int]:
    c = Counter(d["class"] for d in dets)
    return {cl: int(c.get(cl, 0)) for cl in NAMES}


def max_per_side(dets: List[dict], cl: str) -> int:
    cd = [d for d in dets if d["class"] == cl]
    return int(max(Counter(d["side_index"] for d in cd).values())) if cd else 0


def active_sides(dets: List[dict], cl: str) -> int:
    return len({d["side_index"] for d in dets if d["class"] == cl})


def n_sides(dets: List[dict]) -> int:
    return len({d["side_index"] for d in dets}) if dets else 0
