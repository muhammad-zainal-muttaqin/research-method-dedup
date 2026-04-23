"""Grid search hyperparameter GeoLinker di split train+val (228 JSON).

Tune pakai split `train` (161 pohon), pilih yang maksimalkan pct_within_1 di `val` (32 pohon).
Tulis best config ke reports/geo_linker/best.json.
"""
from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dedup.geo_linker import GeoLinker, LinkerConfig, NAMES

BASE = Path(__file__).resolve().parent.parent
JSON_DIR = BASE / "json"
OUT_DIR = BASE / "reports" / "geo_linker"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_split():
    data: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for jp in sorted(JSON_DIR.glob("*.json")):
        d = json.loads(jp.read_text(encoding="utf-8"))
        split = d.get("split", "train")
        gt = {c: d["summary"]["by_class"].get(c, 0) for c in NAMES}
        data.setdefault(split, []).append((d, gt))
    return data


def score(linker: GeoLinker, items: list[tuple[dict, dict]]) -> dict:
    n = len(items)
    w1 = w2 = exact = 0
    mae_total = 0.0
    per_class_err = {c: 0.0 for c in NAMES}
    for tree, gt in items:
        pred = linker.count(tree)
        tot_gt = sum(gt.values())
        tot_pr = sum(pred.values())
        e = abs(tot_gt - tot_pr)
        mae_total += e
        if e == 0:
            exact += 1
        if e <= 1:
            w1 += 1
        if e <= 2:
            w2 += 1
        for c in NAMES:
            per_class_err[c] += abs(gt[c] - pred[c])
    return {
        "n": n,
        "pct_exact": exact / n,
        "pct_within_1": w1 / n,
        "pct_within_2": w2 / n,
        "mae_total": mae_total / n,
        "per_class_mae": {c: per_class_err[c] / n for c in NAMES},
    }


def main():
    data = load_split()
    train, val, test = data["train"], data["val"], data["test"]
    print(f"train={len(train)} val={len(val)} test={len(test)}")
    # JSON `val` split only has 1 tree; use full train (196) untuk tune, test (31) untuk final.
    tune_set = train + val
    print(f"Tuning on {len(tune_set)} trees (train+val), hold-out test={len(test)}")

    # Grid
    grid = {
        "T_y":       [0.05, 0.08, 0.10, 0.12, 0.15, 0.18],
        "T_s":       [2.0, 3.0, 5.0],
        "lam_s":     [0.0, 0.1, 0.3],
        "lam_adj":   [0.0, 0.03, 0.08],
        "mutual_best": [False, True],
        "adjacent_only": [False],
        "T_y_opp":   [None, 0.03],
        "iou_intra": [0.5],
        # B3 lebih rapat → threshold lebih ketat
        "b3_T_y":    [None, 0.03, 0.05, 0.07],
    }
    keys = list(grid.keys())
    combos = list(product(*[grid[k] for k in keys]))
    print(f"Evaluating {len(combos)} hyperparameter combos on VAL...")

    import numpy as np
    results = []
    best = None
    for idx, values in enumerate(combos):
        kwargs = dict(zip(keys, values))
        b3_T_y = kwargs.pop("b3_T_y")
        # T_cost derived: worst valid combo of T_y + λ_s·|log(T_s)| + λ_adj
        kwargs["T_cost"] = kwargs["T_y"] + kwargs["lam_s"] * abs(
            np.log(kwargs["T_s"])
        ) + kwargs["lam_adj"] + 1e-6
        kwargs["per_class_T_y"] = {"B3": b3_T_y} if b3_T_y is not None else {}
        cfg = LinkerConfig(**kwargs)
        linker = GeoLinker(cfg)
        s = score(linker, tune_set)
        row = {**cfg.as_dict(), **{k: v for k, v in s.items() if k != "per_class_mae"},
               **{f"mae_{c}": s["per_class_mae"][c] for c in NAMES}}
        results.append(row)
        # Composite score: within_1 utama, within_2 tie-break, MAE terendah prefer simpler (penalize non-default params)
        n_custom = int(bool(b3_T_y)) + int(kwargs.get("mutual_best", False)) + int(bool(kwargs.get("T_y_opp")))
        # Round ke 2 digit supaya delta 1 pohon (0.005) tidak melenyapkan preferensi robustness
        composite = (round(s["pct_within_1"], 2),
                     round(s["pct_within_2"], 2),
                     -round(s["mae_total"], 2),
                     -n_custom)
        if best is None or composite > (
            round(best["pct_within_1"], 2),
            round(best["pct_within_2"], 2),
            -round(best["mae_total"], 2),
            -best.get("_n_custom", 99)
        ):
            s["_n_custom"] = n_custom
            best = {**s, **cfg.as_dict()}
            print(
                f"  [{idx+1}/{len(combos)}] T_y={kwargs['T_y']:.2f} T_s={kwargs['T_s']:.1f} "
                f"lam_s={kwargs['lam_s']:.2f} lam_adj={kwargs['lam_adj']:.2f} "
                f"| tune within1={s['pct_within_1']:.3f} within2={s['pct_within_2']:.3f} "
                f"mae={s['mae_total']:.3f}"
            )

    best_cfg = LinkerConfig(**{k: best[k] for k in LinkerConfig().as_dict() if k in best})
    test_score = score(GeoLinker(best_cfg), test)

    out = {
        "best_config": best_cfg.as_dict(),
        "tune_metrics":  {k: best[k] for k in ("n", "pct_exact", "pct_within_1",
                                               "pct_within_2", "mae_total")},
        "tune_per_class_mae": {c: best["per_class_mae"][c] for c in NAMES},
        "test_metrics":  {k: test_score[k] for k in ("n", "pct_exact", "pct_within_1",
                                                      "pct_within_2", "mae_total")},
        "test_per_class_mae": test_score["per_class_mae"],
    }
    (OUT_DIR / "best.json").write_text(json.dumps(out, indent=2))

    # Also dump full grid as CSV
    import csv
    if results:
        with open(OUT_DIR / "grid.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader(); w.writerows(results)

    print("\n=== BEST ===")
    print(json.dumps(out, indent=2))
    print(f"\nSaved: {OUT_DIR/'best.json'}  &  {OUT_DIR/'grid.csv'}")


if __name__ == "__main__":
    main()
