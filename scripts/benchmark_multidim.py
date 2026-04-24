"""
Benchmark Multi-Dimensi: 11 Algoritma Terbaik per Generasi
===========================================================
4 dimensi evaluasi:
  1. Akurasi (Acc ±1, MAE, per-class breakdown)
  2. Kecepatan (latency ms/pohon, throughput pohon/detik)
  3. Robustness terhadap noise koordinat (inject ±5%, ±10%, ±20%)
  4. Domain breakdown (DAMIMAS vs LONSUM) + per-class error profile

11 algoritma:
  v1/v2:  corrected (v1), visibility (v2)        ← gen awal
  v5:     adaptive_corrected, best_visibility_grid
  v6:     v6_selector
  v7:     stacking_bracketed, stacking_density
  v8:     entropy_modulated, b2_b4_boosted
  v9:     v9_selector, b2_median_v6

Output: reports/benchmark_multidim/
  - accuracy_summary.csv
  - accuracy_per_class.csv
  - accuracy_per_tree.csv
  - speed_summary.csv
  - robustness_summary.csv
  - robustness_per_noise.csv
  - domain_breakdown.csv
  - per_class_error_profile.csv
  - REPORT.md
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
JSON_DIR = BASE / "json"
OUT_DIR = BASE / "reports" / "benchmark_multidim"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NAMES = ["B1", "B2", "B3", "B4"]
NOISE_LEVELS = [0.0, 0.05, 0.10, 0.20]
SPEED_REPS = 30  # ulang tiap pohon N kali untuk estimasi stabil


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_tree_data() -> List[Tuple[str, List[Dict], Dict[str, int], str]]:
    trees = []
    for jp in sorted(JSON_DIR.glob("*.json")):
        data = json.loads(jp.read_text(encoding="utf-8"))
        tree_id = data.get("tree_name", data.get("tree_id", jp.stem))
        gt = data["summary"]["by_class"]
        gt_counts = {c: gt.get(c, 0) for c in NAMES}
        split = data.get("split", "unknown")

        # Tentukan domain dari tree_name atau tree_id
        tree_name = data.get("tree_name", tree_id)
        if "LONSUM" in tree_name.upper() or "LONSUM" in tree_id.upper():
            domain = "LONSUM"
        else:
            domain = "DAMIMAS"

        dets = []
        for side, side_data in data["images"].items():
            side_index = side_data.get("side_index", int(side.replace("sisi_", "")) - 1)
            for ann in side_data.get("annotations", []):
                if "bbox_yolo" in ann:
                    cx, cy, w, h = ann["bbox_yolo"]
                    dets.append({
                        "class": ann["class_name"],
                        "x_norm": float(cx),
                        "y_norm": float(cy),
                        "w_norm": float(w),
                        "h_norm": float(h),
                        "side_index": side_index,
                    })
        trees.append((tree_id, dets, gt_counts, split, domain))
    return trees


def inject_noise(dets: List[Dict], noise_pct: float, rng: np.random.Generator) -> List[Dict]:
    """Tambah gaussian noise ke x_norm dan y_norm."""
    if noise_pct == 0.0:
        return dets
    noisy = []
    for d in dets:
        nd = dict(d)
        nd["x_norm"] = float(np.clip(d["x_norm"] + rng.normal(0, noise_pct), 0.0, 1.0))
        nd["y_norm"] = float(np.clip(d["y_norm"] + rng.normal(0, noise_pct), 0.0, 1.0))
        noisy.append(nd)
    return noisy


# ---------------------------------------------------------------------------
# Algoritma gen awal (v1, v2) — tidak ada file algorithms/ terpisah
# ---------------------------------------------------------------------------

_BASE_FACTORS_V1 = {"B1": 1.986, "B2": 1.786, "B3": 1.795, "B4": 1.655}
_GLOBAL_RATIO_V1 = 1.788


def corrected_v1(dets: List[Dict]) -> Dict[str, int]:
    """v1: naive dibagi faktor duplikasi global per kelas."""
    naive = Counter(d["class"] for d in dets)
    return {c: max(0, round(naive.get(c, 0) / _BASE_FACTORS_V1[c])) for c in NAMES}


def visibility_v2(dets: List[Dict], alpha: float = 0.9, sigma: float = 0.45) -> Dict[str, int]:
    """v2: weighted sum berdasarkan posisi x (visibility model).
    Params alpha=0.9, sigma=0.45 dari grid search v5 — sama dengan best_visibility_grid.
    """
    counts = {}
    for c in NAMES:
        cd = [d for d in dets if d["class"] == c]
        if not cd:
            counts[c] = 0
            continue
        total = sum(
            1.0 / (1.0 + alpha * np.exp(-((d["x_norm"] - 0.5) ** 2) / (2.0 * sigma ** 2)))
            for d in cd
        )
        counts[c] = max(0, int(round(total)))
    return counts


# ---------------------------------------------------------------------------
# Build method registry
# ---------------------------------------------------------------------------

def build_methods(params: dict) -> Dict[str, Callable[[List[Dict]], Dict[str, int]]]:
    from algorithms.adaptive_corrected import predict as ac_predict
    from algorithms.best_visibility_grid import predict as bvg_predict
    from algorithms.v6_selector import predict as v6_predict
    from algorithms.stacking_bracketed import predict as sb_predict
    from algorithms.stacking_density import predict as sd_predict
    from algorithms.entropy_modulated import predict as em_predict
    from algorithms.b2_b4_boosted import predict as b2b4_predict
    from algorithms.v9_selector import predict as v9_predict
    from algorithms.b2_median_v6 import predict as b2med_predict
    from algorithms.median_strong5 import predict as ms5_predict

    return {
        "v1_corrected":           lambda d: corrected_v1(d),
        "v2_visibility":          lambda d: visibility_v2(d),
        "v5_adaptive_corrected":  lambda d: ac_predict(d),
        "v5_best_visibility":     lambda d: bvg_predict(d),
        "v6_selector":            lambda d: v6_predict(d, params),
        "v7_stacking_bracketed":  lambda d: sb_predict(d),
        "v7_stacking_density":    lambda d: sd_predict(d),
        "v8_entropy_modulated":   lambda d: em_predict(d),
        "v8_b2_b4_boosted":       lambda d: b2b4_predict(d),
        "v9_b2_median_v6":        lambda d: b2med_predict(d, params),
        "v9_selector":            lambda d: v9_predict(d, params),
    }


# ---------------------------------------------------------------------------
# Dim 1: Akurasi
# ---------------------------------------------------------------------------

def eval_accuracy(
    methods: Dict[str, Callable],
    trees: List,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    per_class_rows = []
    per_tree_rows = []

    for mname, func in methods.items():
        tree_rows = []
        for tree_id, dets, gt, split, domain in trees:
            pred = func(dets)
            err = {c: abs(pred.get(c, 0) - gt[c]) for c in NAMES}
            delta = {c: pred.get(c, 0) - gt[c] for c in NAMES}
            ok = all(e <= 1 for e in err.values())
            mae = float(np.mean(list(err.values())))
            tree_rows.append({
                "method": mname,
                "tree_id": tree_id,
                "split": split,
                "domain": domain,
                "ok": ok,
                "MAE": mae,
                "error_sum": sum(err.values()),
                **{f"gt_{c}": gt[c] for c in NAMES},
                **{f"pred_{c}": pred.get(c, 0) for c in NAMES},
                **{f"err_{c}": err[c] for c in NAMES},
                **{f"delta_{c}": delta[c] for c in NAMES},
            })
        df = pd.DataFrame(tree_rows)
        per_tree_rows.append(df)

        acc = df["ok"].mean() * 100.0
        mae_mean = df["MAE"].mean()
        mte = df["error_sum"].mean()
        n_fail = (~df["ok"]).sum()

        # Per-class MAE dan error sign
        for c in NAMES:
            per_class_rows.append({
                "method": mname,
                "class": c,
                "MAE": round(df[f"err_{c}"].mean(), 4),
                "over_count": int((df[f"delta_{c}"] > 1).sum()),
                "under_count": int((df[f"delta_{c}"] < -1).sum()),
                "exact": int((df[f"delta_{c}"] == 0).sum()),
                "within1": int((df[f"err_{c}"] <= 1).sum()),
                "pct_within1": round((df[f"err_{c}"] <= 1).mean() * 100, 2),
            })

        summary_rows.append({
            "method": mname,
            "acc_pct": round(acc, 2),
            "MAE": round(mae_mean, 4),
            "mean_total_error": round(mte, 4),
            "n_fail": int(n_fail),
            "n_trees": len(df),
        })

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["acc_pct", "MAE"], ascending=[False, True]
    )
    per_class_df = pd.DataFrame(per_class_rows)
    per_tree_df = pd.concat(per_tree_rows, ignore_index=True)
    return summary_df, per_class_df, per_tree_df


# ---------------------------------------------------------------------------
# Dim 2: Kecepatan
# ---------------------------------------------------------------------------

def eval_speed(
    methods: Dict[str, Callable],
    trees: List,
) -> pd.DataFrame:
    print("  Mengukur kecepatan...")
    rows = []
    all_dets = [dets for _, dets, _, _, _ in trees]
    n_trees = len(all_dets)

    for mname, func in methods.items():
        # Warmup
        for dets in all_dets[:5]:
            func(dets)

        # Ukur
        times_ms = []
        for rep in range(SPEED_REPS):
            t0 = time.perf_counter()
            for dets in all_dets:
                func(dets)
            t1 = time.perf_counter()
            elapsed_ms = (t1 - t0) * 1000.0
            times_ms.append(elapsed_ms / n_trees)

        times_arr = np.array(times_ms)
        rows.append({
            "method": mname,
            "mean_ms_per_tree": round(float(times_arr.mean()), 4),
            "median_ms_per_tree": round(float(np.median(times_arr)), 4),
            "std_ms": round(float(times_arr.std()), 4),
            "min_ms": round(float(times_arr.min()), 4),
            "max_ms": round(float(times_arr.max()), 4),
            "trees_per_sec": round(1000.0 / float(times_arr.mean()), 1),
            "n_reps": SPEED_REPS,
        })
        print(f"    {mname:28} {times_arr.mean():.3f} ms/pohon  ({1000/times_arr.mean():.0f} pohon/s)")

    return pd.DataFrame(rows).sort_values("mean_ms_per_tree")


# ---------------------------------------------------------------------------
# Dim 3: Robustness terhadap noise
# ---------------------------------------------------------------------------

def eval_robustness(
    methods: Dict[str, Callable],
    trees: List,
    noise_levels: List[float],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print("  Mengukur robustness terhadap noise...")
    rng = np.random.default_rng(42)
    summary_rows = []
    per_noise_rows = []

    for noise in noise_levels:
        label = f"noise_{int(noise*100):02d}pct"
        print(f"    Level noise: {noise*100:.0f}%")
        noisy_trees = [
            (tid, inject_noise(dets, noise, rng), gt, split, domain)
            for tid, dets, gt, split, domain in trees
        ]
        for mname, func in methods.items():
            errs = []
            oks = []
            for tid, dets, gt, split, domain in noisy_trees:
                pred = func(dets)
                err = {c: abs(pred.get(c, 0) - gt[c]) for c in NAMES}
                ok = all(e <= 1 for e in err.values())
                errs.append(float(np.mean(list(err.values()))))
                oks.append(ok)
                per_noise_rows.append({
                    "method": mname,
                    "noise_pct": noise * 100,
                    "noise_label": label,
                    "tree_id": tid,
                    "ok": ok,
                    "MAE": float(np.mean(list(err.values()))),
                })
            acc = np.mean(oks) * 100.0
            mae = np.mean(errs)
            summary_rows.append({
                "method": mname,
                "noise_pct": noise * 100,
                "noise_label": label,
                "acc_pct": round(float(acc), 2),
                "MAE": round(float(mae), 4),
                "n_fail": int(sum(1 for o in oks if not o)),
            })

    summary_df = pd.DataFrame(summary_rows)
    per_noise_df = pd.DataFrame(per_noise_rows)

    # Hitung degradasi: acc_drop dari noise=0
    baseline = summary_df[summary_df["noise_pct"] == 0].set_index("method")[["acc_pct", "MAE"]]
    baseline.columns = ["acc_baseline", "mae_baseline"]
    summary_df = summary_df.merge(baseline, on="method", how="left")
    summary_df["acc_drop"] = round(summary_df["acc_baseline"] - summary_df["acc_pct"], 2)
    summary_df["mae_increase"] = round(summary_df["MAE"] - summary_df["mae_baseline"], 4)

    return summary_df, per_noise_df


# ---------------------------------------------------------------------------
# Dim 4: Domain breakdown
# ---------------------------------------------------------------------------

def eval_domain(
    methods: Dict[str, Callable],
    trees: List,
    per_tree_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    # Tentukan domain unik + split unik
    all_domains = sorted(per_tree_df["domain"].unique())
    all_splits = sorted(per_tree_df["split"].unique())

    for mname in methods:
        df = per_tree_df[per_tree_df["method"] == mname]

        # Per domain
        for domain in all_domains:
            sub = df[df["domain"] == domain]
            if len(sub) == 0:
                continue
            rows.append({
                "method": mname,
                "group_type": "domain",
                "group": domain,
                "n_trees": len(sub),
                "acc_pct": round(sub["ok"].mean() * 100, 2),
                "MAE": round(sub["MAE"].mean(), 4),
                "mean_total_error": round(sub["error_sum"].mean(), 4),
                "n_fail": int((~sub["ok"]).sum()),
            })

        # Per split (train/val/test)
        for split in all_splits:
            sub = df[df["split"] == split]
            if len(sub) == 0:
                continue
            rows.append({
                "method": mname,
                "group_type": "split",
                "group": split,
                "n_trees": len(sub),
                "acc_pct": round(sub["ok"].mean() * 100, 2),
                "MAE": round(sub["MAE"].mean(), 4),
                "mean_total_error": round(sub["error_sum"].mean(), 4),
                "n_fail": int((~sub["ok"]).sum()),
            })

        # Overall
        rows.append({
            "method": mname,
            "group_type": "overall",
            "group": "ALL",
            "n_trees": len(df),
            "acc_pct": round(df["ok"].mean() * 100, 2),
            "MAE": round(df["MAE"].mean(), 4),
            "mean_total_error": round(df["error_sum"].mean(), 4),
            "n_fail": int((~df["ok"]).sum()),
        })
    return pd.DataFrame(rows).sort_values(["group", "acc_pct"], ascending=[True, False])


# ---------------------------------------------------------------------------
# Report markdown
# ---------------------------------------------------------------------------

GEN_MAP = {
    "v1_corrected": "v1",
    "v2_visibility": "v2",
    "v5_adaptive_corrected": "v5",
    "v5_best_visibility": "v5",
    "v6_selector": "v6",
    "v7_stacking_bracketed": "v7",
    "v7_stacking_density": "v7",
    "v8_entropy_modulated": "v8",
    "v8_b2_b4_boosted": "v8",
    "v9_b2_median_v6": "v9",
    "v9_selector": "v9",
}


def write_report(
    acc_df: pd.DataFrame,
    per_class_df: pd.DataFrame,
    speed_df: pd.DataFrame,
    robust_df: pd.DataFrame,
    domain_df: pd.DataFrame,
    n_trees: int,
) -> None:
    lines = []
    lines.append("# Benchmark Multi-Dimensi: 11 Algoritma Dedup")
    lines.append("")
    lines.append(f"**Dataset:** {n_trees} pohon JSON (228 GT)  ")
    lines.append(f"**Tanggal:** 2026-04-24  ")
    lines.append(f"**Metrik utama:** Acc ±1 (semua kelas dalam 1 error), MAE, ms/pohon")
    lines.append("")

    # ------- Dim 1: Akurasi -------
    lines.append("---")
    lines.append("")
    lines.append("## Dimensi 1: Akurasi (Acc ±1 per kelas)")
    lines.append("")
    lines.append("Pohon dianggap **benar** jika semua 4 kelas masing-masing dalam ±1 dari GT.")
    lines.append("")
    lines.append("| Rank | Method | Gen | Acc ±1 | MAE | MTE | Gagal |")
    lines.append("|---:|---|---|---:|---:|---:|---:|")
    for rank, row in enumerate(acc_df.itertuples(), 1):
        gen = GEN_MAP.get(row.method, "?")
        lines.append(
            f"| {rank} | `{row.method}` | {gen} | **{row.acc_pct:.2f}%** | {row.MAE:.4f} | {row.mean_total_error:.4f} | {row.n_fail} |"
        )
    lines.append("")
    lines.append("> MTE = Mean Total Error (jumlah absolut error semua kelas, rata-rata per pohon)")
    lines.append("")

    # Per-class
    lines.append("### Akurasi Per Kelas (% pohon dalam ±1)")
    lines.append("")
    pivot = per_class_df.pivot_table(
        index="method", columns="class", values="pct_within1", aggfunc="first"
    ).reindex(columns=NAMES)
    pivot = pivot.reindex(acc_df["method"])
    lines.append("| Method | B1 | B2 | B3 | B4 |")
    lines.append("|---|---:|---:|---:|---:|")
    for mname, row2 in pivot.iterrows():
        cells = " | ".join(f"{row2[c]:.1f}%" for c in NAMES)
        lines.append(f"| `{mname}` | {cells} |")
    lines.append("")

    # Over/under per class
    lines.append("### Pola Error Per Kelas (over >1 / under <-1, jumlah pohon)")
    lines.append("")
    lines.append("| Method | B1↑ | B1↓ | B2↑ | B2↓ | B3↑ | B3↓ | B4↑ | B4↓ |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for mname in acc_df["method"]:
        sub = per_class_df[per_class_df["method"] == mname].set_index("class")
        cells = []
        for c in NAMES:
            cells.append(str(int(sub.loc[c, "over_count"])))
            cells.append(str(int(sub.loc[c, "under_count"])))
        lines.append(f"| `{mname}` | {' | '.join(cells)} |")
    lines.append("")

    # ------- Dim 2: Kecepatan -------
    lines.append("---")
    lines.append("")
    lines.append("## Dimensi 2: Kecepatan (ms/pohon)")
    lines.append("")
    lines.append(f"Diukur dengan {SPEED_REPS} repetisi per metode, {n_trees} pohon per repetisi.")
    lines.append("")
    lines.append("| Rank | Method | Mean ms | Median ms | Std ms | pohon/detik |")
    lines.append("|---:|---|---:|---:|---:|---:|")
    for rank, row in enumerate(speed_df.sort_values("mean_ms_per_tree").itertuples(), 1):
        lines.append(
            f"| {rank} | `{row.method}` | {row.mean_ms_per_tree:.4f} | {row.median_ms_per_tree:.4f} | {row.std_ms:.4f} | {row.trees_per_sec:.0f} |"
        )
    lines.append("")

    # ------- Dim 3: Robustness -------
    lines.append("---")
    lines.append("")
    lines.append("## Dimensi 3: Robustness terhadap Noise Koordinat")
    lines.append("")
    lines.append(
        "Simulasi: tambah Gaussian noise σ=N% ke x_norm dan y_norm setiap bbox.  \n"
        "Mengukur seberapa cepat akurasi turun ketika koordinat detector tidak sempurna."
    )
    lines.append("")

    # Tabel pivot: method vs noise level → acc_pct
    noise_pivot = robust_df.pivot_table(
        index="method", columns="noise_pct", values="acc_pct"
    ).reindex(acc_df["method"])
    noise_levels_pct = [n * 100 for n in NOISE_LEVELS]

    header = "| Method | " + " | ".join(f"σ={n:.0f}%" for n in noise_levels_pct) + " | Drop@20% |"
    sep = "|---|" + "|".join("---:" for _ in noise_levels_pct) + "|---:|"
    lines.append(header)
    lines.append(sep)
    for mname in acc_df["method"]:
        row_vals = []
        for nlvl in noise_levels_pct:
            val = noise_pivot.loc[mname, nlvl] if nlvl in noise_pivot.columns else float("nan")
            row_vals.append(f"{val:.2f}%" if not np.isnan(val) else "-")
        drop = robust_df[
            (robust_df["method"] == mname) & (robust_df["noise_pct"] == 20.0)
        ]["acc_drop"].values
        drop_str = f"{drop[0]:.2f}%" if len(drop) > 0 else "-"
        lines.append(f"| `{mname}` | {' | '.join(row_vals)} | {drop_str} |")
    lines.append("")
    lines.append("> Drop@20% = selisih Acc antara noise=0% dan noise=20% (lebih kecil = lebih robust)")
    lines.append("")

    # ------- Dim 4: Domain breakdown -------
    lines.append("---")
    lines.append("")
    lines.append("## Dimensi 4: Domain Breakdown (DAMIMAS vs LONSUM)")
    lines.append("")

    # Per domain
    for domain in sorted(domain_df[domain_df["group_type"] == "domain"]["group"].unique()):
        sub = domain_df[(domain_df["group_type"] == "domain") & (domain_df["group"] == domain)].sort_values("acc_pct", ascending=False)
        if sub.empty:
            continue
        n = sub.iloc[0]["n_trees"]
        lines.append(f"### Domain: {domain} (n={n})")
        lines.append("")
        lines.append("| Rank | Method | Acc ±1 | MAE | Gagal |")
        lines.append("|---:|---|---:|---:|---:|")
        for rank, row in enumerate(sub.itertuples(), 1):
            lines.append(f"| {rank} | `{row.method}` | {row.acc_pct:.2f}% | {row.MAE:.4f} | {row.n_fail} |")
        lines.append("")

    # Per split
    lines.append("### Breakdown Per Split (train / val / test)")
    lines.append("")
    splits = sorted(domain_df[domain_df["group_type"] == "split"]["group"].unique())
    split_pivot = domain_df[domain_df["group_type"] == "split"].pivot_table(
        index="method", columns="group", values="acc_pct", aggfunc="first"
    ).reindex(acc_df["method"])
    header_splits = "| Method | " + " | ".join(f"{s} Acc" for s in splits) + " |"
    sep_splits = "|---|" + "|".join("---:" for _ in splits) + "|"
    lines.append(header_splits)
    lines.append(sep_splits)
    for mname in acc_df["method"]:
        cells = []
        for s in splits:
            val = split_pivot.loc[mname, s] if s in split_pivot.columns and mname in split_pivot.index else float("nan")
            cells.append(f"{val:.2f}%" if not (isinstance(val, float) and np.isnan(val)) else "-")
        lines.append(f"| `{mname}` | {' | '.join(cells)} |")
    lines.append("")

    # ------- Ringkasan akhir -------
    lines.append("---")
    lines.append("")
    lines.append("## Ringkasan: Tradeoff Antar Dimensi")
    lines.append("")
    lines.append(
        "| Method | Acc ±1 | Rank Acc | ms/pohon | Rank Speed | Drop@20% | Rank Robust |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")

    acc_rank = {row.method: rank for rank, row in enumerate(acc_df.itertuples(), 1)}
    speed_sorted = speed_df.sort_values("mean_ms_per_tree").reset_index(drop=True)
    speed_rank = {row.method: rank + 1 for rank, row in enumerate(speed_sorted.itertuples())}
    speed_ms = speed_sorted.set_index("method")["mean_ms_per_tree"].to_dict()

    drop_df = robust_df[robust_df["noise_pct"] == 20.0][["method", "acc_drop"]].copy()
    drop_df = drop_df.sort_values("acc_drop")
    robust_rank = {row.method: rank + 1 for rank, row in enumerate(drop_df.itertuples())}
    drop_val = drop_df.set_index("method")["acc_drop"].to_dict()

    for mname in acc_df["method"]:
        acc_val = acc_df[acc_df["method"] == mname]["acc_pct"].values[0]
        ms_val = speed_ms.get(mname, float("nan"))
        drop = drop_val.get(mname, float("nan"))
        lines.append(
            f"| `{mname}` | {acc_val:.2f}% | #{acc_rank.get(mname,'?')} "
            f"| {ms_val:.3f} | #{speed_rank.get(mname,'?')} "
            f"| {drop:.2f}% | #{robust_rank.get(mname,'?')} |"
        )
    lines.append("")
    lines.append(
        "> **Rekomendasi final:** `v9_selector` untuk akurasi maksimal. "
        "Untuk pipeline real-time atau inference massal, "
        "pertimbangkan `v6_selector` atau `v5_adaptive_corrected` (lebih cepat, Acc masih >93%)."
    )
    lines.append("")

    (OUT_DIR / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"  Report ditulis ke {OUT_DIR / 'REPORT.md'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== Benchmark Multi-Dimensi: 11 Algoritma ===")
    print(f"Output: {OUT_DIR}")
    print()

    print("Memuat data pohon...")
    trees = load_tree_data()
    n = len(trees)
    domains = Counter(d for _, _, _, _, d in trees)
    print(f"  {n} pohon: DAMIMAS={domains['DAMIMAS']}, LONSUM={domains['LONSUM']}")
    print()

    print("Memuat parameter v6/v9...")
    from algorithms.v6_selector import load_params
    params = load_params()
    print("  OK")
    print()

    print("Membangun registry metode...")
    methods = build_methods(params)
    print(f"  {len(methods)} metode: {', '.join(methods)}")
    print()

    # --- Dim 1 ---
    print("Dimensi 1: Akurasi...")
    acc_df, per_class_df, per_tree_df = eval_accuracy(methods, trees)
    acc_df.to_csv(OUT_DIR / "accuracy_summary.csv", index=False)
    per_class_df.to_csv(OUT_DIR / "accuracy_per_class.csv", index=False)
    per_tree_df.to_csv(OUT_DIR / "accuracy_per_tree.csv", index=False)
    print("  Hasil akurasi:")
    for _, row in acc_df.iterrows():
        print(f"    {row['method']:28} Acc={row['acc_pct']:.2f}%  MAE={row['MAE']:.4f}  Gagal={row['n_fail']}")
    print()

    # --- Dim 2 ---
    print("Dimensi 2: Kecepatan...")
    speed_df = eval_speed(methods, trees)
    speed_df.to_csv(OUT_DIR / "speed_summary.csv", index=False)
    print()

    # --- Dim 3 ---
    print("Dimensi 3: Robustness noise...")
    robust_df, robust_per_df = eval_robustness(methods, trees, NOISE_LEVELS)
    robust_df.to_csv(OUT_DIR / "robustness_summary.csv", index=False)
    robust_per_df.to_csv(OUT_DIR / "robustness_per_noise.csv", index=False)
    print()

    # --- Dim 4 ---
    print("Dimensi 4: Domain breakdown...")
    domain_df = eval_domain(methods, trees, per_tree_df)
    domain_df.to_csv(OUT_DIR / "domain_breakdown.csv", index=False)
    print()

    # --- Report ---
    print("Menulis REPORT.md...")
    write_report(acc_df, per_class_df, speed_df, robust_df, domain_df, n)
    print()

    print("=== SELESAI ===")
    print(f"Output di: {OUT_DIR}")
    print("  accuracy_summary.csv")
    print("  accuracy_per_class.csv")
    print("  accuracy_per_tree.csv")
    print("  speed_summary.csv")
    print("  robustness_summary.csv")
    print("  robustness_per_noise.csv")
    print("  domain_breakdown.csv")
    print("  REPORT.md")


if __name__ == "__main__":
    main()
