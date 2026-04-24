"""Generate per-method breakdown reports with full data traceability.

Reads canonical benchmark CSVs and emits, for every algorithm:
    reports/methods/<method>.md             — primary metrics with source citations
    reports/methods/<method>_per_tree.csv   — per-tree slice (filtered)

Every number in the README points to one of these files so readers can trace
back to the raw data without hunting through CSVs.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "reports" / "benchmark_multidim"
OUT = REPO / "reports" / "methods"

ACC_TREE = BENCH / "accuracy_per_tree.csv"
ACC_CLASS = BENCH / "accuracy_per_class.csv"
ACC_SUMMARY = BENCH / "accuracy_summary.csv"
SPEED = BENCH / "speed_summary.csv"
ROBUST = BENCH / "robustness_summary.csv"
SPLIT = BENCH / "domain_breakdown.csv"

ALGO_FILE = {
    "v9_selector": "algorithms/v9_selector.py",
    "v9_b2_median_v6": "algorithms/b2_median_v6.py",
    "v6_selector": "algorithms/v6_selector.py",
    "v8_b2_b4_boosted": "algorithms/b2_b4_boosted.py",
    "v7_stacking_bracketed": "algorithms/stacking_bracketed.py",
    "v2_visibility": None,
    "v5_best_visibility": "algorithms/best_visibility_grid.py",
    "v7_stacking_density": "algorithms/stacking_density.py",
    "v8_entropy_modulated": "algorithms/entropy_modulated.py",
    "v5_adaptive_corrected": "algorithms/adaptive_corrected.py",
    "v1_corrected": None,
}


def fmt_signed(x: float) -> str:
    if abs(x) < 1e-9:
        return "0.000"
    return f"{x:+.3f}"


def load() -> dict[str, pd.DataFrame]:
    return {
        "tree": pd.read_csv(ACC_TREE),
        "cls": pd.read_csv(ACC_CLASS),
        "summary": pd.read_csv(ACC_SUMMARY),
        "speed": pd.read_csv(SPEED) if SPEED.exists() else None,
        "robust": pd.read_csv(ROBUST) if ROBUST.exists() else None,
    }


def build_method_report(method: str, data: dict[str, pd.DataFrame]) -> str:
    tree = data["tree"]
    cls = data["cls"]
    summary = data["summary"]
    speed = data["speed"]
    robust = data["robust"]

    t = tree[tree["method"] == method].copy()
    if t.empty:
        raise SystemExit(f"No per-tree rows for {method}")
    n = len(t)

    gt_tot = t["gt_B1"] + t["gt_B2"] + t["gt_B3"] + t["gt_B4"]
    pr_tot = t["pred_B1"] + t["pred_B2"] + t["pred_B3"] + t["pred_B4"]
    tot_err = (pr_tot - gt_tot).abs()
    total_mae = tot_err.mean()
    total_within1 = (tot_err <= 1).mean() * 100

    exact_mask = (t["err_B1"] == 0) & (t["err_B2"] == 0) & (t["err_B3"] == 0) & (t["err_B4"] == 0)
    exact_count = int(exact_mask.sum())
    exact_pct = exact_count / n * 100

    per_class_mae = {c: t[f"err_{c}"].abs().mean() for c in ("B1", "B2", "B3", "B4")}
    macro_mae = sum(per_class_mae.values()) / 4

    per_class_bias = {c: (t[f"pred_{c}"] - t[f"gt_{c}"]).mean() for c in ("B1", "B2", "B3", "B4")}

    fail_mask = ~t["ok"].astype(bool)
    failed = t[fail_mask][["tree_id", "split", "domain", "MAE", "err_B1", "err_B2", "err_B3", "err_B4"]]

    sum_row = summary[summary["method"] == method].iloc[0]
    acc_pct = float(sum_row["acc_pct"])
    n_fail = int(sum_row["n_fail"])

    impl = ALGO_FILE.get(method)
    impl_link = f"[`{impl}`](../../{impl})" if impl else "_(tidak punya file algoritma terpisah — lihat scripts/dedup_research_v*.py)_"

    lines: list[str] = []
    lines.append(f"# `{method}` — Primary Metrics Breakdown")
    lines.append("")
    lines.append(f"**Implementasi:** {impl_link}  ")
    lines.append(f"**Dataset:** 228 pohon JSON ({n} baris cocok dengan `method={method}`)  ")
    lines.append(f"**Raw data lengkap:** [`../benchmark_multidim/accuracy_per_tree.csv`](../benchmark_multidim/accuracy_per_tree.csv)  ")
    lines.append(f"**Per-method slice (filter sudah diterapkan):** [`{method}_per_tree.csv`]({method}_per_tree.csv)  ")
    lines.append(f"**Summary CSV:** [`../benchmark_multidim/accuracy_summary.csv`](../benchmark_multidim/accuracy_summary.csv)")
    lines.append("")
    lines.append("Seluruh angka di bawah dihitung ulang dari `accuracy_per_tree.csv` oleh `scripts/generate_method_reports.py`.")
    lines.append("")
    lines.append("## Primary Metrics")
    lines.append("")
    lines.append("| Metric | Value | Derivation |")
    lines.append("|---|---:|---|")
    lines.append(f"| Macro class-MAE | **{macro_mae:.4f}** | mean(per-class MAE) |")
    lines.append(f"| Exact accuracy | **{exact_pct:.2f}%** | {exact_count}/{n} pohon dengan err_B* = 0 di semua kelas |")
    lines.append(f"| Total count MAE | **{total_mae:.4f}** | mean \\|Σpred − Σgt\\| per pohon |")
    lines.append(f"| Total ±1 accuracy | **{total_within1:.2f}%** | {int((tot_err<=1).sum())}/{n} pohon dengan \\|Σpred − Σgt\\| ≤ 1 |")
    lines.append(f"| Acc ±1 per kelas per pohon (pelengkap) | {acc_pct:.2f}% | {n - n_fail}/{n} pohon dengan semua err_B* dalam ±1 |")
    lines.append("")
    lines.append("## Per-Class MAE")
    lines.append("")
    lines.append("Sumber: kolom `err_B*` di `accuracy_per_tree.csv` (sudah absolute).")
    lines.append("")
    lines.append("| Class | MAE | Derivation |")
    lines.append("|---|---:|---|")
    for c in ("B1", "B2", "B3", "B4"):
        lines.append(f"| {c} | **{per_class_mae[c]:.4f}** | mean(err_{c}) across {n} pohon |")
    lines.append("")
    lines.append("Cross-check versus [`accuracy_per_class.csv`](../benchmark_multidim/accuracy_per_class.csv):")
    lines.append("")
    lines.append("| Class | MAE (csv) | over_count | under_count | exact | within1 | pct_within1 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    cross = cls[cls["method"] == method]
    for _, r in cross.iterrows():
        lines.append(
            f"| {r['class']} | {r['MAE']:.4f} | {int(r['over_count'])} | {int(r['under_count'])} | "
            f"{int(r['exact'])} | {int(r['within1'])} | {r['pct_within1']:.2f}% |"
        )
    lines.append("")
    lines.append("## Per-Class Mean Error (Bias)")
    lines.append("")
    lines.append("Sumber: `pred_B* − gt_B*` di `accuracy_per_tree.csv`. Nilai `+` = overcount, `−` = undercount, `0` = tidak bias.")
    lines.append("")
    lines.append("| Class | Mean Error | Derivation |")
    lines.append("|---|---:|---|")
    for c in ("B1", "B2", "B3", "B4"):
        lines.append(f"| {c} | **{fmt_signed(per_class_bias[c])}** | mean(pred_{c} − gt_{c}) across {n} pohon |")
    lines.append("")

    if speed is not None and (speed["method"] == method).any():
        s = speed[speed["method"] == method].iloc[0]
        lines.append("## Kecepatan (pelengkap)")
        lines.append("")
        lines.append(f"Sumber: [`speed_summary.csv`](../benchmark_multidim/speed_summary.csv) ({int(s['n_reps'])} repetisi × {n} pohon)")
        lines.append("")
        lines.append(f"- Mean: **{s['mean_ms_per_tree']:.4f} ms/pohon** ({s['trees_per_sec']:,.0f} pohon/detik)")
        lines.append(f"- Median: {s['median_ms_per_tree']:.4f} ms")
        lines.append(f"- Std: {s['std_ms']:.4f} ms")
        lines.append("")

    if robust is not None and (robust["method"] == method).any():
        rows = robust[robust["method"] == method].sort_values("noise_pct")
        lines.append("## Robustness terhadap Noise Koordinat (pelengkap)")
        lines.append("")
        lines.append(f"Sumber: [`robustness_summary.csv`](../benchmark_multidim/robustness_summary.csv)")
        lines.append("")
        lines.append("| σ (noise_pct) | Acc ±1 | MAE | n_fail | Acc drop vs σ=0 |")
        lines.append("|---:|---:|---:|---:|---:|")
        for _, r in rows.iterrows():
            lines.append(
                f"| {r['noise_pct']:.0f}% | {r['acc_pct']:.2f}% | {r['MAE']:.4f} | "
                f"{int(r['n_fail'])} | {r['acc_drop']:+.2f}% |"
            )
        lines.append("")

    lines.append(f"## Pohon yang Gagal (Acc±1 fail = {len(failed)})")
    lines.append("")
    if failed.empty:
        lines.append("_Tidak ada._")
    else:
        lines.append("| tree_id | split | domain | MAE | err_B1 | err_B2 | err_B3 | err_B4 |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|")
        for _, r in failed.iterrows():
            lines.append(
                f"| `{r['tree_id']}` | {r['split']} | {r['domain']} | {r['MAE']:.2f} | "
                f"{int(r['err_B1'])} | {int(r['err_B2'])} | {int(r['err_B3'])} | {int(r['err_B4'])} |"
            )
    lines.append("")
    lines.append("## Sample 10 Baris Per-Tree")
    lines.append("")
    lines.append("Kolom penuh tersedia di per-method CSV di atas. Preview:")
    lines.append("")
    sample_cols = ["tree_id", "split", "ok", "gt_B1", "gt_B2", "gt_B3", "gt_B4",
                   "pred_B1", "pred_B2", "pred_B3", "pred_B4"]
    lines.append("| " + " | ".join(sample_cols) + " |")
    lines.append("|" + "|".join("---" for _ in sample_cols) + "|")
    for _, r in t.head(10).iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in sample_cols) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = load()
    tree = data["tree"]
    methods = list(ALGO_FILE.keys())

    for m in methods:
        slice_df = tree[tree["method"] == m]
        if slice_df.empty:
            print(f"skip {m}: no per-tree rows")
            continue
        slice_df.to_csv(OUT / f"{m}_per_tree.csv", index=False)
        md = build_method_report(m, data)
        (OUT / f"{m}.md").write_text(md, encoding="utf-8")
        print(f"wrote {m}.md + {m}_per_tree.csv")


if __name__ == "__main__":
    main()
