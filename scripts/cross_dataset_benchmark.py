"""
Cross-Dataset Benchmark: v9 vs v10
Evaluasi generalisasi algoritma pada 3 dataset:
1. json/ (22 Apr 2026) - 143 trees (original)
2. json_28 April 2026/ - 322 trees  
3. json_30 April 2026/ - 442 trees (largest, superset)

Output: Comparison report dan CSV untuk analisis.
"""

import json
import sys
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from algorithms.v9_selector import predict as v9_predict, load_params as v9_load_params
from algorithms.v10_selector import predict as v10_predict

NAMES = ["B1", "B2", "B3", "B4"]


def load_dataset(json_dir: str):
    """Load dataset dan group by tree_id."""
    json_path = Path(json_dir)
    if not json_path.exists():
        print(f"Warning: {json_dir} not found")
        return {}
    
    files = list(json_path.glob("*.json"))
    print(f"  Loading {len(files)} files from {json_dir}...")
    
    trees = defaultdict(list)
    for f in files:
        tree_id = f.stem.split("__")[0]
        trees[tree_id].append(f)
    
    return trees


def parse_tree(tree_files: list) -> tuple:
    """Parse tree files -> detections + GT counts."""
    detections = []
    gt_counts = Counter()
    
    for json_file in tree_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # GT dari summary
        if "summary" in data:
            for c in NAMES:
                gt_counts[c] += data["summary"].get("by_class", {}).get(c, 0)
        elif "bunches" in data:
            for bunch in data.get("bunches", []):
                c = bunch.get("class", "B3")
                gt_counts[c] += 1
        
        # Deteksi dari images
        images = data.get("images", {})
        for side_name, side_data in images.items():
            side_idx = int(side_name.replace("sisi_", "")) - 1 if side_name.startswith("sisi_") else 0
            for ann in side_data.get("annotations", []):
                detections.append({
                    "class": ann.get("class_name", "B3"),
                    "x_norm": ann["bbox_yolo"][0],
                    "y_norm": ann["bbox_yolo"][1],
                    "side_index": side_idx
                })
    
    return detections, dict(gt_counts)


def evaluate_tree(detections: list, gt_counts: dict, method_fn, method_name: str, params=None) -> dict:
    """Evaluate single tree dengan method."""
    if params:
        pred = method_fn(detections, params)
    else:
        pred = method_fn(detections)
    
    # Calculate metrics
    errors = [abs(pred.get(c, 0) - gt_counts.get(c, 0)) for c in NAMES]
    within_1 = all(e <= 1 for e in errors)
    total_error = sum(errors)
    
    return {
        "method": method_name,
        "pred": pred,
        "gt": gt_counts,
        "errors": {c: e for c, e in zip(NAMES, errors)},
        "within_1": within_1,
        "total_error": total_error,
        "mae": np.mean(errors)
    }


def evaluate_dataset(trees: dict, dataset_name: str, v9_params: dict) -> pd.DataFrame:
    """Evaluate seluruh dataset dengan v9 dan v10."""
    results = []
    
    print(f"\nEvaluating {dataset_name} ({len(trees)} trees)...")
    
    for tree_id, tree_files in trees.items():
        try:
            detections, gt_counts = parse_tree(tree_files)
            if not detections or sum(gt_counts.values()) == 0:
                continue
            
            # v9
            r9 = evaluate_tree(detections, gt_counts, v9_predict, "v9", v9_params)
            r9["tree_id"] = tree_id
            r9["dataset"] = dataset_name
            results.append(r9)
            
            # v10
            r10 = evaluate_tree(detections, gt_counts, v10_predict, "v10", None)
            r10["tree_id"] = tree_id
            r10["dataset"] = dataset_name
            results.append(r10)
            
        except Exception as e:
            print(f"  Error on {tree_id}: {e}")
    
    return pd.DataFrame(results)


def compute_summary(df: pd.DataFrame) -> dict:
    """Compute summary statistics."""
    summary = {}
    
    for method in ["v9", "v10"]:
        mdf = df[df["method"] == method]
        if len(mdf) == 0:
            continue
        
        # Acc within ±1
        acc = mdf["within_1"].mean() * 100
        
        # MAE per class
        mae_per_class = {}
        for c in NAMES:
            errors = mdf["errors"].apply(lambda x: x.get(c, 0) if isinstance(x, dict) else 0)
            mae_per_class[c] = errors.mean()
        
        # Overall MAE
        all_errors = []
        for _, row in mdf.iterrows():
            if isinstance(row["errors"], dict):
                all_errors.extend(row["errors"].values())
        
        summary[method] = {
            "accuracy_within_1": acc,
            "mae_overall": np.mean(all_errors) if all_errors else 0,
            "mae_per_class": mae_per_class,
            "mean_total_error": mdf["total_error"].mean(),
            "n_trees": len(mdf)
        }
    
    return summary


def main():
    print("=" * 60)
    print("CROSS-DATASET BENCHMARK: v9 vs v10")
    print("=" * 60)
    
    # Load v9 params
    v9_params = v9_load_params()
    
    # Load datasets
    datasets = {
        "json_22Apr": "json",
        "json_28Apr": "json_28 April 2026",
        "json_30Apr": "json_30 April 2026"
    }
    
    all_results = []
    all_summaries = {}
    
    for name, path in datasets.items():
        print(f"\n{'='*50}")
        print(f"Dataset: {name}")
        print(f"{'='*50}")
        
        trees = load_dataset(path)
        if not trees:
            continue
        
        df = evaluate_dataset(trees, name, v9_params)
        all_results.append(df)
        
        # Compute summary
        summary = compute_summary(df)
        all_summaries[name] = summary
        
        # Print results
        print(f"\n  Results for {name}:")
        print(f"  {'Method':<10} {'Acc ±1':<10} {'MAE':<10} {'Mean Total Err':<15}")
        print(f"  {'-'*50}")
        for method in ["v9", "v10"]:
            if method in summary:
                s = summary[method]
                print(f"  {method:<10} {s['accuracy_within_1']:.2f}%     {s['mae_overall']:.4f}     {s['mean_total_error']:.3f}")
    
    # Combine results
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
    else:
        combined = pd.DataFrame()
    
    # Create reports directory
    report_dir = Path("reports/cross_dataset_benchmark")
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results
    combined.to_csv(report_dir / "detailed_results.csv", index=False)
    
    # Create comparison table
    print(f"\n\n{'='*60}")
    print("FINAL COMPARISON TABLE")
    print(f"{'='*60}")
    print(f"\n{'Dataset':<15} {'v9 Acc':<12} {'v10 Acc':<12} {'v9 MAE':<12} {'v10 MAE':<12} {'Delta Acc':<10}")
    print(f"{'-'*75}")
    
    comparison_rows = []
    for name in datasets.keys():
        if name in all_summaries:
            v9_acc = all_summaries[name]["v9"]["accuracy_within_1"]
            v10_acc = all_summaries[name]["v10"]["accuracy_within_1"]
            v9_mae = all_summaries[name]["v9"]["mae_overall"]
            v10_mae = all_summaries[name]["v10"]["mae_overall"]
            delta = v10_acc - v9_acc
            
            print(f"{name:<15} {v9_acc:.2f}%       {v10_acc:.2f}%       {v9_mae:.4f}       {v10_mae:.4f}       {delta:+.2f}%")
            
            comparison_rows.append({
                "dataset": name,
                "v9_acc": v9_acc,
                "v10_acc": v10_acc,
                "v9_mae": v9_mae,
                "v10_mae": v10_mae,
                "delta_acc": delta
            })
    
    # Save comparison
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(report_dir / "comparison_summary.csv", index=False)
    
    # Save detailed summary
    with open(report_dir / "summary_report.txt", 'w') as f:
        f.write("CROSS-DATASET BENCHMARK: v9 vs v10\n")
        f.write("="*60 + "\n\n")
        
        for name, summary in all_summaries.items():
            f.write(f"\nDataset: {name}\n")
            f.write("-" * 40 + "\n")
            for method, stats in summary.items():
                f.write(f"\n  {method.upper()}:\n")
                f.write(f"    Acc ±1: {stats['accuracy_within_1']:.2f}%\n")
                f.write(f"    MAE: {stats['mae_overall']:.4f}\n")
                f.write(f"    Mean Total Error: {stats['mean_total_error']:.3f}\n")
                f.write(f"    Per-class MAE:\n")
                for c, mae in stats['mae_per_class'].items():
                    f.write(f"      {c}: {mae:.4f}\n")
    
    print(f"\n\nReports saved to {report_dir}/")
    print(f"  - detailed_results.csv")
    print(f"  - comparison_summary.csv")
    print(f"  - summary_report.txt")


if __name__ == "__main__":
    main()
