"""
Fit parameter v10 berdasarkan dataset besar (json_30 April 2026/ - 442 trees)
Metode: Algorithmic fitting (no backprop, deterministic)
Output: Parameter CSV untuk v10_selector
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict
import itertools

NAMES = ["B1", "B2", "B3", "B4"]


def load_json_dataset(json_dir: str):
    """Load semua JSON dan kelompokkan per tree_id."""
    json_path = Path(json_dir)
    files = list(json_path.glob("*.json"))
    print(f"Found {len(files)} JSON files in {json_dir}")
    
    # Group by tree_id
    trees = defaultdict(list)
    for f in files:
        tree_id = f.stem.split("__")[0]
        trees[tree_id].append(f)
    
    print(f"Unique trees: {len(trees)}")
    return trees


def parse_tree(tree_files: list) -> tuple:
    """
    Parse tree files dan ekstrak:
    - detections: list bbox dengan class, x_norm, y_norm, side_index
    - gt_counts: ground truth unique count per kelas
    """
    detections = []
    gt_counts = Counter()
    
    for json_file in tree_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Ambil GT dari summary
        if "summary" in data:
            summary = data["summary"]
            for c in NAMES:
                gt_counts[c] += summary.get("by_class", {}).get(c, 0)
        elif "bunches" in data:
            # Fallback ke bunches
            for bunch in data.get("bunches", []):
                c = bunch.get("class", "B3")
                gt_counts[c] += 1
        
        # Ambil deteksi dari images
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


def fit_base_factors(trees_data: list) -> dict:
    """
    Fit base factors (divisors) yang optimal per kelas.
    base_factor = mean(naive_count / gt_count) untuk setiap kelas.
    """
    factors = {c: [] for c in NAMES}
    
    for detections, gt_counts in trees_data:
        naive_counts = Counter(d["class"] for d in detections)
        for c in NAMES:
            if gt_counts.get(c, 0) > 0:
                ratio = naive_counts.get(c, 0) / gt_counts[c]
                factors[c].append(ratio)
    
    base_factors = {}
    for c in NAMES:
        if factors[c]:
            # Gunakan median untuk robustness
            base_factors[c] = float(np.median(factors[c]))
        else:
            base_factors[c] = 1.8  # default
    
    print(f"\n=== Base Factors (fitted on {len(trees_data)} trees) ===")
    for c in NAMES:
        print(f"  {c}: {base_factors[c]:.3f}")
    
    return base_factors


def fit_visibility_params(trees_data: list) -> dict:
    """
    Grid search untuk alpha dan sigma visibility model.
    """
    best_params = {}
    
    # Grid
    alphas = np.arange(0.3, 1.5, 0.1)
    sigmas = np.arange(0.25, 0.75, 0.05)
    
    best_mae = float('inf')
    best_alpha, best_sigma = 0.9, 0.45
    
    for alpha, sigma in itertools.product(alphas, sigmas):
        total_mae = 0
        n_valid = 0
        
        for detections, gt_counts in trees_data:
            pred = _visibility_predict(detections, alpha, sigma)
            for c in NAMES:
                total_mae += abs(pred.get(c, 0) - gt_counts.get(c, 0))
                n_valid += 1
        
        mae = total_mae / n_valid if n_valid > 0 else float('inf')
        if mae < best_mae:
            best_mae = mae
            best_alpha, best_sigma = alpha, sigma
    
    best_params["vis_alpha"] = round(best_alpha, 3)
    best_params["vis_sigma"] = round(best_sigma, 3)
    
    print(f"\n=== Visibility Params ===")
    print(f"  alpha: {best_params['vis_alpha']}, sigma: {best_params['vis_sigma']} (MAE: {best_mae:.4f})")
    
    return best_params


def _visibility_predict(detections: list, alpha: float, sigma: float) -> dict:
    """Predict using visibility model."""
    counts = {}
    for c in NAMES:
        cd = [d for d in detections if d["class"] == c]
        if not cd:
            counts[c] = 0
            continue
        total = sum(
            1.0 / (1.0 + alpha * np.exp(-((d["x_norm"] - 0.5) ** 2) / (2.0 * sigma ** 2)))
            for d in cd
        )
        counts[c] = max(0, int(round(total)))
    return counts


def fit_class_aware_params(trees_data: list) -> dict:
    """
    Grid search untuk class-aware visibility (B1B4 vs B2B3)."""
    alphas = np.arange(0.3, 1.5, 0.1)
    sigmas = np.arange(0.25, 0.75, 0.05)
    
    best_mae = float('inf')
    best_params = {}
    
    for a14, a23, s14, s23 in itertools.product(alphas, alphas, sigmas, sigmas):
        total_mae = 0
        n_valid = 0
        
        for detections, gt_counts in trees_data:
            pred = _class_aware_predict(detections, a14, a23, s14, s23)
            for c in NAMES:
                total_mae += abs(pred.get(c, 0) - gt_counts.get(c, 0))
                n_valid += 1
        
        mae = total_mae / n_valid if n_valid > 0 else float('inf')
        if mae < best_mae:
            best_mae = mae
            best_params = {
                "class_alpha_B1B4": round(a14, 3),
                "class_alpha_B2B3": round(a23, 3),
                "class_sigma_B1B4": round(s14, 3),
                "class_sigma_B2B3": round(s23, 3)
            }
    
    print(f"\n=== Class-Aware Params ===")
    print(f"  B1B4: alpha={best_params['class_alpha_B1B4']}, sigma={best_params['class_sigma_B1B4']}")
    print(f"  B2B3: alpha={best_params['class_alpha_B2B3']}, sigma={best_params['class_sigma_B2B3']}")
    print(f"  MAE: {best_mae:.4f}")
    
    return best_params


def _class_aware_predict(detections: list, a14: float, a23: float, s14: float, s23: float) -> dict:
    """Predict using class-aware visibility."""
    counts = {}
    for c in NAMES:
        cd = [d for d in detections if d["class"] == c]
        if not cd:
            counts[c] = 0
            continue
        alpha = a23 if c in ("B2", "B3") else a14
        sigma = s23 if c in ("B2", "B3") else s14
        total = sum(
            1.0 / (1.0 + alpha * np.exp(-((d["x_norm"] - 0.5) ** 2) / (2.0 * sigma ** 2)))
            for d in cd
        )
        counts[c] = max(0, int(round(total)))
    return counts


def fit_adaptive_scale(trees_data: list) -> tuple:
    """
    Fit adaptive density scale function parameters.
    scale = clip(a - b * n_total, min_val, max_val) / norm
    """
    # Cari parameter optimal
    candidates = []
    
    for a in np.arange(1.8, 2.5, 0.05):
        for b in np.arange(0.005, 0.025, 0.002):
            errors = []
            for detections, gt_counts in trees_data:
                n_total = len(detections)
                scale = float(np.clip(a - b * n_total, 1.3, 2.3) / 1.8)
                
                naive = Counter(d["class"] for d in detections)
                base_factors_local = {"B1": 2.0, "B2": 1.8, "B3": 1.8, "B4": 1.65}
                
                for c in NAMES:
                    pred = max(0, round(naive.get(c, 0) / (base_factors_local[c] * scale)))
                    errors.append(abs(pred - gt_counts.get(c, 0)))
            
            candidates.append((np.mean(errors), a, b))
    
    candidates.sort()
    best = candidates[0]
    
    print(f"\n=== Adaptive Scale Params ===")
    print(f"  a={best[1]:.3f}, b={best[2]:.4f}")
    print(f"  Mean MAE: {best[0]:.4f}")
    
    return best[1], best[2]


def extract_regime_examples(trees_data: list, base_factors: dict):
    """
    Extract examples untuk v10 regime learning.
    Identifikasi pola pohon yang sering gagal di metode baseline.
    """
    print(f"\n=== Regime Analysis ===")
    
    # Simulate baseline predictions
    failures = []
    
    for detections, gt_counts in trees_data:
        # Simulate adaptive_corrected
        n_total = len(detections)
        scale = float(np.clip(2.05 - 0.014 * n_total, 1.45, 2.10) / 1.79)
        naive = Counter(d["class"] for d in detections)
        pred = {c: max(0, round(naive.get(c, 0) / (base_factors.get(c, 1.8) * scale))) for c in NAMES}
        
        # Check if error > 1 for any class
        has_error = any(abs(pred[c] - gt_counts.get(c, 0)) > 1 for c in NAMES)
        
        if has_error:
            failures.append((detections, gt_counts, pred))
    
    print(f"  Trees with >1 error: {len(failures)} / {len(trees_data)}")
    
    # Analyze patterns in failures
    by_class_dist = Counter()
    total_dets_dist = Counter()
    
    for det, gt, pred in failures:
        naive = Counter(d["class"] for d in det)
        class_str = f"B1={naive.get('B1',0)},B2={naive.get('B2',0)},B3={naive.get('B3',0)},B4={naive.get('B4',0)}"
        by_class_dist[class_str] += 1
        total_dets_dist[len(det)] += 1
    
    print(f"  Common failure patterns: {by_class_dist.most_common(5)}")
    
    return failures


def main():
    # Load dataset besar
    trees = load_json_dataset("json_30 April 2026")
    
    # Parse semua tree
    trees_data = []
    for tree_id, tree_files in trees.items():
        try:
            detections, gt_counts = parse_tree(tree_files)
            if detections and sum(gt_counts.values()) > 0:
                trees_data.append((detections, gt_counts))
        except Exception as e:
            print(f"Error parsing {tree_id}: {e}")
    
    print(f"\nSuccessfully parsed: {len(trees_data)} trees")
    
    # Fit semua parameter
    base_factors = fit_base_factors(trees_data)
    vis_params = fit_visibility_params(trees_data)
    class_params = fit_class_aware_params(trees_data)
    a_scale, b_scale = fit_adaptive_scale(trees_data)
    failures = extract_regime_examples(trees_data, base_factors)
    
    # Save parameters
    params = {
        "base_factors": base_factors,
        "vis_alpha": vis_params["vis_alpha"],
        "vis_sigma": vis_params["vis_sigma"],
        **class_params,
        "adaptive_a": round(a_scale, 3),
        "adaptive_b": round(b_scale, 4),
        "n_trees_fit": len(trees_data)
    }
    
    # Save to CSV
    output_dir = Path("reports/dedup_research_v10")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame([params])
    df.to_csv(output_dir / "v10_params_fitted.csv", index=False)
    
    # Also save as JSON for readability
    with open(output_dir / "v10_params_fitted.json", 'w') as f:
        json.dump(params, f, indent=2)
    
    print(f"\n✓ Parameters saved to {output_dir}")
    print(f"\nSummary: Fitted on {len(trees_data)} trees")
    
    return params


if __name__ == "__main__":
    main()
