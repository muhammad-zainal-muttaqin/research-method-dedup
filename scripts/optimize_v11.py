"""
Hyperparameter Optimization untuk v11
Menggunakan grid search + threshold learning pada full dataset (725 files)
"""

import json
import sys
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
import itertools

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

NAMES = ["B1", "B2", "B3", "B4"]


def load_all_files(json_dir: str):
    """Load all JSON files."""
    files = list(Path(json_dir).glob("*.json"))
    data = []
    
    for f in files:
        with open(f, 'r', encoding='utf-8') as file:
            d = json.load(file)
        
        # Parse GT
        gt_counts = {}
        if "summary" in d:
            for c in NAMES:
                gt_counts[c] = d["summary"].get("by_class", {}).get(c, 0)
        
        # Parse detections
        detections = []
        images = d.get("images", {})
        for side_name, side_data in images.items():
            side_idx = int(side_name.replace("sisi_", "")) - 1 if side_name.startswith("sisi_") else 0
            for ann in side_data.get("annotations", []):
                detections.append({
                    "class": ann.get("class_name", "B3"),
                    "x_norm": ann["bbox_yolo"][0],
                    "y_norm": ann["bbox_yolo"][1],
                    "side_index": side_idx
                })
        
        if detections and sum(gt_counts.values()) > 0:
            data.append({
                'file': f.name,
                'detections': detections,
                'gt': gt_counts
            })
    
    return data


def adaptive_corrected(detections, base_factors, density_a, density_b):
    """Adaptive corrected dengan parameter yang bisa di-tune."""
    n_total = len(detections)
    scale = float(np.clip(density_a - density_b * n_total, 1.3, 2.3) / 1.8)
    naive = Counter(d["class"] for d in detections)
    return {c: max(0, round(naive.get(c, 0) / (base_factors.get(c, 1.8) * scale))) for c in NAMES}


def visibility(detections, alpha, sigma):
    """Visibility model."""
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


def class_aware_vis(detections, a14, a23, s14, s23):
    """Class-aware visibility."""
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


def evaluate_params(data, params):
    """Evaluate parameter set on dataset."""
    correct = 0
    total_mae = 0
    
    for item in data:
        det = item['detections']
        gt = item['gt']
        
        # Compute predictions
        pred_adaptive = adaptive_corrected(det, params['base_factors'], 
                                           params['density_a'], params['density_b'])
        pred_vis = visibility(det, params['vis_alpha'], params['vis_sigma'])
        pred_class = class_aware_vis(det, params['a14'], params['a23'], 
                                      params['s14'], params['s23'])
        
        # Selector logic (simplified)
        n_total = len(det)
        naive = Counter(d["class"] for d in det)
        
        # Simple selector: use class_aware for high B3 ratio, else adaptive
        b3_ratio = naive.get('B3', 0) / max(max(Counter(d['side_index'] for d in det if d['class'] == 'B3').values(), default=1), 1)
        
        if b3_ratio > params.get('b3_threshold', 3.0):
            pred = pred_class
        else:
            pred = pred_adaptive
        
        # Check accuracy
        errors = [abs(pred.get(c, 0) - gt.get(c, 0)) for c in NAMES]
        if all(e <= 1 for e in errors):
            correct += 1
        total_mae += np.mean(errors)
    
    n = len(data)
    return {
        'accuracy': correct / n * 100,
        'mae': total_mae / n
    }


def grid_search(data):
    """Grid search untuk parameter optimal."""
    print("Starting grid search for v11 parameters...")
    
    # Search space
    base_factors_options = [
        {"B1": 2.0, "B2": 1.8, "B3": 1.8, "B4": 1.65},
        {"B1": 2.0, "B2": 1.85, "B3": 1.84, "B4": 1.67},
        {"B1": 1.95, "B2": 1.82, "B3": 1.82, "B4": 1.70},
    ]
    
    density_a_range = [2.0, 2.05, 2.08, 2.10]
    density_b_range = [0.012, 0.0135, 0.014, 0.015]
    
    vis_alpha_range = [0.8, 0.85, 0.9, 0.95]
    vis_sigma_range = [0.42, 0.45, 0.48, 0.50, 0.55]
    
    a_range = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    s_range = [0.40, 0.42, 0.45, 0.50, 0.52, 0.55]
    
    b3_thresh_range = [2.5, 3.0, 3.5, 4.0]
    
    best_params = None
    best_acc = 0
    
    # Coarse search
    print("Phase 1: Coarse grid search...")
    for bf, da, db, va, vs, a14, a23, s14, s23, b3t in itertools.product(
        base_factors_options,
        density_a_range[::2],
        density_b_range[::2],
        vis_alpha_range[::2],
        vis_sigma_range[::2],
        a_range[::2],
        a_range[::2],
        s_range[::2],
        s_range[::2],
        b3_thresh_range[::2]
    ):
        params = {
            'base_factors': bf,
            'density_a': da,
            'density_b': db,
            'vis_alpha': va,
            'vis_sigma': vs,
            'a14': a14,
            'a23': a23,
            's14': s14,
            's23': s23,
            'b3_threshold': b3t
        }
        
        result = evaluate_params(data, params)
        if result['accuracy'] > best_acc:
            best_acc = result['accuracy']
            best_params = params
            print(f"  New best: {best_acc:.2f}% (MAE: {result['mae']:.4f})")
    
    print(f"\nBest coarse params: {best_acc:.2f}%")
    return best_params, best_acc


def main():
    print("Loading dataset...")
    data = load_all_files("json_30 April 2026")
    print(f"Loaded {len(data)} files")
    
    best_params, best_acc = grid_search(data)
    
    print("\n" + "="*60)
    print("BEST PARAMETERS FOUND")
    print("="*60)
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print(f"\nAccuracy: {best_acc:.2f}%")
    
    # Save params
    output = Path("reports/dedup_research_v11")
    output.mkdir(parents=True, exist_ok=True)
    
    with open(output / "v11_params.json", 'w') as f:
        json.dump(best_params, f, indent=2)
    
    print(f"\nSaved to {output}/v11_params.json")


if __name__ == "__main__":
    main()
