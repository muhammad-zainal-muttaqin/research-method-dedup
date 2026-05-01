"""
Final Benchmark: All Methods vs All Datasets
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import json
from collections import Counter
import pandas as pd

# Import all methods
from algorithms.v9_selector import predict as v9_predict, load_params as v9_load_params
from algorithms.v10_selector import predict as v10_predict
from algorithms.v11_selector import predict as v11_predict
from algorithms.v12_selector import predict as v12_predict

NAMES = ['B1', 'B2', 'B3', 'B4']
METHODS = {
    'v9': v9_predict,
    'v10': v10_predict,
    'v11': v11_predict,
    'v12': v12_predict,
}

def evaluate_all_methods(json_dir: str, dataset_name: str):
    files = list(Path(json_dir).glob('*.json'))
    print(f'\n=== {dataset_name}: {len(files)} files ===')
    
    v9_params = v9_load_params()
    results = {m: {'correct': 0, 'total_mae': 0} for m in METHODS}
    
    for f in files:
        try:
            with open(f, 'r') as file:
                data = json.load(file)
            
            gt_counts = {}
            if 'summary' in data:
                for c in NAMES:
                    gt_counts[c] = data['summary'].get('by_class', {}).get(c, 0)
            
            detections = []
            images = data.get('images', {})
            for side_name, side_data in images.items():
                side_idx = int(side_name.replace('sisi_', '')) - 1 if side_name.startswith('sisi_') else 0
                for ann in side_data.get('annotations', []):
                    detections.append({
                        'class': ann.get('class_name', 'B3'),
                        'x_norm': ann['bbox_yolo'][0],
                        'y_norm': ann['bbox_yolo'][1],
                        'side_index': side_idx
                    })
            
            if not detections or sum(gt_counts.values()) == 0:
                continue
            
            for mname, mfn in METHODS.items():
                if mname == 'v9':
                    pred = mfn(detections, v9_params)
                else:
                    pred = mfn(detections)
                
                errors = [abs(pred.get(c, 0) - gt_counts.get(c, 0)) for c in NAMES]
                if all(e <= 1 for e in errors):
                    results[mname]['correct'] += 1
                results[mname]['total_mae'] += sum(errors) / 4
                
        except:
            pass
    
    n = len(files)
    print(f'{'Method':<10} {'Acc ±1':<10} {'MAE':<10}')
    print('-' * 30)
    for m in METHODS:
        acc = results[m]['correct'] / n * 100
        mae = results[m]['total_mae'] / n
        print(f'{m:<10} {acc:.2f}%      {mae:.4f}')
    
    return {m: results[m]['correct'] / n * 100 for m in METHODS}

# Run on all datasets
datasets = [
    ('json/ (22 Apr)', 'json'),
    ('json_28 Apr', 'json_28 April 2026'),
    ('json_30 Apr', 'json_30 April 2026'),
]

all_results = {}
for name, path in datasets:
    all_results[name] = evaluate_all_methods(path, name)

# Summary table
print('\n' + '='*60)
print('SUMMARY: All Methods Performance')
print('='*60)
print(f"{'Dataset':<20} {'v9':<8} {'v10':<8} {'v11':<8} {'v12':<8}")
print('-'*60)
for name in [d[0] for d in datasets]:
    vals = [f"{all_results[name][m]:.1f}%" for m in ['v9', 'v10', 'v11', 'v12']]
    print(f"{name:<20} {vals[0]:<8} {vals[1]:<8} {vals[2]:<8} {vals[3]:<8}")
