"""
Error Pattern Analysis for json_30 April 2026 (727 files)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import json
from collections import Counter

NAMES = ['B1', 'B2', 'B3', 'B4']
from algorithms.v9_selector import predict as v9_predict, load_params as v9_load_params

v9_params = v9_load_params()

error_patterns = Counter()
b2b3_errors = 0
b1_errors = 0
b4_errors = 0
b1b4_errors = 0
total_err_files = 0
total_files = 0

files = list(Path('json_30 April 2026').glob('*.json'))

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
        
        total_files += 1
        pred = v9_predict(detections, v9_params)
        
        errs = {c: abs(pred.get(c, 0) - gt_counts.get(c, 0)) for c in NAMES}
        
        if any(e > 1 for e in errs.values()):
            total_err_files += 1
            pattern = f"B1e{errs['B1']}_B2e{errs['B2']}_B3e{errs['B3']}_B4e{errs['B4']}"
            error_patterns[pattern] += 1
            
            # Categorize
            b23 = errs['B2'] + errs['B3']
            b14 = errs['B1'] + errs['B4']
            
            if b23 > b14 and b23 >= 2:
                b2b3_errors += 1
            elif b14 > b23 and b14 >= 2:
                b1b4_errors += 1
            elif errs['B1'] >= 2:
                b1_errors += 1
            elif errs['B4'] >= 2:
                b4_errors += 1
            
    except Exception as e:
        pass

print('='*60)
print('ERROR PATTERN ANALYSIS - json_30 April 2026 (727 files)')
print('='*60)
print(f'Total files: {total_files}')
print(f'Files with errors: {total_err_files} ({total_err_files/total_files*100:.1f}%)')
print(f'Accuracy: {(total_files-total_err_files)/total_files*100:.1f}%')
print()
print(f'Error breakdown:')
print(f'  B2/B3 dominant: {b2b3_errors} files ({b2b3_errors/max(total_err_files,1)*100:.1f}%)')
print(f'  B1/B4 dominant: {b1b4_errors} files ({b1b4_errors/max(total_err_files,1)*100:.1f}%)')
print(f'  B1-only: {b1_errors} files')
print(f'  B4-only: {b4_errors} files')
print()
print(f'Top 15 error signatures:')
for p, c in error_patterns.most_common(15):
    print(f'  {p}: {c} files')
print('='*60)
