# Clustering/Tracking Milestone Assertions (>95% acc on 228 JSON trees)

## Context
- **Dataset**: 228 trees with GT JSON bunch linking (subset of 953 total).
- **Input**: Raw detections (class, y_norm, area) from 4 views per tree (GT labels).
- **Goal**: Estimate per-class unique bunch counts matching `summary.by_class` (no linking used).
- **Research done**: `scripts/dedup_research.py` ran variants:
  | Method | acc_within_1 (%) | MAE (per class) | Total Error (sum errs/tree) |
  |--------|------------------|-----------------|-----------------------------|
  | corrected_naive | **90.79** | **0.2851** | **1.14** |
  | naive | 2.63 | 2.1294 | 8.52 |
  | feature_cluster | ~13 | ~1.45 | ~5.8 |
  | y_bin | ~6.6 | ~1.73 | ~6.9 |
- **Logs**: [method_comparison.csv](method_comparison.csv), [best_method_details.csv](best_method_details.csv), [summary.md](summary.md)
- **Current best**: `corrected_naive` (GT naive_sum / per-class_overcount_factor, e.g. B1:1.986x).
- **Advanced research**: ByteTrack (Kalman+IoU all-dets), Kalman+Hungarian (global assign), ReID feats (aspect/conf/embeddings).

## Milestone Criteria (VAL-CLUST-*)
All must pass on 228 JSON trees (primarily val split) for >95% acc claim:

1. **VAL-CLUST-1**: >=95% trees have *all per-class* |pred_c - gt_c| <=1.  
   *Current: 90.79% (corrected)* → FAIL, target improvement +4.2%.

2. **VAL-CLUST-2**: Mean per-class MAE <=0.25 across 4 classes.  
   *Current: 0.2851* → FAIL (close).

3. **VAL-CLUST-3**: >=95% trees have |sum_pred_classes - sum_gt_classes| <=1 (total count within1).  
   *TBD: Compute from best_details.csv (likely >90%, errors often compensate)*.

4. **VAL-CLUST-4**: Mean total_error (sum per-class errs/tree) <=1.0 w/ 95% CI excluding current 1.14.  
   *Current: 1.14* → FAIL.

5. **VAL-CLUST-5**: Per-domain (DAMIMAS/LONSUM) + per-class breakdown passes above on val/test.

## Within-1 Trees Plot Recommendation
```
python -c "
import pandas as pd; import matplotlib.pyplot as plt; import seaborn as sns
df = pd.read_csv('reports/dedup_research/best_method_details.csv')
df['total_err'] = abs(df.total_pred - df.total_gt)
fig, ax = plt.subplots(1,2,figsize=(12,5))
df.within_1.value_counts(normalize=True).plot(kind='bar', ax=ax[0]); ax[0].set_title('% Trees Within1 Per-Class')
df.total_err.value_counts(sort=False).plot(kind='bar', ax=ax[1]); ax[1].set_title('Total Count Error Dist.')
plt.savefig('reports/dedup_research/contract-work/within1_plot.png')
"
```
Expected: Histogram bars for err=0,1 dominant (>95% target cumulative err<=1).

## Next: Advanced Variants to Hit >95%
- **ByteTrack-like**: Kalman predict pos across views, associate hi/lo-conf dets.
- **Kalman+Hung**: Global Hungarian on Mahalanobis dist (pos/conf).
- **Feat ReID**: CNN embed crops + aspect/conf gating.
- Implement in `MultiViewAggregator` (RESEARCH.md §23), re-run variants.

Milestone verified when all VAL-CLUST-* pass + plot confirms within1 trees >=95%.
