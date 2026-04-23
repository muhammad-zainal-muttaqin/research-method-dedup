# Dedup Research V4 Report (Pixel-Aware Empirical Geometry)
**Date:** 2026-04-23
**Best:** visibility | Acc: 92.11% | MAE: 0.2719

## Key Improvement
- Added **HSV mean per crop** from actual images
- Learned **Mahalanobis distance** from _confirmedLinks (geometry + color)
- Used **Hungarian matching** with cylindrical priors
- Ensemble of visibility + pixel-aware matching + corrected

## Next Steps Recommendation
If still <95%, add Laplacian texture variance for spiny B4 distinction or refine cylindrical priors further.

Outputs in `reports/dedup_research_v4/`
