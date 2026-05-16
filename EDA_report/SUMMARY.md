# EDA Report - Brand-New-Dataset-YOLO

## Scope
- Source: `Brand-New-Dataset-YOLO/`
- JSON GT files analyzed from `json/*.json`
- Label detections analyzed from `labels/*.txt`
- Split metadata from `split_manifest.csv`
- Optional parquet read from `data/ground_truth.parquet`

## Global Counts
- Trees (JSON): **953**
- Unique bunches: **9,823**
- Annotation rows (YOLO-like entries in JSON images): **18,540**
- Confirmed links: **8,717**

## Side Distribution (Trees)
- 4 sides: 908 trees
- 8 sides: 45 trees

## Appearance Distribution (Unique Bunches) — per tree-type

Theoretical max appearance = `n_sides` (camera positions). Empty buckets shown explicitly.

### 4-side trees (n_bunches=9,278, theoretical_max=4)
- appearance_count=1: 2,394 (25.8%)
- appearance_count=2: 6,165 (66.4%)
- appearance_count=3: 719 (7.7%)
- appearance_count=4: 0 (0.0%)

### 8-side trees (n_bunches=545, theoretical_max=8)
- appearance_count=1: 101 (18.5%)
- appearance_count=2: 99 (18.2%)
- appearance_count=3: 115 (21.1%)
- appearance_count=4: 147 (27.0%)
- appearance_count=5: 71 (13.0%)
- appearance_count=6: 12 (2.2%)
- appearance_count=7: 0 (0.0%)
- appearance_count=8: 0 (0.0%)

## Unique Side Count Distribution — per tree-type

### 4-side trees (n_bunches=9,278, theoretical_max=4)
- unique_side_count=1: 2,394 (25.8%)
- unique_side_count=2: 6,165 (66.4%)
- unique_side_count=3: 719 (7.7%)
- unique_side_count=4: 0 (0.0%)

### 8-side trees (n_bunches=545, theoretical_max=8)
- unique_side_count=1: 101 (18.5%)
- unique_side_count=2: 99 (18.2%)
- unique_side_count=3: 115 (21.1%)
- unique_side_count=4: 147 (27.0%)
- unique_side_count=5: 71 (13.0%)
- unique_side_count=6: 12 (2.2%)
- unique_side_count=7: 0 (0.0%)
- unique_side_count=8: 0 (0.0%)

## Same-side Duplicates
- Bunches with 0 same-side duplicates: **9,823** / 9,823
- Bunches with ≥1 same-side duplicate: **0** (GT clean post-fix 2026-05-16)

## Key Anomaly Counters
- Bunches with `appearance_count > 4`:
  - 4-side trees: **N/A** (theoretical max = 4)
  - 8-side trees: **83** / 545 (15.2%)
- Bunches with `appearance_count > tree_n_sides` (impossible): **0**
- Rows in `tables/mismatches.csv`: **0**
- Rows in `tables/appearance_gt_tree_sides_cases.csv`: **0**

## Per-tree Yield Statistics
|   n_sides |   n_trees |   unique_mean |   unique_median |   unique_std |   det_mean |   det_median |   det_per_unique_mean |   det_per_unique_median |
|----------:|----------:|--------------:|----------------:|-------------:|-----------:|-------------:|----------------------:|------------------------:|
|         4 |       908 |         10.22 |              10 |         3.71 |      18.59 |           19 |                 1.845 |                   1.833 |
|         8 |        45 |         12.11 |              12 |         3.89 |      36.87 |           38 |                 3.107 |                   3.062 |

## Integrity Audit (JSON/TXT/Image)
- Side rows audited: **3,992**
- Missing images: **0**
- Missing labels: **0**
- JSON vs label count exact match: **100.00%**
- JSON vs bbox_count exact match: **100.00%**
- JSON vs summary.by_side exact match: **100.00%**

## Link-Graph Diagnostics
- Trees with cycle_rank > 0: **0**
- Max cycle_rank: **0**
- Max graph degree: **2**

## Class Distribution
- JSON unique bunch B1: 954
- JSON unique bunch B2: 1,791
- JSON unique bunch B3: 5,067
- JSON unique bunch B4: 2,011

### Class Mix per Tree-Type (4-side vs 8-side)
|   n_sides |   n_trees |   B1_total |   B2_total |   B3_total |   B4_total |   B1_per_tree |   B2_per_tree |   B3_per_tree |   B4_per_tree |   B1_pct |   B2_pct |   B3_pct |   B4_pct |
|----------:|----------:|-----------:|-----------:|-----------:|-----------:|--------------:|--------------:|--------------:|--------------:|---------:|---------:|---------:|---------:|
|         4 |       908 |        898 |       1687 |       4756 |       1937 |         0.989 |         1.858 |         5.238 |         2.133 |     9.68 |    18.18 |    51.26 |    20.88 |
|         8 |        45 |         56 |        104 |        311 |         74 |         1.244 |         2.311 |         6.911 |         1.644 |    10.28 |    19.08 |    57.06 |    13.58 |

### Detection Distribution from labels/*.txt
- Label class 0 (B1): 2,032
- Label class 1 (B2): 3,500
- Label class 2 (B3): 9,701
- Label class 3 (B4): 3,307

## Split Summary (from JSON)
| split   |   B1 |   B2 |   B3 |   B4 |   total_unique_bunches |   total_detections |
|:--------|-----:|-----:|-----:|-----:|-----------------------:|-------------------:|
| test    |  163 |  283 |  890 |  378 |                   1714 |               3141 |
| train   |  667 | 1172 | 3240 | 1233 |                   6312 |              11926 |
| val     |  124 |  336 |  937 |  400 |                   1797 |               3473 |

## Top Trees by Detection-per-Unique-Bunch Ratio
| tree_id           | split   |   n_sides |   total_detections |   total_unique_bunches |   det_per_unique |
|:------------------|:--------|----------:|-------------------:|-----------------------:|-----------------:|
| DAMIMAS_A21B_0820 | train   |         8 |                 47 |                     10 |          4.7     |
| DAMIMAS_A21B_0836 | train   |         8 |                 36 |                      9 |          4       |
| DAMIMAS_A21B_0831 | train   |         8 |                 52 |                     13 |          4       |
| DAMIMAS_A21B_0818 | train   |         8 |                 40 |                     10 |          4       |
| DAMIMAS_A21B_0839 | val     |         8 |                 35 |                      9 |          3.88889 |
| DAMIMAS_A21B_0824 | val     |         8 |                 41 |                     11 |          3.72727 |
| DAMIMAS_A21B_0846 | train   |         8 |                 26 |                      7 |          3.71429 |
| DAMIMAS_A21B_0832 | train   |         8 |                 37 |                     10 |          3.7     |
| DAMIMAS_A21B_0815 | train   |         8 |                 51 |                     14 |          3.64286 |
| DAMIMAS_A21B_0848 | val     |         8 |                 25 |                      7 |          3.57143 |
| DAMIMAS_A21B_0826 | val     |         8 |                 21 |                      6 |          3.5     |
| DAMIMAS_A21B_0817 | val     |         8 |                 28 |                      8 |          3.5     |
| DAMIMAS_A21B_0814 | test    |         8 |                 24 |                      7 |          3.42857 |
| DAMIMAS_A21B_0812 | train   |         8 |                 40 |                     12 |          3.33333 |
| DAMIMAS_A21B_0850 | train   |         8 |                 43 |                     13 |          3.30769 |

## Sample Mismatch Cases (same bunch repeated in same side)
- No mismatch rows.

## split_manifest.csv quick checks
- Rows in split_manifest.csv: **953**
- Unique tree_id in split_manifest.csv: **953**

## ground_truth.parquet
- Rows: **953**
- Columns (11): `tree_id, split, varietas, num_sides, total_unique_bunches, B1, B2, B3, B4, total_detections, duplicates_linked`

## Outputs
- Tables: `EDA_report/tables/*.csv`
- Plots: `EDA_report/plots/*.png`
- This summary: `EDA_report/SUMMARY.md`
- Advanced stats: `statistical_drift_tests.csv`, `data_quality_scorecard.csv`, `tree_outlier_scores.csv`
