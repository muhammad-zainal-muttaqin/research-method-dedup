# EDA Report - Brand-New-Dataset-YOLO

## Scope
- Source: `Brand-New-Dataset-YOLO/`
- JSON GT files analyzed from `json/*.json`
- Label detections analyzed from `labels/*.txt`
- Split metadata from `split_manifest.csv`
- Optional parquet read from `data/ground_truth.parquet`

## Global Counts
- Trees (JSON): **953**
- Unique bunches: **9,739**
- Annotation rows (YOLO-like entries in JSON images): **18,541**
- Confirmed links: **8,802**

## Side Distribution (Trees)
- 4 sides: 908 trees
- 8 sides: 45 trees

## Appearance Distribution (Unique Bunches)
- appearance_count=1: 2,448 (25.1%)
- appearance_count=2: 6,203 (63.7%)
- appearance_count=3: 800 (8.2%)
- appearance_count=4: 186 (1.9%)
- appearance_count=5: 78 (0.8%)
- appearance_count=6: 17 (0.2%)
- appearance_count=7: 5 (0.1%)
- appearance_count=8: 2 (0.0%)

## Unique Side Count Distribution (Unique Bunches)
- unique_side_count=1: 2,448 (25.1%)
- unique_side_count=2: 6,203 (63.7%)
- unique_side_count=3: 800 (8.2%)
- unique_side_count=4: 197 (2.0%)
- unique_side_count=5: 71 (0.7%)
- unique_side_count=6: 15 (0.2%)
- unique_side_count=7: 4 (0.0%)
- unique_side_count=8: 1 (0.0%)

## Same-side Duplicate Distribution
- same_side_duplicate_count=0: 9,728 (99.89%)
- same_side_duplicate_count=1: 7 (0.07%)
- same_side_duplicate_count=2: 2 (0.02%)
- same_side_duplicate_count=3: 1 (0.01%)
- same_side_duplicate_count=4: 1 (0.01%)

## Key Anomaly Counters
- Bunches with `appearance_count > 4`: **102**
- Bunches with `appearance_count > tree_n_sides`: **11**
- Rows in `tables/mismatches.csv`: **11**
- Rows in `tables/appearance_gt_tree_sides_cases.csv`: **11**

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
- JSON unique bunch B1: 937
- JSON unique bunch B2: 1,780
- JSON unique bunch B3: 5,013
- JSON unique bunch B4: 2,009

### Detection Distribution from labels/*.txt
- Label class 0 (B1): 2,046
- Label class 1 (B2): 3,493
- Label class 2 (B3): 9,688
- Label class 3 (B4): 3,314

## Split Summary (from JSON)
| split   |   B1 |   B2 |   B3 |   B4 |   total_unique_bunches |   total_detections |
|:--------|-----:|-----:|-----:|-----:|-----------------------:|-------------------:|
| test    |  160 |  283 |  888 |  376 |                   1707 |               3154 |
| train   |  648 | 1158 | 3188 | 1231 |                   6225 |              11851 |
| val     |  129 |  339 |  937 |  402 |                   1807 |               3536 |

## Top Trees by Detection-per-Unique-Bunch Ratio
| tree_id           | split   |   n_sides |   total_detections |   total_unique_bunches |   det_per_unique |
|:------------------|:--------|----------:|-------------------:|-----------------------:|-----------------:|
| DAMIMAS_A21B_0848 | val     |         8 |                 25 |                      5 |          5       |
| DAMIMAS_A21B_0820 | train   |         8 |                 47 |                     10 |          4.7     |
| DAMIMAS_A21B_0323 | test    |         4 |                 13 |                      3 |          4.33333 |
| DAMIMAS_A21B_0824 | val     |         8 |                 41 |                     10 |          4.1     |
| DAMIMAS_A21B_0823 | train   |         8 |                 57 |                     14 |          4.07143 |
| DAMIMAS_A21B_0836 | train   |         8 |                 36 |                      9 |          4       |
| DAMIMAS_A21B_0831 | train   |         8 |                 52 |                     13 |          4       |
| DAMIMAS_A21B_0818 | train   |         8 |                 40 |                     10 |          4       |
| DAMIMAS_A21B_0839 | val     |         8 |                 35 |                      9 |          3.88889 |
| DAMIMAS_A21B_0846 | train   |         8 |                 26 |                      7 |          3.71429 |
| DAMIMAS_A21B_0362 | train   |         4 |                 26 |                      7 |          3.71429 |
| DAMIMAS_A21B_0832 | train   |         8 |                 37 |                     10 |          3.7     |
| DAMIMAS_A21B_0815 | train   |         8 |                 51 |                     14 |          3.64286 |
| DAMIMAS_A21B_0812 | val     |         8 |                 40 |                     11 |          3.63636 |
| LONSUM_A21A_0049  | test    |         4 |                  7 |                      2 |          3.5     |

## Sample Mismatch Cases (same bunch repeated in same side)
| tree_id           | split   | domain   |   bunch_id | class   |   appearance_count |   unique_side_count |   same_side_duplicate_count |   tree_n_sides |
|:------------------|:--------|:---------|-----------:|:--------|-------------------:|--------------------:|----------------------------:|---------------:|
| DAMIMAS_A21B_0362 | train   | DAMIMAS  |          1 | B3      |                  8 |                   4 |                           4 |              4 |
| DAMIMAS_A21B_0323 | test    | DAMIMAS  |          1 | B3      |                  7 |                   4 |                           3 |              4 |
| DAMIMAS_A21B_0335 | train   | DAMIMAS  |          1 | B1      |                  6 |                   4 |                           2 |              4 |
| DAMIMAS_A21B_0362 | train   | DAMIMAS  |          3 | B3      |                  6 |                   4 |                           2 |              4 |
| DAMIMAS_A21B_0287 | train   | DAMIMAS  |          1 | B1      |                  5 |                   4 |                           1 |              4 |
| DAMIMAS_A21B_0309 | train   | DAMIMAS  |          1 | B1      |                  5 |                   4 |                           1 |              4 |
| DAMIMAS_A21B_0320 | train   | DAMIMAS  |          4 | B3      |                  5 |                   4 |                           1 |              4 |
| DAMIMAS_A21B_0323 | test    | DAMIMAS  |          2 | B3      |                  5 |                   4 |                           1 |              4 |
| DAMIMAS_A21B_0336 | train   | DAMIMAS  |          1 | B1      |                  5 |                   4 |                           1 |              4 |
| DAMIMAS_A21B_0359 | val     | DAMIMAS  |          1 | B1      |                  5 |                   4 |                           1 |              4 |
| DAMIMAS_A21B_0362 | train   | DAMIMAS  |          2 | B3      |                  5 |                   4 |                           1 |              4 |

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
