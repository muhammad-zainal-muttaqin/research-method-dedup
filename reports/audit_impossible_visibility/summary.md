# Audit: impossible bunch visibility (geometric adjacency rule)

## Rule

Bunch wajib punya appearance di **home side** (posisi fisik bunch).
Appearance lain harus dalam circular distance ≤ `max_dist` dari home.

| n_sides_total | max_dist (hop) | normal max sides | hard max sides |
|---:|---:|---:|---:|
| 4 | 1 | 2 | 3 |
| 8 | 3 | 4 | 6 |

## Results

- JSON scanned: 953
- Trees with violation: **0**
- Trees with warning only: 469
- Bunches violation: **0**
- Bunches warning: 802
- Trees skipped (n_sides not in [4, 8]): 0

## Violations

Bunch yg tidak punya geometric valid home — secara fisik mustahil.

(none)

## Warnings

Bunch valid (geometric OK) tapi pakai full reach — borderline normal.

| tree_id | bunch | class | sides_bunch | sides total | appearance_sides | valid_home |
|---|---:|:---:|:---:|:---:|---|:---:|
| DAMIMAS_A21B_0001 | 5 | B2 | 3 | 4 | side_1,side_2,side_3 | side_2 |
| DAMIMAS_A21B_0001 | 6 | B3 | 3 | 4 | side_2,side_3,side_4 | side_3 |
| DAMIMAS_A21B_0002 | 2 | B4 | 3 | 4 | side_1,side_2,side_3 | side_2 |
| DAMIMAS_A21B_0002 | 5 | B3 | 3 | 4 | side_1,side_2,side_3 | side_2 |
| DAMIMAS_A21B_0002 | 8 | B4 | 3 | 4 | side_2,side_3,side_4 | side_3 |
| DAMIMAS_A21B_0002 | 9 | B1 | 3 | 4 | side_2,side_3,side_4 | side_3 |
| DAMIMAS_A21B_0003 | 5 | B3 | 3 | 4 | side_1,side_3,side_4 | side_4 |
| DAMIMAS_A21B_0003 | 6 | B2 | 3 | 4 | side_2,side_3,side_4 | side_3 |
| DAMIMAS_A21B_0005 | 3 | B3 | 3 | 4 | side_1,side_2,side_3 | side_2 |
| DAMIMAS_A21B_0005 | 5 | B2 | 3 | 4 | side_2,side_3,side_4 | side_3 |
| DAMIMAS_A21B_0006 | 2 | B4 | 3 | 4 | side_1,side_2,side_3 | side_2 |
| DAMIMAS_A21B_0006 | 3 | B3 | 3 | 4 | side_1,side_2,side_3 | side_2 |
| DAMIMAS_A21B_0009 | 4 | B3 | 3 | 4 | side_2,side_3,side_4 | side_3 |
| DAMIMAS_A21B_0010 | 1 | B2 | 3 | 4 | side_1,side_2,side_3 | side_2 |
| DAMIMAS_A21B_0010 | 7 | B3 | 3 | 4 | side_2,side_3,side_4 | side_3 |
| DAMIMAS_A21B_0011 | 2 | B3 | 3 | 4 | side_1,side_2,side_4 | side_1 |
| DAMIMAS_A21B_0011 | 4 | B3 | 3 | 4 | side_2,side_3,side_4 | side_3 |
| DAMIMAS_A21B_0012 | 1 | B1 | 3 | 4 | side_1,side_3,side_4 | side_4 |
| DAMIMAS_A21B_0012 | 4 | B4 | 3 | 4 | side_1,side_3,side_4 | side_4 |
| DAMIMAS_A21B_0015 | 1 | B1 | 3 | 4 | side_1,side_2,side_3 | side_2 |
| DAMIMAS_A21B_0017 | 5 | B4 | 3 | 4 | side_1,side_3,side_4 | side_4 |
| DAMIMAS_A21B_0018 | 2 | B3 | 3 | 4 | side_1,side_2,side_3 | side_2 |
| DAMIMAS_A21B_0024 | 2 | B1 | 3 | 4 | side_1,side_3,side_4 | side_4 |
| DAMIMAS_A21B_0024 | 5 | B3 | 3 | 4 | side_1,side_2,side_4 | side_1 |
| DAMIMAS_A21B_0025 | 1 | B1 | 3 | 4 | side_1,side_2,side_3 | side_2 |
| DAMIMAS_A21B_0027 | 1 | B1 | 3 | 4 | side_1,side_2,side_4 | side_1 |
| DAMIMAS_A21B_0028 | 8 | B3 | 3 | 4 | side_2,side_3,side_4 | side_3 |
| DAMIMAS_A21B_0029 | 5 | B2 | 3 | 4 | side_1,side_2,side_3 | side_2 |
| DAMIMAS_A21B_0029 | 7 | B2 | 3 | 4 | side_2,side_3,side_4 | side_3 |
| DAMIMAS_A21B_0029 | 8 | B4 | 3 | 4 | side_2,side_3,side_4 | side_3 |
| DAMIMAS_A21B_0030 | 2 | B3 | 3 | 4 | side_1,side_3,side_4 | side_4 |
| DAMIMAS_A21B_0031 | 2 | B3 | 3 | 4 | side_1,side_2,side_3 | side_2 |
| DAMIMAS_A21B_0037 | 8 | B3 | 3 | 4 | side_2,side_3,side_4 | side_3 |
| DAMIMAS_A21B_0038 | 4 | B3 | 3 | 4 | side_1,side_2,side_4 | side_1 |
| DAMIMAS_A21B_0041 | 2 | B4 | 3 | 4 | side_1,side_3,side_4 | side_4 |
| DAMIMAS_A21B_0042 | 1 | B1 | 3 | 4 | side_1,side_3,side_4 | side_4 |
| DAMIMAS_A21B_0043 | 2 | B3 | 3 | 4 | side_1,side_2,side_4 | side_1 |
| DAMIMAS_A21B_0043 | 3 | B2 | 3 | 4 | side_1,side_2,side_3 | side_2 |
| DAMIMAS_A21B_0043 | 5 | B3 | 3 | 4 | side_1,side_3,side_4 | side_4 |
| DAMIMAS_A21B_0046 | 2 | B3 | 3 | 4 | side_1,side_3,side_4 | side_4 |
| DAMIMAS_A21B_0046 | 6 | B4 | 3 | 4 | side_2,side_3,side_4 | side_3 |
| DAMIMAS_A21B_0047 | 1 | B2 | 3 | 4 | side_1,side_3,side_4 | side_4 |
| DAMIMAS_A21B_0048 | 1 | B3 | 3 | 4 | side_1,side_2,side_4 | side_1 |
| DAMIMAS_A21B_0048 | 6 | B3 | 3 | 4 | side_2,side_3,side_4 | side_3 |
| DAMIMAS_A21B_0049 | 2 | B3 | 3 | 4 | side_1,side_2,side_3 | side_2 |
| DAMIMAS_A21B_0050 | 6 | B3 | 3 | 4 | side_2,side_3,side_4 | side_3 |
| DAMIMAS_A21B_0052 | 8 | B1 | 3 | 4 | side_2,side_3,side_4 | side_3 |
| DAMIMAS_A21B_0054 | 3 | B2 | 3 | 4 | side_1,side_3,side_4 | side_4 |
| DAMIMAS_A21B_0054 | 4 | B3 | 3 | 4 | side_1,side_3,side_4 | side_4 |
| DAMIMAS_A21B_0058 | 1 | B1 | 3 | 4 | side_1,side_3,side_4 | side_4 |
| DAMIMAS_A21B_0058 | 2 | B3 | 3 | 4 | side_1,side_2,side_4 | side_1 |
| DAMIMAS_A21B_0059 | 1 | B1 | 3 | 4 | side_1,side_2,side_4 | side_1 |
| DAMIMAS_A21B_0059 | 2 | B3 | 3 | 4 | side_1,side_2,side_3 | side_2 |
| DAMIMAS_A21B_0060 | 1 | B3 | 3 | 4 | side_1,side_2,side_3 | side_2 |
| DAMIMAS_A21B_0060 | 2 | B3 | 3 | 4 | side_1,side_2,side_4 | side_1 |
| DAMIMAS_A21B_0060 | 3 | B3 | 3 | 4 | side_1,side_2,side_4 | side_1 |
| DAMIMAS_A21B_0061 | 5 | B3 | 3 | 4 | side_1,side_3,side_4 | side_4 |
| DAMIMAS_A21B_0062 | 1 | B2 | 3 | 4 | side_1,side_3,side_4 | side_4 |
| DAMIMAS_A21B_0062 | 3 | B3 | 3 | 4 | side_1,side_2,side_3 | side_2 |
| DAMIMAS_A21B_0062 | 5 | B2 | 3 | 4 | side_2,side_3,side_4 | side_3 |
| DAMIMAS_A21B_0064 | 4 | B2 | 3 | 4 | side_2,side_3,side_4 | side_3 |
| DAMIMAS_A21B_0065 | 1 | B1 | 3 | 4 | side_1,side_2,side_4 | side_1 |
| DAMIMAS_A21B_0067 | 3 | B1 | 3 | 4 | side_1,side_2,side_3 | side_2 |
| DAMIMAS_A21B_0067 | 6 | B4 | 3 | 4 | side_1,side_2,side_4 | side_1 |
| DAMIMAS_A21B_0068 | 3 | B3 | 3 | 4 | side_1,side_3,side_4 | side_4 |
| DAMIMAS_A21B_0069 | 1 | B1 | 3 | 4 | side_1,side_2,side_4 | side_1 |
| DAMIMAS_A21B_0070 | 1 | B3 | 3 | 4 | side_1,side_2,side_3 | side_2 |
| DAMIMAS_A21B_0070 | 6 | B3 | 3 | 4 | side_2,side_3,side_4 | side_3 |
| DAMIMAS_A21B_0071 | 3 | B3 | 3 | 4 | side_1,side_2,side_4 | side_1 |
| DAMIMAS_A21B_0071 | 7 | B2 | 3 | 4 | side_2,side_3,side_4 | side_3 |
| DAMIMAS_A21B_0075 | 1 | B1 | 3 | 4 | side_1,side_2,side_4 | side_1 |
| DAMIMAS_A21B_0076 | 1 | B1 | 3 | 4 | side_1,side_3,side_4 | side_4 |
| DAMIMAS_A21B_0077 | 2 | B1 | 3 | 4 | side_1,side_3,side_4 | side_4 |
| DAMIMAS_A21B_0078 | 5 | B2 | 3 | 4 | side_1,side_2,side_3 | side_2 |
| DAMIMAS_A21B_0079 | 1 | B2 | 3 | 4 | side_1,side_2,side_3 | side_2 |
| DAMIMAS_A21B_0079 | 6 | B3 | 3 | 4 | side_1,side_2,side_3 | side_2 |
| DAMIMAS_A21B_0079 | 9 | B3 | 3 | 4 | side_2,side_3,side_4 | side_3 |
| DAMIMAS_A21B_0081 | 7 | B2 | 3 | 4 | side_2,side_3,side_4 | side_3 |
| DAMIMAS_A21B_0081 | 9 | B3 | 3 | 4 | side_2,side_3,side_4 | side_3 |
| DAMIMAS_A21B_0084 | 3 | B3 | 3 | 4 | side_1,side_2,side_3 | side_2 |

(... 722 more — see findings.csv)
