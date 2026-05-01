# Folder ini berisi implementasi algoritma dedup yang telah dibersihkan dari
# boilerplate evaluasi. Setiap file = satu algoritma dengan fungsi predict().
#
# Semua algoritma bersifat:
#   - Deterministik (output sama untuk input yang sama)
#   - Tanpa training / gradient / embedding
#   - Closed-form atau heuristik berbasis statistik dataset
#
# Ranking performa pada benchmark JSON 228 pohon (Acc ±1):
#   Rank  File                      Gen  Acc      MAE     Gagal
#   1     v9_selector.py            v9   97.37%   0.2533  6   (original 228; v10 92.11%)
#   2     v10_selector.py           v10  92.11%   0.2785  18  <- B23-density on 228 (regressed)
#   2*    v10_selector.py           v10  89.27%   0.3167  78  <- B23-density on 727 (MAE improved vs v9)

#   2     b2_median_v6.py           v9   96.49%   0.2588  8
#   3     v6_selector.py            v6   96.49%   0.2632  8
#   4     median_strong5.py         v9   95.18%   0.2390  11
#   5     stacking_bracketed.py     v7   94.30%   0.2643  13
#   6     stacking_density.py       v7   94.30%   0.2708  13
#   7     entropy_modulated.py      v8   94.30%   0.2763  13
#   8     adaptive_corrected.py     v5   93.86%   0.2774  14
#   9     b2_b4_boosted.py          v8   92.54%   0.2632  17
#   10    best_visibility_grid.py   v5   92.54%   0.2664  17
#   11    ordinal_b3.py             v7   91.23%   0.2939  20   ← broken di non-JSON
#   12    class_aware_vis.py        v5   85.09%   0.3805  34
#   13    side_agreement.py         v8   83.33%   0.3618  38
#   14    floor_anchor_50.py        v8   69.74%   0.4211  69   ← spesialis
#   —     multi_consensus.py        v8   18.86%   0.9583  185  ← undercount ekstrem
#   —     per_side_median.py        v8   18.86%   0.9583  185  ← undercount ekstrem
