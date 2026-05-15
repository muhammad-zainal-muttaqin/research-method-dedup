# Anomaly Casebook

Source: `EDA_report/tables/appearance_gt_tree_sides_cases.csv`

Cases where `appearance_count > tree_n_sides` with side-level evidence and `_confirmedLinks` edges touching the bunch.

Total cases: **11**

## DAMIMAS_A21B_0287 / bunch_id=1

- class: `B1`
- tree_n_sides: `4`
- appearance_count: `5`
- unique_side_count: `4`
- same_side_duplicates: `1`

Appearances:
- side `sisi_1` (`0`) / box_index `0` / class `B1`
- side `sisi_2` (`1`) / box_index `0` / class `B1`
- side `sisi_3` (`2`) / box_index `0` / class `B1`
- side `sisi_4` (`3`) / box_index `0` / class `B1`
- side `sisi_4` (`3`) / box_index `2` / class `B1`

Duplicated side slots:
- side_index `3` has multiple boxes: `[0, 2]`

Touching `_confirmedLinks`:
- `lnk-0`: side `0`/b`0` <-> side `1`/b`0` (both_in_bunch=1)
- `lnk-1`: side `1`/b`0` <-> side `2`/b`0` (both_in_bunch=1)
- `lnk-2`: side `2`/b`0` <-> side `3`/b`0` (both_in_bunch=1)
- `lnk-3`: side `3`/b`2` <-> side `0`/b`0` (both_in_bunch=1)

Interpretation:
- This case exceeds tree-side limit, so count inflation is caused by multi-box appearances within one or more sides, not extra camera sides.

## DAMIMAS_A21B_0309 / bunch_id=1

- class: `B1`
- tree_n_sides: `4`
- appearance_count: `5`
- unique_side_count: `4`
- same_side_duplicates: `1`

Appearances:
- side `sisi_1` (`0`) / box_index `0` / class `B1`
- side `sisi_2` (`1`) / box_index `0` / class `B1`
- side `sisi_3` (`2`) / box_index `0` / class `B1`
- side `sisi_3` (`2`) / box_index `4` / class `B1`
- side `sisi_4` (`3`) / box_index `0` / class `B1`

Duplicated side slots:
- side_index `2` has multiple boxes: `[0, 4]`

Touching `_confirmedLinks`:
- `lnk-0`: side `0`/b`0` <-> side `1`/b`0` (both_in_bunch=1)
- `lnk-1`: side `1`/b`0` <-> side `2`/b`0` (both_in_bunch=1)
- `lnk-2`: side `2`/b`4` <-> side `3`/b`0` (both_in_bunch=1)
- `lnk-3`: side `3`/b`0` <-> side `0`/b`0` (both_in_bunch=1)

Interpretation:
- This case exceeds tree-side limit, so count inflation is caused by multi-box appearances within one or more sides, not extra camera sides.

## DAMIMAS_A21B_0320 / bunch_id=4

- class: `B3`
- tree_n_sides: `4`
- appearance_count: `5`
- unique_side_count: `4`
- same_side_duplicates: `1`

Appearances:
- side `sisi_1` (`0`) / box_index `3` / class `B3`
- side `sisi_2` (`1`) / box_index `3` / class `B3`
- side `sisi_3` (`2`) / box_index `4` / class `B3`
- side `sisi_4` (`3`) / box_index `5` / class `B3`
- side `sisi_4` (`3`) / box_index `6` / class `B3`

Duplicated side slots:
- side_index `3` has multiple boxes: `[5, 6]`

Touching `_confirmedLinks`:
- `lnk-4`: side `3`/b`6` <-> side `0`/b`3` (both_in_bunch=1)
- `lnk-5`: side `0`/b`3` <-> side `1`/b`3` (both_in_bunch=1)
- `lnk-6`: side `1`/b`3` <-> side `2`/b`4` (both_in_bunch=1)
- `lnk-7`: side `2`/b`4` <-> side `3`/b`5` (both_in_bunch=1)

Interpretation:
- This case exceeds tree-side limit, so count inflation is caused by multi-box appearances within one or more sides, not extra camera sides.

## DAMIMAS_A21B_0323 / bunch_id=1

- class: `B3`
- tree_n_sides: `4`
- appearance_count: `7`
- unique_side_count: `4`
- same_side_duplicates: `3`

Appearances:
- side `sisi_1` (`0`) / box_index `0` / class `B3`
- side `sisi_1` (`0`) / box_index `2` / class `B3`
- side `sisi_2` (`1`) / box_index `1` / class `B3`
- side `sisi_3` (`2`) / box_index `0` / class `B3`
- side `sisi_3` (`2`) / box_index `2` / class `B3`
- side `sisi_4` (`3`) / box_index `0` / class `B3`
- side `sisi_4` (`3`) / box_index `2` / class `B3`

Duplicated side slots:
- side_index `0` has multiple boxes: `[0, 2]`
- side_index `2` has multiple boxes: `[0, 2]`
- side_index `3` has multiple boxes: `[0, 2]`

Touching `_confirmedLinks`:
- `lnk-4`: side `0`/b`2` <-> side `1`/b`1` (both_in_bunch=1)
- `lnk-5`: side `1`/b`1` <-> side `2`/b`0` (both_in_bunch=1)
- `lnk-6`: side `2`/b`0` <-> side `3`/b`0` (both_in_bunch=1)
- `lnk-7`: side `2`/b`2` <-> side `3`/b`2` (both_in_bunch=1)
- `lnk-8`: side `3`/b`2` <-> side `0`/b`2` (both_in_bunch=1)
- `lnk-9`: side `3`/b`0` <-> side `0`/b`0` (both_in_bunch=1)

Interpretation:
- This case exceeds tree-side limit, so count inflation is caused by multi-box appearances within one or more sides, not extra camera sides.

## DAMIMAS_A21B_0323 / bunch_id=2

- class: `B3`
- tree_n_sides: `4`
- appearance_count: `5`
- unique_side_count: `4`
- same_side_duplicates: `1`

Appearances:
- side `sisi_1` (`0`) / box_index `1` / class `B3`
- side `sisi_2` (`1`) / box_index `0` / class `B3`
- side `sisi_2` (`1`) / box_index `2` / class `B3`
- side `sisi_3` (`2`) / box_index `1` / class `B3`
- side `sisi_4` (`3`) / box_index `1` / class `B3`

Duplicated side slots:
- side_index `1` has multiple boxes: `[0, 2]`

Touching `_confirmedLinks`:
- `lnk-0`: side `0`/b`1` <-> side `1`/b`0` (both_in_bunch=1)
- `lnk-1`: side `1`/b`2` <-> side `2`/b`1` (both_in_bunch=1)
- `lnk-2`: side `2`/b`1` <-> side `3`/b`1` (both_in_bunch=1)
- `lnk-3`: side `3`/b`1` <-> side `0`/b`1` (both_in_bunch=1)

Interpretation:
- This case exceeds tree-side limit, so count inflation is caused by multi-box appearances within one or more sides, not extra camera sides.

## DAMIMAS_A21B_0335 / bunch_id=1

- class: `B1`
- tree_n_sides: `4`
- appearance_count: `6`
- unique_side_count: `4`
- same_side_duplicates: `2`

Appearances:
- side `sisi_1` (`0`) / box_index `0` / class `B1`
- side `sisi_1` (`0`) / box_index `1` / class `B1`
- side `sisi_2` (`1`) / box_index `0` / class `B1`
- side `sisi_2` (`1`) / box_index `3` / class `B1`
- side `sisi_3` (`2`) / box_index `1` / class `B1`
- side `sisi_4` (`3`) / box_index `0` / class `B1`

Duplicated side slots:
- side_index `0` has multiple boxes: `[0, 1]`
- side_index `1` has multiple boxes: `[0, 3]`

Touching `_confirmedLinks`:
- `lnk-0`: side `1`/b`3` <-> side `2`/b`1` (both_in_bunch=1)
- `lnk-1`: side `2`/b`1` <-> side `3`/b`0` (both_in_bunch=1)
- `lnk-2`: side `3`/b`0` <-> side `0`/b`1` (both_in_bunch=1)
- `lnk-3`: side `0`/b`0` <-> side `1`/b`3` (both_in_bunch=1)
- `lnk-4`: side `0`/b`1` <-> side `1`/b`0` (both_in_bunch=1)

Interpretation:
- This case exceeds tree-side limit, so count inflation is caused by multi-box appearances within one or more sides, not extra camera sides.

## DAMIMAS_A21B_0336 / bunch_id=1

- class: `B1`
- tree_n_sides: `4`
- appearance_count: `5`
- unique_side_count: `4`
- same_side_duplicates: `1`

Appearances:
- side `sisi_1` (`0`) / box_index `0` / class `B1`
- side `sisi_2` (`1`) / box_index `0` / class `B1`
- side `sisi_2` (`1`) / box_index `2` / class `B1`
- side `sisi_3` (`2`) / box_index `0` / class `B1`
- side `sisi_4` (`3`) / box_index `0` / class `B1`

Duplicated side slots:
- side_index `1` has multiple boxes: `[0, 2]`

Touching `_confirmedLinks`:
- `lnk-0`: side `0`/b`0` <-> side `1`/b`0` (both_in_bunch=1)
- `lnk-1`: side `2`/b`0` <-> side `3`/b`0` (both_in_bunch=1)
- `lnk-2`: side `3`/b`0` <-> side `0`/b`0` (both_in_bunch=1)
- `lnk-3`: side `1`/b`2` <-> side `2`/b`0` (both_in_bunch=1)

Interpretation:
- This case exceeds tree-side limit, so count inflation is caused by multi-box appearances within one or more sides, not extra camera sides.

## DAMIMAS_A21B_0359 / bunch_id=1

- class: `B1`
- tree_n_sides: `4`
- appearance_count: `5`
- unique_side_count: `4`
- same_side_duplicates: `1`

Appearances:
- side `sisi_1` (`0`) / box_index `0` / class `B1`
- side `sisi_2` (`1`) / box_index `1` / class `B1`
- side `sisi_3` (`2`) / box_index `2` / class `B1`
- side `sisi_4` (`3`) / box_index `0` / class `B1`
- side `sisi_4` (`3`) / box_index `4` / class `B1`

Duplicated side slots:
- side_index `3` has multiple boxes: `[0, 4]`

Touching `_confirmedLinks`:
- `lnk-3`: side `0`/b`0` <-> side `1`/b`1` (both_in_bunch=1)
- `lnk-4`: side `1`/b`1` <-> side `2`/b`2` (both_in_bunch=1)
- `lnk-5`: side `2`/b`2` <-> side `3`/b`0` (both_in_bunch=1)
- `lnk-6`: side `3`/b`4` <-> side `0`/b`0` (both_in_bunch=1)

Interpretation:
- This case exceeds tree-side limit, so count inflation is caused by multi-box appearances within one or more sides, not extra camera sides.

## DAMIMAS_A21B_0362 / bunch_id=1

- class: `B3`
- tree_n_sides: `4`
- appearance_count: `8`
- unique_side_count: `4`
- same_side_duplicates: `4`

Appearances:
- side `sisi_1` (`0`) / box_index `0` / class `B3`
- side `sisi_1` (`0`) / box_index `2` / class `B3`
- side `sisi_2` (`1`) / box_index `0` / class `B3`
- side `sisi_2` (`1`) / box_index `5` / class `B3`
- side `sisi_3` (`2`) / box_index `1` / class `B3`
- side `sisi_3` (`2`) / box_index `3` / class `B3`
- side `sisi_4` (`3`) / box_index `0` / class `B3`
- side `sisi_4` (`3`) / box_index `1` / class `B3`

Duplicated side slots:
- side_index `0` has multiple boxes: `[0, 2]`
- side_index `1` has multiple boxes: `[0, 5]`
- side_index `2` has multiple boxes: `[1, 3]`
- side_index `3` has multiple boxes: `[0, 1]`

Touching `_confirmedLinks`:
- `lnk-5`: side `0`/b`0` <-> side `1`/b`0` (both_in_bunch=1)
- `lnk-6`: side `0`/b`2` <-> side `1`/b`5` (both_in_bunch=1)
- `lnk-7`: side `2`/b`3` <-> side `3`/b`0` (both_in_bunch=1)
- `lnk-8`: side `2`/b`1` <-> side `3`/b`1` (both_in_bunch=1)
- `lnk-9`: side `3`/b`0` <-> side `0`/b`0` (both_in_bunch=1)
- `lnk-10`: side `1`/b`0` <-> side `2`/b`1` (both_in_bunch=1)
- `lnk-11`: side `1`/b`5` <-> side `2`/b`3` (both_in_bunch=1)

Interpretation:
- This case exceeds tree-side limit, so count inflation is caused by multi-box appearances within one or more sides, not extra camera sides.

## DAMIMAS_A21B_0362 / bunch_id=2

- class: `B3`
- tree_n_sides: `4`
- appearance_count: `5`
- unique_side_count: `4`
- same_side_duplicates: `1`

Appearances:
- side `sisi_1` (`0`) / box_index `1` / class `B3`
- side `sisi_1` (`0`) / box_index `4` / class `B3`
- side `sisi_2` (`1`) / box_index `6` / class `B3`
- side `sisi_3` (`2`) / box_index `2` / class `B3`
- side `sisi_4` (`3`) / box_index `4` / class `B3`

Duplicated side slots:
- side_index `0` has multiple boxes: `[1, 4]`

Touching `_confirmedLinks`:
- `lnk-12`: side `0`/b`1` <-> side `1`/b`6` (both_in_bunch=1)
- `lnk-13`: side `2`/b`2` <-> side `3`/b`4` (both_in_bunch=1)
- `lnk-14`: side `3`/b`4` <-> side `0`/b`4` (both_in_bunch=1)
- `lnk-15`: side `1`/b`6` <-> side `2`/b`2` (both_in_bunch=1)

Interpretation:
- This case exceeds tree-side limit, so count inflation is caused by multi-box appearances within one or more sides, not extra camera sides.

## DAMIMAS_A21B_0362 / bunch_id=3

- class: `B3`
- tree_n_sides: `4`
- appearance_count: `6`
- unique_side_count: `4`
- same_side_duplicates: `2`

Appearances:
- side `sisi_1` (`0`) / box_index `3` / class `B3`
- side `sisi_2` (`1`) / box_index `4` / class `B3`
- side `sisi_3` (`2`) / box_index `4` / class `B3`
- side `sisi_3` (`2`) / box_index `5` / class `B3`
- side `sisi_4` (`3`) / box_index `2` / class `B3`
- side `sisi_4` (`3`) / box_index `3` / class `B3`

Duplicated side slots:
- side_index `2` has multiple boxes: `[4, 5]`
- side_index `3` has multiple boxes: `[2, 3]`

Touching `_confirmedLinks`:
- `lnk-0`: side `0`/b`3` <-> side `1`/b`4` (both_in_bunch=1)
- `lnk-1`: side `2`/b`4` <-> side `3`/b`3` (both_in_bunch=1)
- `lnk-2`: side `2`/b`5` <-> side `3`/b`2` (both_in_bunch=1)
- `lnk-3`: side `3`/b`3` <-> side `0`/b`3` (both_in_bunch=1)
- `lnk-4`: side `1`/b`4` <-> side `2`/b`5` (both_in_bunch=1)

Interpretation:
- This case exceeds tree-side limit, so count inflation is caused by multi-box appearances within one or more sides, not extra camera sides.
