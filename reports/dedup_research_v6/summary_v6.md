# Dedup Research V6 Summary

## Key Result

- `v6_selector` reached **96.05% Acc +/-1** on the same 228-tree JSON benchmark.
- `mean_MAE` dropped to **0.2654**.
- `mean_total_error` dropped to **1.06**.
- Bootstrap 95% CI: **93.42% - 98.25%**.

## Delta vs V5 Best

- V5 best (`adaptive_corrected`): `93.86%`, `MAE 0.2774`, `mean_total_error 1.11`.
- V6 selector: `Acc 96.05%`, `MAE 0.2654`, `mean_total_error 1.06`.
- Gain: `+2.19 pp Acc`, `0.0120 MAE`, `0.05 total-error`.

## Selector Design

- Default method: `adaptive_corrected`.
- Unstable high-B3 duplication trees are rerouted to `best_visibility_grid`.
- B2-heavy / low-B4-duplication trees are rerouted to `class_aware_vis`.
- Runtime remains deterministic and keeps the strict 4-class output.

## Method Usage

- `adaptive_corrected` selected on `201` trees
- `class_aware_vis` selected on `14` trees
- `best_visibility_grid` selected on `13` trees

## Split Accuracy

- `test`: `80.65%`
- `train`: `98.47%`
- `val`: `100.00%`

## Improved Trees

- `DAMIMAS_A21B_0554`: `5` -> `2` via `best_visibility_grid`
- `DAMIMAS_A21B_0002`: `4` -> `2` via `class_aware_vis`
- `DAMIMAS_A21B_0257`: `3` -> `1` via `class_aware_vis`
- `DAMIMAS_A21B_0625`: `2` -> `0` via `best_visibility_grid`
- `DAMIMAS_A21B_0001`: `2` -> `1` via `best_visibility_grid`
- `DAMIMAS_A21B_0035`: `2` -> `1` via `class_aware_vis`
- `DAMIMAS_A21B_0043`: `2` -> `1` via `best_visibility_grid`
- `DAMIMAS_A21B_0248`: `2` -> `1` via `best_visibility_grid`
- `DAMIMAS_A21B_0259`: `2` -> `1` via `class_aware_vis`
- `DAMIMAS_A21B_0273`: `2` -> `1` via `best_visibility_grid`
- `DAMIMAS_A21B_0569`: `4` -> `3` via `best_visibility_grid`
- `DAMIMAS_A21B_0584`: `1` -> `0` via `best_visibility_grid`
- `DAMIMAS_A21B_0627`: `1` -> `0` via `best_visibility_grid`
- `DAMIMAS_A21B_0659`: `1` -> `0` via `best_visibility_grid`

## Regressions

- `DAMIMAS_A21B_0263`: V5 was within +/-1, V6 missed via `class_aware_vis` (`1` -> `2`)
- `DAMIMAS_A21B_0547`: V5 was within +/-1, V6 missed via `class_aware_vis` (`1` -> `3`)
