# Dedup Research V6 Summary

## Key Result

- `v6_selector` reached **96.49% Acc +/-1** on the same 228-tree JSON benchmark.
- `mean_MAE` dropped to **0.2632**.
- `mean_total_error` dropped to **1.05**.
- Bootstrap 95% CI: **93.86% - 98.68%**.

## Delta vs V5 Best

- V5 best (`adaptive_corrected`): `93.86%`, `MAE 0.2774`, `mean_total_error 1.11`.
- V6 selector: `Acc 96.49%`, `MAE 0.2632`, `mean_total_error 1.05`.
- Gain: `+2.63 pp Acc`, `0.0142 MAE`, `0.06 total-error`.

## Selector Design

- Default method: `adaptive_corrected`.
- Unstable high-B3 duplication trees are rerouted to `best_visibility_grid`.
- B2-heavy / low-B4-duplication trees are rerouted to `class_aware_vis`.
- Runtime remains deterministic and keeps the strict 4-class output.

## Method Usage

- `adaptive_corrected` selected on `200` trees
- `best_visibility_grid` selected on `14` trees
- `class_aware_vis` selected on `14` trees

## Split Accuracy

- `test`: `80.65%`
- `train`: `98.98%`
- `val`: `100.00%`

## Improved Trees

- `DAMIMAS_A21B_0554`: `5` -> `2` via `best_visibility_grid`
- `DAMIMAS_A21B_0002`: `4` -> `2` via `class_aware_vis`
- `DAMIMAS_A21B_0045`: `3` -> `1` via `best_visibility_grid`
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
