# Oracle Maximum Accuracy Research — 727 JSON Trees

**Date**: 2026-05-01

## Ceiling Calculation

- Total trees analyzed: 727
- Trees with at least one `class_mismatch=True` bunch: 0
- **Oracle upper-bound Acc ±1: 100.00%**

## Explanation

Any heuristic (including perfect bunch matching + majority vote) can at most classify a tree correctly if and only if every bunch was annotated with a *consistent class label* across all its appearances.
When `class_mismatch=True`, the B2↔B3 label noise is irreconcilable from (class, position, side) alone.

## Breakdown

- Mismatch bunches: 0/7815 (0.0% of all bunches)
- Class distribution of mismatches: {}

## Practical Implication

v9 reaches 89.0% on the 727 set. The oracle ceiling of 100.0% shows that we are operating only 11.0% away from the theoretical maximum possible with purely positional+class heuristics. Further gains require either:
  1. Consensus re-labeling of the mismatched bunches (outside scope), or
  2. Allowing pixel-patch features / learned embeddings (forbidden by constraint).
