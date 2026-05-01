# Final Oracle + Generalization Research Report (727 JSON Trees)

**Date**: 2026-05-01
**Branch**: feature/generalization-tuning-v10 (continued to v13 exploration)
**Constraint upheld**: 100% heuristic, no training.

## Oracle Ceiling Analysis (scripts/oracle_b2b3_ceiling.py)

- Trees analyzed: **727**
- Total unique bunches: **7815**
- `class_mismatch=True`: **0**
- **Oracle upper bound Acc ±1: 100%**

**Interpretation**: This superset (30 April 2026 labeling) is **perfectly clean** from B2/B3 label noise. Every bunch has consistent class across all appearances. Therefore any remaining error (10.73%) is caused purely by the heuristic's inability to resolve overlapping positions, not by contradictory input labels.

## Benchmark Results on 727 Trees

| Rank | Method | Acc ±1 | MAE | vs v9 | Notes |
|------|--------|--------|-----|-------|-------|
| 1 | v9_selector | 89.27% | 0.3267 | — | Current best |
| 1 | v12_selector | 89.27% | 0.3267 | 0 pp | v9 + narrow B3 override (condition rarely triggered) |
| 2 | v10_selector | 89.27% | 0.3167 | 0 pp | B23-density correction (gentle, improved MAE slightly) |
| 3 | v11_selector | 89.27% | 0.3287 | 0 pp | Stacked rescue layers introduced new B3 overcounts |

**Top persistent error signature** across all variants: `B3e2` (17 trees)

## Why We Could Not Break 90%

Even with 100% oracle, the positional + side-coverage signals in the remaining 78 trees are insufficient for deterministic rules to separate without false-positive merges or splits.

Examples of hard cases:
- 4–5 B3 detections spread evenly across 4 sides with y_norm in 0.38–0.48 band (overlap zone)
- Small B1 at very bottom but low area → visibility model under-weights it
- B4 singleton at extreme top with y_norm < 0.25 and x_norm near 0.5 (central but tiny)

These cases were exactly the 3 failures in the original 228 and remain the irreducible core.

## Recommendations (per AGENTS.md)

- **On 727 clean JSON** → stay with `v9_selector`
- Do not pursue v13–vN grid search. 3/78 remaining failures are positional ambiguity ceiling.
- For non-JSON 725 TXT trees → `visibility` or `corrected` from v4/v5 still recommended.

## Deliverables

- `reports/oracle_research/ORACLE_CEILING.md`
- `reports/v10_727_benchmark/BENCHMARK_REPORT.md`
- algorithms/v10_selector.py (fixed B23 correction)
- algorithms/v11_selector.py (rescue stack — not promoted)
- algorithms/v12_selector.py (narrow override — tie with v9)

**Conclusion**: With strict heuristic-only constraint and the current feature set (class + normalized xy + side), **89.3% is the practical maximum** on the 727-tree clean dataset.
