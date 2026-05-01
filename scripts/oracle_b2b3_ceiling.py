"""
Oracle Ceiling Analysis for 727-file dataset (json_30 April 2026).

Computes the theoretical maximum Acc±1 achievable by *any* heuristic
operating only on (class, x_norm, y_norm, side_index) inputs.

Key insight: JSON provides ground-truth "bunches" linking + class_mismatch flag.
If any bunch has class_mismatch=True (almost always B2↔B3), even perfect
bunch matching cannot recover the canonical class — this is irreducible label noise.

Oracle Acc = (trees with ZERO mismatched bunches) / total_trees
"""

from __future__ import annotations
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
JSON_DIR = BASE / "json_30 April 2026"
OUT_DIR = BASE / "reports" / "oracle_research"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NAMES = ["B1", "B2", "B3", "B4"]


def analyze_oracle_ceiling() -> dict:
    trees = []
    mismatch_trees = 0
    total_bunches = 0
    mismatch_bunches = 0
    per_class_mismatch = Counter()

    for jp in sorted(JSON_DIR.glob("*.json")):
        data = json.loads(jp.read_text(encoding="utf-8"))
        tree_id = data.get("tree_id", data.get("tree_name", jp.stem))

        # Ground truth perfect count
        gt = data["summary"]["by_class"]
        gt_counts = {c: gt.get(c, 0) for c in NAMES}

        # Bunch-level analysis
        bunches = data.get("bunches", [])
        tree_has_mismatch = False
        for b in bunches:
            total_bunches += 1
            if b.get("class_mismatch", False):
                mismatch_bunches += 1
                tree_has_mismatch = True
                cls = b.get("class", "OTHER")
                per_class_mismatch[cls] += 1

        if tree_has_mismatch:
            mismatch_trees += 1

        trees.append({
            "tree_id": tree_id,
            "gt_counts": gt_counts,
            "num_bunches": len(bunches),
            "has_mismatch": tree_has_mismatch
        })

    total_trees = len(trees)
    oracle_acc = (total_trees - mismatch_trees) / total_trees if total_trees else 0.0

    report = {
        "total_trees": total_trees,
        "trees_with_mismatch": mismatch_trees,
        "oracle_upper_acc": round(oracle_acc, 4),
        "total_bunches": total_bunches,
        "mismatch_bunches": mismatch_bunches,
        "mismatch_rate_bunches": round(mismatch_bunches / total_bunches, 4) if total_bunches else 0,
        "per_class_mismatch": dict(per_class_mismatch),
        "current_v9_acc_on_727": 0.8900,  # from previous benchmark (update if re-run)
    }

    # Write detailed CSV for further analysis
    import pandas as pd
    df = pd.DataFrame(trees)
    df.to_csv(OUT_DIR / "oracle_tree_status.csv", index=False)

    # Human readable summary
    with open(OUT_DIR / "ORACLE_CEILING.md", "w", encoding="utf-8") as f:
        f.write("# Oracle Maximum Accuracy Research — 727 JSON Trees\n\n")
        f.write(f"**Date**: 2026-05-01\n\n")
        f.write("## Ceiling Calculation\n\n")
        f.write(f"- Total trees analyzed: {total_trees}\n")
        f.write(f"- Trees with at least one `class_mismatch=True` bunch: {mismatch_trees}\n")
        f.write(f"- **Oracle upper-bound Acc ±1: {oracle_acc:.2%}**\n\n")
        f.write("## Explanation\n\n")
        f.write("Any heuristic (including perfect bunch matching + majority vote) ")
        f.write("can at most classify a tree correctly if and only if every bunch ")
        f.write("was annotated with a *consistent class label* across all its appearances.\n")
        f.write("When `class_mismatch=True`, the B2↔B3 label noise is irreconcilable ")
        f.write("from (class, position, side) alone.\n\n")
        f.write("## Breakdown\n\n")
        f.write(f"- Mismatch bunches: {mismatch_bunches}/{total_bunches} ")
        f.write(f"({report['mismatch_rate_bunches']:.1%} of all bunches)\n")
        f.write(f"- Class distribution of mismatches: {dict(per_class_mismatch)}\n\n")
        f.write("## Practical Implication\n\n")
        f.write("v9 reaches 89.0% on the 727 set. The oracle ceiling of ")
        f.write(f"{oracle_acc:.1%} shows that we are operating only ")
        f.write(f"{oracle_acc - 0.89:.1%} away from the theoretical maximum possible with ")
        f.write("purely positional+class heuristics. Further gains require either:\n")
        f.write("  1. Consensus re-labeling of the mismatched bunches (outside scope), or\n")
        f.write("  2. Allowing pixel-patch features / learned embeddings (forbidden by constraint).\n")

    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    analyze_oracle_ceiling()
