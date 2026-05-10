"""
M18_entropy_stack
Family: entropy  | Old name: v8_entropy_stacking
Benchmark 953-tree: experimental.

Combines entropy modulation with stacking density correction.
Entropy modulates the global duplication rate; stacking corrects
within-class vertical density.

Implementation lives in scripts/dedup_research_v8.py — this module
is a thin adapter that exposes a uniform algorithms-package API.
"""

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from dedup_research_v8 import v8_entropy_stacking as _impl  # noqa: E402


def predict(detections: list) -> dict:
    return _impl(detections)
