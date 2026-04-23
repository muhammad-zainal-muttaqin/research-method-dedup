"""Pure-geometry cross-view deduplication linker for multi-view oil-palm bunch counting.

Input per tree: GT annotations (class + bbox_yolo) untuk setiap sisi.
Output: unique bunch count per class {B1, B2, B3, B4}.

Algoritma (no ML, no embedding):
  1. Intra-view IoU dedup (safety).
  2. Untuk setiap kelas, Hungarian matching pada setiap pasangan sisi dengan cost
     = |Δcy| + λ_s · |log(area ratio)|.  Adjacent pair biaya bonus via λ_adj.
  3. Edge valid jika cost ≤ T_cost dan |Δcy| ≤ T_y dan area_ratio ≤ T_s.
  4. Union-find semua edge → komponen.
  5. Enforce "≤1 annotation per sisi per komponen": split edge cost tertinggi sampai valid.
  6. Count = jumlah komponen per kelas.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np
NAMES = ("B1", "B2", "B3", "B4")


@dataclass
class LinkerConfig:
    T_y: float = 0.12          # max |Δcy_center| normalized
    T_s: float = 3.0           # max area ratio (>= 1)
    lam_s: float = 0.2         # bobot |log(area ratio)| dalam cost
    lam_adj: float = 0.0       # penalti tambahan untuk pasangan non-adjacent (0 = netral)
    iou_intra: float = 0.5     # threshold IoU intra-view dedup
    T_cost: float = 0.15       # upper bound cost agar edge di-accept
    adjacent_only: bool = False  # jika True, skip pasangan opposite-side sama sekali
    T_y_opp: float | None = None # threshold Δy untuk opposite pair (None = pakai T_y)
    mutual_best: bool = False  # filter: i↔j hanya kalau j adalah min-cost neighbor untuk i DAN i min-cost untuk j
    per_class_T_y: dict[str, float] = field(default_factory=dict)  # override T_y per kelas (mis. {'B3': 0.04})

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class _Ann:
    side: str
    idx: int
    cls: str
    cx: float
    cy: float
    w: float
    h: float

    @property
    def area(self) -> float:
        return max(self.w * self.h, 1e-9)


def _iou_yolo(a: _Ann, b: _Ann) -> float:
    ax1, ay1 = a.cx - a.w / 2, a.cy - a.h / 2
    ax2, ay2 = a.cx + a.w / 2, a.cy + a.h / 2
    bx1, by1 = b.cx - b.w / 2, b.cy - b.h / 2
    bx2, by2 = b.cx + b.w / 2, b.cy + b.h / 2
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    return inter / max(a.area + b.area - inter, 1e-9)


def _intra_dedup(anns_by_side: dict[str, list[_Ann]], iou_thr: float) -> dict[str, list[_Ann]]:
    cleaned: dict[str, list[_Ann]] = {}
    for side, anns in anns_by_side.items():
        keep: list[_Ann] = []
        for a in sorted(anns, key=lambda x: -x.area):
            if any(a.cls == k.cls and _iou_yolo(a, k) > iou_thr for k in keep):
                continue
            keep.append(a)
        cleaned[side] = keep
    return cleaned


def _side_index(side: str) -> int:
    return int(side.split("_")[-1]) - 1


def _is_adjacent(sa: str, sb: str, n_sides: int) -> bool:
    ia, ib = _side_index(sa), _side_index(sb)
    diff = abs(ia - ib) % n_sides
    return diff == 1 or diff == n_sides - 1


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


class GeoLinker:
    def __init__(self, cfg: LinkerConfig | None = None):
        self.cfg = cfg or LinkerConfig()

    # ------------------------------------------------------------------ public

    def count(self, tree: dict[str, Any]) -> dict[str, int]:
        """tree = {'images': {'sisi_k': {'annotations': [...]}}}

        Accept both JSON schema (class_name + bbox_yolo) and pseudo-dict.
        """
        anns_by_side: dict[str, list[_Ann]] = {}
        for side, sd in tree["images"].items():
            lst: list[_Ann] = []
            for i, ann in enumerate(sd.get("annotations", [])):
                cls = ann.get("class_name")
                if cls not in NAMES:
                    continue
                bb = ann["bbox_yolo"]
                lst.append(_Ann(side=side, idx=i, cls=cls,
                                cx=float(bb[0]), cy=float(bb[1]),
                                w=float(bb[2]), h=float(bb[3])))
            anns_by_side[side] = lst

        anns_by_side = _intra_dedup(anns_by_side, self.cfg.iou_intra)
        n_sides = len(anns_by_side)
        sides = sorted(anns_by_side.keys(), key=_side_index)

        counts = {c: 0 for c in NAMES}
        for cls in NAMES:
            # Flatten annotations of this class with stable indexing
            flat: list[_Ann] = []
            side_of: list[str] = []
            for s in sides:
                for a in anns_by_side[s]:
                    if a.cls == cls:
                        flat.append(a)
                        side_of.append(s)
            n = len(flat)
            if n == 0:
                continue

            edges = self._candidate_edges(flat, side_of, sides, n_sides, cls)
            clusters = self._cluster_with_constraint(n, edges, side_of)
            counts[cls] = len(clusters)

        return counts

    # ------------------------------------------------------------------ internal

    def _candidate_edges(
        self,
        flat: list[_Ann],
        side_of: list[str],
        sides: list[str],
        n_sides: int,
        cls: str = "",
    ) -> list[tuple[int, int, float]]:
        """Kumpulkan semua edge (i, j) beda-sisi yang memenuhi gate Δy, T_s, T_cost."""
        cfg = self.cfg
        T_y_base = cfg.per_class_T_y.get(cls, cfg.T_y)
        edges: list[tuple[int, int, float]] = []
        n = len(flat)
        best_for: dict[int, tuple[int, float]] = {}  # untuk mutual-best
        for i in range(n):
            for j in range(i + 1, n):
                sa, sb = side_of[i], side_of[j]
                if sa == sb:
                    continue
                is_adj = _is_adjacent(sa, sb, n_sides)
                if not is_adj and cfg.adjacent_only:
                    continue
                adj_pen = 0.0 if is_adj else cfg.lam_adj
                T_y_pair = T_y_base if is_adj or cfg.T_y_opp is None else cfg.T_y_opp
                ai, aj = flat[i], flat[j]
                dy = abs(ai.cy - aj.cy)
                ratio = max(ai.area, aj.area) / min(ai.area, aj.area)
                if dy > T_y_pair or ratio > cfg.T_s:
                    continue
                c = dy + cfg.lam_s * abs(np.log(ratio)) + adj_pen
                if c > cfg.T_cost:
                    continue
                edges.append((i, j, float(c)))
                if cfg.mutual_best:
                    if i not in best_for or c < best_for[i][1]:
                        best_for[i] = (j, c)
                    if j not in best_for or c < best_for[j][1]:
                        best_for[j] = (i, c)
        if cfg.mutual_best and edges:
            edges = [(i, j, c) for (i, j, c) in edges
                     if best_for.get(i, (None, None))[0] == j
                     and best_for.get(j, (None, None))[0] == i]
        return edges

    def _cluster_with_constraint(
        self,
        n: int,
        edges: list[tuple[int, int, float]],
        side_of: list[str],
    ) -> list[list[int]]:
        """Union-find lalu split komponen yang melanggar one-per-side."""
        edges_sorted = sorted(edges, key=lambda e: e[2])   # ascending cost
        uf = UnionFind(n)
        accepted: list[tuple[int, int, float]] = []
        # Build component → set of sides incrementally, reject edge yang membuat konflik
        comp_sides: dict[int, set[str]] = {i: {side_of[i]} for i in range(n)}
        comp_members: dict[int, list[int]] = {i: [i] for i in range(n)}
        for a, b, c in edges_sorted:
            ra, rb = uf.find(a), uf.find(b)
            if ra == rb:
                continue
            sides_a, sides_b = comp_sides[ra], comp_sides[rb]
            if sides_a & sides_b:
                continue  # would violate one-per-side
            uf.union(ra, rb)
            new_root = uf.find(a)
            old_root = rb if new_root == ra else ra
            comp_sides[new_root] = sides_a | sides_b
            comp_members[new_root] = comp_members[ra] + comp_members[rb]
            if old_root in comp_sides and old_root != new_root:
                del comp_sides[old_root]; del comp_members[old_root]
            accepted.append((a, b, c))

        clusters: dict[int, list[int]] = {}
        for i in range(n):
            r = uf.find(i)
            clusters.setdefault(r, []).append(i)
        return list(clusters.values())


# ---------------------------------------------------------------------- helpers


def txt_to_tree_dict(tree_id: str, label_files: dict[str, Any]) -> dict[str, Any]:
    """Adapter: YOLO TXT (per sisi) → tree dict dengan schema `images[sisi_k].annotations`.

    label_files: {sisi_k: Path_to_txt}. class_id → class_name via CLASS_MAP.
    """
    CLASS_MAP = {0: "B1", 1: "B2", 2: "B3", 3: "B4"}
    images: dict[str, dict[str, Any]] = {}
    for side, fp in label_files.items():
        anns: list[dict[str, Any]] = []
        text = fp.read_text(encoding="utf-8") if hasattr(fp, "read_text") else open(fp).read()
        for line in text.splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cls = CLASS_MAP.get(cls_id)
            if cls is None:
                continue
            cx, cy, w, h = (float(x) for x in parts[1:5])
            anns.append({"class_name": cls, "bbox_yolo": [cx, cy, w, h]})
        images[side] = {"annotations": anns}
    return {"tree_id": tree_id, "images": images}
