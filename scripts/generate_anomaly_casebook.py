from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "Brand-New-Dataset-YOLO"
EDA = ROOT / "EDA_report"
OUT_MD = EDA / "ANOMALY_CASEBOOK.md"

CASES_CSV = EDA / "tables" / "appearance_gt_tree_sides_cases.csv"


def parse_bbox_id(v: str) -> int | None:
    if not isinstance(v, str):
        return None
    if not v.startswith("b"):
        return None
    s = v[1:]
    return int(s) if s.isdigit() else None


def node_key(side_index: int, box_index: int) -> tuple[int, int]:
    return side_index, box_index


def main() -> None:
    if not CASES_CSV.exists():
        raise FileNotFoundError(f"Missing cases file: {CASES_CSV}")

    cases = pd.read_csv(CASES_CSV, encoding="utf-8")
    if len(cases) == 0:
        OUT_MD.write_text("# Anomaly Casebook\n\nNo cases found.\n", encoding="utf-8")
        print(f"Wrote {OUT_MD}")
        return

    lines: list[str] = []
    lines.append("# Anomaly Casebook")
    lines.append("")
    lines.append("Source: `EDA_report/tables/appearance_gt_tree_sides_cases.csv`")
    lines.append("")
    lines.append(
        "Cases where `appearance_count > tree_n_sides` with side-level evidence and `_confirmedLinks` edges touching the bunch."
    )
    lines.append("")
    lines.append(f"Total cases: **{len(cases)}**")
    lines.append("")

    for _, row in cases.sort_values(["tree_id", "bunch_id"]).iterrows():
        tree_id = row["tree_id"]
        bunch_id = int(row["bunch_id"])
        tree_n_sides = int(row["tree_n_sides"])
        app_count = int(row["appearance_count"])
        unique_side_count = int(row["unique_side_count"])
        klass = row["class"]

        json_path = DATASET / "json" / f"{tree_id}.json"
        if not json_path.exists():
            continue
        with json_path.open("r", encoding="utf-8-sig") as f:
            rec = json.load(f)

        bunch_obj = None
        for b in rec.get("bunches", []):
            if int(b.get("bunch_id", -1)) == bunch_id:
                bunch_obj = b
                break
        if bunch_obj is None:
            continue

        appearances = bunch_obj.get("appearances", [])
        by_side = defaultdict(list)
        bunch_nodes = set()
        for a in appearances:
            si = int(a.get("side_index", -1))
            bi = int(a.get("box_index", -1))
            by_side[si].append(bi)
            bunch_nodes.add(node_key(si, bi))

        side_duplicates = {
            si: sorted(v) for si, v in by_side.items() if len(v) > 1
        }

        touching = []
        for lk in rec.get("_confirmedLinks", []):
            sa = int(lk.get("sideA", -1))
            ba = parse_bbox_id(lk.get("bboxIdA", ""))
            sb = int(lk.get("sideB", -1))
            bb = parse_bbox_id(lk.get("bboxIdB", ""))
            if ba is None or bb is None:
                continue
            na = node_key(sa, ba)
            nb = node_key(sb, bb)
            if na in bunch_nodes or nb in bunch_nodes:
                touching.append(
                    {
                        "link_id": lk.get("linkId", ""),
                        "sideA": sa,
                        "bboxA": ba,
                        "sideB": sb,
                        "bboxB": bb,
                        "both_in_bunch": int(na in bunch_nodes and nb in bunch_nodes),
                    }
                )

        lines.append(f"## {tree_id} / bunch_id={bunch_id}")
        lines.append("")
        lines.append(f"- class: `{klass}`")
        lines.append(f"- tree_n_sides: `{tree_n_sides}`")
        lines.append(f"- appearance_count: `{app_count}`")
        lines.append(f"- unique_side_count: `{unique_side_count}`")
        lines.append(f"- same_side_duplicates: `{app_count - unique_side_count}`")
        lines.append("")
        lines.append("Appearances:")
        for a in sorted(appearances, key=lambda x: (int(x.get("side_index", 999)), int(x.get("box_index", 999)))):
            lines.append(
                f"- side `{a.get('side')}` (`{a.get('side_index')}`) / box_index `{a.get('box_index')}` / class `{a.get('class_name')}`"
            )
        lines.append("")
        if side_duplicates:
            lines.append("Duplicated side slots:")
            for si, bix in sorted(side_duplicates.items()):
                lines.append(f"- side_index `{si}` has multiple boxes: `{bix}`")
            lines.append("")

        if touching:
            lines.append("Touching `_confirmedLinks`:")
            for lk in touching:
                lines.append(
                    f"- `{lk['link_id']}`: side `{lk['sideA']}`/b`{lk['bboxA']}` <-> side `{lk['sideB']}`/b`{lk['bboxB']}` (both_in_bunch={lk['both_in_bunch']})"
                )
            lines.append("")

        lines.append("Interpretation:")
        lines.append(
            "- This case exceeds tree-side limit, so count inflation is caused by multi-box appearances within one or more sides, not extra camera sides."
        )
        lines.append("")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
