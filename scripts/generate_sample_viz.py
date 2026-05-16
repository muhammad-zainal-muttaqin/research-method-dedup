"""Generate SawitMVC README sample visualizations."""

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parent.parent / "Brand-New-Dataset-YOLO"
JSON_DIR = BASE / "json"
IMG_DIR = BASE / "images"
OUT_DIR = BASE

PALETTE = [
    "#FF4B4B", "#4BFF6E", "#4B8FFF", "#FFD94B", "#FF4BCC",
    "#4BFFF0", "#FF8C4B", "#A04BFF", "#B8FF4B", "#FF4B82",
    "#4BFFCF", "#FF6B4B", "#4BAAFF", "#FFEE4B", "#7BFF4B",
    "#4B5FFF", "#FFC04B", "#FF4BFF", "#4BFFE0", "#FF9F4B",
]
SOLO_COLOR = "#888888"

HEADER_H = 38
FOOTER_H = 54
LABEL_SZ = 15
HEAD_SZ = 18
FOOT_SZ = 13


def hex2rgb(value):
    value = value.lstrip("#")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def load_font(size, bold=False):
    names = ("arialbd", "calibrib", "segoeuib") if bold else ("arial", "calibri", "segoeui")
    for name in names:
        try:
            return ImageFont.truetype(f"C:/Windows/Fonts/{name}.ttf", size)
        except OSError:
            pass
    return ImageFont.load_default()


FONT_LABEL = load_font(LABEL_SZ)
FONT_HEAD = load_font(HEAD_SZ, bold=True)
FONT_FOOT = load_font(FOOT_SZ)


def side_number(side_key, side_info):
    idx = side_info.get("side_index")
    if isinstance(idx, int):
        return idx + 1
    if isinstance(side_key, str) and "_" in side_key and side_key.rsplit("_", 1)[1].isdigit():
        return int(side_key.rsplit("_", 1)[1])
    return 999


def score_tree(data, target_sides):
    bunches = data.get("bunches", [])
    sides = data.get("images", {})
    if len(sides) != target_sides:
        return -1
    n_pairs = sum(1 for b in bunches if b.get("appearance_count", 1) > 1)
    n_classes = len({b.get("class") for b in bunches})
    n_unique = len(bunches)
    lo, hi = (7, 13) if target_sides == 4 else (10, 22)
    if not (lo <= n_unique <= hi):
        return 0
    ideal = (lo + hi) // 2
    return n_pairs * 4 + n_classes * 6 + (hi - abs(n_unique - ideal))


def best_tree(target_sides):
    best_score = -1
    best_name = None
    best_data = None
    for path in sorted(JSON_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        score = score_tree(data, target_sides)
        if score > best_score:
            best_score = score
            best_name = path.stem
            best_data = data
    return best_name, best_data


def render_panel(side_key, side_info, box_to_bunch, bunch_color, panel_w, panel_h):
    img_path = IMG_DIR / side_info["filename"]
    if img_path.exists():
        image = Image.open(img_path).convert("RGB").resize((panel_w, panel_h), Image.LANCZOS)
    else:
        image = Image.new("RGB", (panel_w, panel_h), (40, 40, 40))

    scale_x = panel_w / side_info.get("width", 960)
    scale_y = panel_h / side_info.get("height", 1280)
    draw = ImageDraw.Draw(image, "RGBA")
    counts = {}

    for ann in side_info.get("annotations", []):
        x1, y1, x2, y2 = ann["bbox_pixel"]
        x1, x2 = x1 * scale_x, x2 * scale_x
        y1, y2 = y1 * scale_y, y2 * scale_y
        cls = ann["class_name"]
        bunch_id = box_to_bunch.get((side_key, ann["box_index"]))
        color = bunch_color.get(bunch_id, SOLO_COLOR)
        rgb = hex2rgb(color)

        draw.rectangle([x1, y1, x2, y2], fill=(*rgb, 35), outline=(*rgb, 220), width=2)
        label = f"#{bunch_id} {cls}" if bunch_id else cls
        label_x = x1 + 2
        label_y = max(y1 - LABEL_SZ - 2, 0)
        bb = draw.textbbox((label_x, label_y), label, font=FONT_LABEL)
        draw.rectangle([bb[0] - 1, bb[1] - 1, bb[2] + 2, bb[3] + 2], fill=(*rgb, 200))
        draw.text((label_x, label_y), label, fill=(255, 255, 255), font=FONT_LABEL)
        counts[cls] = counts.get(cls, 0) + 1

    return image, counts


def compose(tree_name, data, panel_w, panel_h, cols, out_path):
    bunches = data["bunches"]
    sides_data = data["images"]
    side_items = sorted(sides_data.items(), key=lambda item: side_number(item[0], item[1]))

    bunch_color = {
        b["bunch_id"]: (PALETTE[i % len(PALETTE)] if b.get("appearance_count", 1) > 1 else SOLO_COLOR)
        for i, b in enumerate(bunches)
    }
    box_to_bunch = {
        (app["side"], app["box_index"]): b["bunch_id"]
        for b in bunches
        for app in b.get("appearances", [])
    }

    panels = []
    for side_key, side_info in side_items:
        panel, counts = render_panel(side_key, side_info, box_to_bunch, bunch_color, panel_w, panel_h)
        panels.append((side_key, side_info, panel, counts))

    rows = (len(panels) + cols - 1) // cols
    canvas_w = panel_w * cols
    row_h = panel_h + HEADER_H
    canvas_h = row_h * rows + FOOTER_H
    canvas = Image.new("RGB", (canvas_w, canvas_h), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)

    for idx, (side_key, side_info, panel, counts) in enumerate(panels):
        row, col = divmod(idx, cols)
        x0 = col * panel_w
        y0 = row * row_h
        label = f"SIDE {side_number(side_key, side_info)}"
        draw.rectangle([x0, y0, x0 + panel_w - 1, y0 + HEADER_H - 1], fill=(35, 35, 35))
        text_w = draw.textlength(label, font=FONT_HEAD)
        draw.text((x0 + (panel_w - text_w) / 2, y0 + 7), label, fill=(225, 225, 225), font=FONT_HEAD)
        canvas.paste(panel, (x0, y0 + HEADER_H))

    for row in range(rows):
        y_top = row * row_h
        y_bot = y_top + row_h
        for col in range(1, cols):
            x = col * panel_w
            draw.line([x, y_top, x, y_bot], fill=(65, 65, 65), width=2)
    for row in range(1, rows):
        y = row * row_h
        draw.line([0, y, canvas_w, y], fill=(65, 65, 65), width=2)

    y_foot = rows * row_h
    for idx, (_, _, _, counts) in enumerate(panels):
        col = idx % cols
        row = idx // cols
        if row == rows - 1:
            x0 = col * panel_w
            draw.rectangle([x0, y_foot, x0 + panel_w - 1, canvas_h - 1], fill=(26, 26, 26))
            count_text = "  ".join(f"{v}x{k}" for k, v in sorted(counts.items()))
            draw.text((x0 + 7, y_foot + 5), f"{sum(counts.values())} boxes: {count_text}",
                      fill=(160, 160, 160), font=FONT_FOOT)

    n_unique = len(bunches)
    n_pairs = sum(1 for b in bunches if b.get("appearance_count", 1) > 1)
    summary = (
        f"{tree_name}   |   {len(panels)} views   |   {n_unique} unique bunches   |   "
        f"{n_pairs} cross-view pairs   |   same color = same bunch"
    )
    text_w = draw.textlength(summary, font=FONT_FOOT)
    draw.text(((canvas_w - text_w) / 2, y_foot + 28), summary, fill=(240, 200, 65), font=FONT_FOOT)

    canvas.save(out_path, quality=92)
    print(f"Saved {out_path} {canvas.size}")


def main():
    fixed = {
        4: "DAMIMAS_A21B_0140",
        8: "DAMIMAS_A21B_0834",
    }
    for target, cols, panel_w, panel_h in [(4, 4, 480, 640), (8, 4, 360, 480)]:
        name = fixed.get(target)
        data = None
        if name and (JSON_DIR / f"{name}.json").exists():
            data = json.loads((JSON_DIR / f"{name}.json").read_text(encoding="utf-8-sig"))
        else:
            name, data = best_tree(target)
        out = OUT_DIR / f"sample_{target}view_{name}.jpg"
        compose(name, data, panel_w, panel_h, cols, out)


if __name__ == "__main__":
    main()
