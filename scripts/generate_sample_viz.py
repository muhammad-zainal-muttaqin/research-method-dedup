"""
Generate README sample visualizations:
  - sample_4view_<TREE>.jpg  : best 4-side tree, single row
  - sample_8view_<TREE>.jpg  : best 8-side tree, 2x4 grid
Same color across panels = same physical bunch (cross-view pair).
"""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

BASE    = Path(__file__).parent.parent / "Brand-New-Dataset-YOLO"
JSON_DIR = BASE / "json"
IMG_DIR  = BASE / "images"
OUT_DIR  = BASE / "samples"
OUT_DIR.mkdir(exist_ok=True)

# ── colors ──────────────────────────────────────────────────────────────────
PALETTE = [
    "#FF4B4B","#4BFF6E","#4B8FFF","#FFD94B","#FF4BCC",
    "#4BFFF0","#FF8C4B","#A04BFF","#B8FF4B","#FF4B82",
    "#4BFFCF","#FF6B4B","#4BAAFF","#FFEE4B","#7BFF4B",
    "#4B5FFF","#FFC04B","#FF4BFF","#4BFFE0","#FF9F4B",
]
SOLO_COLOR = "#888888"

HEADER_H = 38
FOOTER_H = 54
LABEL_SZ = 15
HEAD_SZ  = 18
FOOT_SZ  = 13

def hex2rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def load_font(size, bold=False):
    for name in (("arialbd" if bold else "arial"), ("calibrib" if bold else "calibri"), "segoeui"):
        try:
            return ImageFont.truetype(f"C:/Windows/Fonts/{name}.ttf", size)
        except:
            pass
    return ImageFont.load_default()

font_lbl  = load_font(LABEL_SZ)
font_head = load_font(HEAD_SZ, bold=True)
font_foot = load_font(FOOT_SZ)

# ── scoring ──────────────────────────────────────────────────────────────────
def score(data, target_sides):
    bunches = data.get("bunches", [])
    sides   = data.get("images", {})
    if len(sides) != target_sides:
        return -1
    n_pairs   = sum(1 for b in bunches if b.get("appearance_count", 1) > 1)
    n_classes = len({b["class"] for b in bunches})
    n_unique  = len(bunches)
    lo, hi = (7, 13) if target_sides == 4 else (10, 22)
    if not (lo <= n_unique <= hi):
        return 0
    ideal = (lo + hi) // 2
    return n_pairs * 4 + n_classes * 6 + (hi - abs(n_unique - ideal))

def best_tree(target_sides):
    best_s, best_d, best_n = -1, None, None
    for f in sorted(JSON_DIR.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        s = score(d, target_sides)
        if s > best_s:
            best_s, best_d, best_n = s, d, f.stem
    return best_n, best_d

# ── render one panel (annotated image) ──────────────────────────────────────
def render_panel(side_key, side_info, box_to_bunch, bunch_color, pw, ph):
    img_path = IMG_DIR / side_info["filename"]
    if not img_path.exists():
        img = Image.new("RGB", (pw, ph), (40, 40, 40))
    else:
        img = Image.open(img_path).convert("RGB").resize((pw, ph), Image.LANCZOS)

    sx = pw / side_info.get("width",  960)
    sy = ph / side_info.get("height", 1280)
    draw = ImageDraw.Draw(img, "RGBA")
    counts = {}

    for ann in side_info.get("annotations", []):
        bx, by, bx2, by2 = ann["bbox_pixel"]
        bx,  bx2 = bx*sx,  bx2*sx
        by,  by2 = by*sy,  by2*sy
        cls      = ann["class_name"]
        bid      = box_to_bunch.get((side_key, ann["box_index"]))
        color    = bunch_color.get(bid, SOLO_COLOR)
        rgb      = hex2rgb(color)

        draw.rectangle([bx, by, bx2, by2], fill=(*rgb,35), outline=(*rgb,220), width=2)

        label = f"#{bid} {cls}" if bid else cls
        lx, ly = bx+2, max(by - LABEL_SZ - 2, 0)
        bb = draw.textbbox((lx, ly), label, font=font_lbl)
        draw.rectangle([bb[0]-1, bb[1]-1, bb[2]+2, bb[3]+2], fill=(*rgb,200))
        draw.text((lx, ly), label, fill=(255,255,255), font=font_lbl)
        counts[cls] = counts.get(cls, 0) + 1

    return img, counts

# ── compose canvas (1 row or 2x4 grid) ──────────────────────────────────────
def compose(tree_name, data, pw, ph, cols, out_path):
    bunches   = data["bunches"]
    sides_data = data["images"]
    side_keys = sorted(sides_data.keys())

    bunch_color = {
        b["bunch_id"]: (PALETTE[i % len(PALETTE)] if b.get("appearance_count",1)>1 else SOLO_COLOR)
        for i, b in enumerate(bunches)
    }
    box_to_bunch = {
        (app["side"], app["box_index"]): b["bunch_id"]
        for b in bunches for app in b.get("appearances",[])
    }

    panels = []
    for sk in side_keys:
        panel, counts = render_panel(sk, sides_data[sk], box_to_bunch, bunch_color, pw, ph)
        panels.append((sk, panel, counts))

    rows      = (len(panels) + cols - 1) // cols
    canvas_w  = pw * cols
    row_h     = ph + HEADER_H          # image + header per row
    canvas_h  = row_h * rows + FOOTER_H

    canvas = Image.new("RGB", (canvas_w, canvas_h), (18,18,18))
    dc = ImageDraw.Draw(canvas)

    for idx, (sk, panel, counts) in enumerate(panels):
        row, col = divmod(idx, cols)
        x0 = col * pw
        y0 = row * row_h

        # header
        lbl = sk.replace("_"," ").upper()
        dc.rectangle([x0, y0, x0+pw-1, y0+HEADER_H-1], fill=(35,35,35))
        tw = dc.textlength(lbl, font=font_head)
        dc.text((x0+(pw-tw)/2, y0+7), lbl, fill=(225,225,225), font=font_head)

        # panel
        canvas.paste(panel, (x0, y0+HEADER_H))

    # vertical dividers per row
    for row in range(rows):
        y_top = row * row_h
        y_bot = y_top + row_h
        for col in range(1, cols):
            x = col * pw
            dc.line([x, y_top, x, y_bot], fill=(65,65,65), width=2)

    # horizontal dividers between rows
    for row in range(1, rows):
        y = row * row_h
        dc.line([0, y, canvas_w, y], fill=(65,65,65), width=2)

    # footer: per-panel counts row (bottom strip)
    y_foot = rows * row_h
    for idx, (sk, panel, counts) in enumerate(panels):
        col = idx % cols
        row = idx // cols
        if row == rows - 1:   # only show footer counts for last row panels
            x0 = col * pw
            dc.rectangle([x0, y_foot, x0+pw-1, canvas_h-1], fill=(26,26,26))
            cs = "  ".join(f"{v}x{k}" for k,v in sorted(counts.items()))
            dc.text((x0+7, y_foot+5), f"{sum(counts.values())} bbox: {cs}",
                    fill=(160,160,160), font=font_foot)

    # summary line
    n_unique = len(bunches)
    n_pairs  = sum(1 for b in bunches if b.get("appearance_count",1)>1)
    n_sides  = len(panels)
    summary  = f"{tree_name}   |   {n_sides} views   |   {n_unique} unique bunches   |   {n_pairs} cross-view pairs   |   same color = same bunch"
    tw = dc.textlength(summary, font=font_foot)
    dc.text(((canvas_w-tw)/2, y_foot+28), summary, fill=(240,200,65), font=font_foot)

    canvas.save(out_path, quality=92)
    print(f"Saved {n_sides}-view -> {out_path}  {canvas.size}")

# ── main ─────────────────────────────────────────────────────────────────────
for target, cols, pw, ph in [(4, 4, 480, 640), (8, 4, 360, 480)]:
    name, data = best_tree(target)
    print(f"Best {target}-view tree: {name}")
    out = OUT_DIR / f"sample_{target}view_{name}.jpg"
    compose(name, data, pw, ph, cols, out)
