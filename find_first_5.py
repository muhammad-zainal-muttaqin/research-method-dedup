import os
from pathlib import Path
from collections import defaultdict

def find_first_5_per_sisi():
    base_dir = Path("dataset")
    images_dir = base_dir / "images"
    labels_dir = base_dir / "labels"
    
    # Group images by sisi
    sisi_groups = defaultdict(list)
    
    for split_dir in images_dir.iterdir():
        if not split_dir.is_dir():
            continue
        split = split_dir.name
        for img_path in sorted(split_dir.glob("*.jpg")):
            stem = img_path.stem  # e.g., DAMIMAS_A21B_0005_1
            parts = stem.rsplit("_", 1)
            if len(parts) != 2:
                continue
            _, sisi = parts
            
            # Check corresponding txt in labels/{split}
            txt_path = labels_dir / split / f"{stem}.txt"
            if txt_path.exists() and txt_path.stat().st_size > 0:
                sisi_groups[sisi].append({
                    "image": str(img_path),
                    "txt": str(txt_path),
                    "split": split,
                    "stem": stem
                })
    
    # Sort each group and take first 5
    output_lines = []
    for sisi in sorted(sisi_groups.keys(), key=lambda x: int(x) if x.isdigit() else x):
        group = sisi_groups[sisi]
        # Already sorted by image path
        first_5 = group[:5]
        output_lines.append(f"=== Sisi {sisi} ({len(first_5)} files) ===")
        for item in first_5:
            output_lines.append(f"  IMAGE: {item['image']}")
            output_lines.append(f"  TXT  : {item['txt']}")
            output_lines.append(f"  SPLIT: {item['split']}")
            output_lines.append("")
        output_lines.append("")
    
    output_path = Path("find_first_5.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
    
    print(f"Saved to {output_path.resolve()}")
    print("Summary:")
    for sisi in sorted(sisi_groups.keys(), key=lambda x: int(x) if x.isdigit() else x):
        total = len(sisi_groups[sisi])
        shown = min(5, total)
        print(f"  Sisi {sisi}: {shown}/{total} files written")

if __name__ == "__main__":
    find_first_5_per_sisi()
