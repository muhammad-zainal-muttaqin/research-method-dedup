"""
E2E per-image evaluation (permintaan dosen) — evaluasi per foto, bukan per pohon.

YOLO deteksi per gambar → hitung deteksi langsung → bandingkan GT per gambar
(GT per gambar = annotations di JSON per sisi)

Usage:
  python scripts/run_e2e_per_image.py --name y26n_vanilla_local \
      --weights ml-track/baseline-run/weights/y26n_vanilla_local.pt
"""
import json, argparse, csv
import numpy as np
from pathlib import Path
from collections import defaultdict

REPO     = Path(__file__).resolve().parent.parent
IMG_DIR  = REPO / "Brand-New-Dataset-YOLO" / "images"
JSON_DIR = REPO / "Brand-New-Dataset-YOLO" / "json"
CLASSES  = ["B1", "B2", "B3", "B4"]
CLASS_MAP = {0: "B1", 1: "B2", 2: "B3", 3: "B4"}


def load_gt_per_image():
    """GT per gambar: {filename: {B1:int, B2:int, B3:int, B4:int, split:str}}"""
    gt = {}
    for jp in JSON_DIR.glob("*.json"):
        d = json.loads(jp.read_text(encoding="utf-8-sig"))
        tree_name = d.get("tree_name") or d.get("tree_id") or jp.stem
        split = d.get("split", "unknown")
        for side_key, side_data in d["images"].items():
            # derive filename: tree_name + side number
            side_num = side_key.replace("sisi_", "")
            fname = f"{tree_name}_{int(side_num):01d}.jpg"
            counts = {c: 0 for c in CLASSES}
            for ann in side_data.get("annotations", []):
                cn = ann.get("class_name") or ann.get("class")
                if cn in counts:
                    counts[cn] += 1
            gt[fname] = {**counts, "split": split, "tree": tree_name}
    return gt


def run_inference(name, weights_path):
    from ultralytics import YOLO
    out_dir = REPO / "ml-track" / "predictions" / f"{name}_inference"
    if not list(out_dir.glob("*.json")):
        print(f"[ERROR] Inference JSONs not found in {out_dir}. Run E2E pipeline first.")
        return None
    # Load per-image predictions from inference JSONs
    preds = {}
    for fp in out_dir.glob("*.json"):
        d = json.loads(fp.read_text())
        tree_name = d.get("tree_name") or fp.stem
        split = d.get("split", "unknown")
        for side_key, side_data in d["images"].items():
            side_num = side_key.replace("sisi_", "")
            fname = f"{tree_name}_{int(side_num):01d}.jpg"
            counts = {c: 0 for c in CLASSES}
            for ann in side_data.get("annotations", []):
                cn = ann.get("class_name") or ann.get("class")
                if cn in counts:
                    counts[cn] += 1
            preds[fname] = {**counts, "split": split}
    return preds


def compute_metrics(rows, split_filter=None):
    if split_filter:
        rows = [r for r in rows if r["split"] == split_filter]
    if not rows:
        return {}
    m = {}
    for c in CLASSES:
        errs = [abs(r[f"pred_{c}"] - r[f"gt_{c}"]) for r in rows]
        m[f"MAE_{c}"]     = float(np.mean(errs))
        m[f"acc_pm1_{c}"] = float(np.mean([e <= 1 for e in errs]))
        m[f"bias_{c}"]    = float(np.mean([r[f"pred_{c}"] - r[f"gt_{c}"] for r in rows]))
    m["macro_class_mae"] = float(np.mean([m[f"MAE_{c}"] for c in CLASSES]))
    m["macro_acc_pm1"]   = float(np.mean([m[f"acc_pm1_{c}"] for c in CLASSES]))
    total_errs = [abs(sum(r[f"pred_{c}"] for c in CLASSES) - sum(r[f"gt_{c}"] for c in CLASSES)) for r in rows]
    m["total_count_mae"] = float(np.mean(total_errs))
    m["total_pm1_acc"]   = float(np.mean([e <= 1 for e in total_errs]))
    m["n_images"] = len(rows)
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--weights", required=True)
    args = parser.parse_args()

    print(f"\nE2E Per-Image Evaluation: {args.name}")
    gt = load_gt_per_image()
    preds = run_inference(args.name, args.weights)
    if preds is None:
        return

    rows = []
    for fname, pred in preds.items():
        if fname not in gt:
            continue
        row = {"image": fname, "split": gt[fname]["split"], "tree": gt[fname]["tree"]}
        for c in CLASSES:
            row[f"pred_{c}"] = pred.get(c, 0)
            row[f"gt_{c}"]   = gt[fname].get(c, 0)
        rows.append(row)

    print(f"  Matched images: {len(rows)}")

    out_dir = REPO / "reports" / f"e2e_per_image_{args.name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    mt = compute_metrics(rows, "test")
    mv = compute_metrics(rows, "val")
    mt["split"] = "test"; mv["split"] = "val"

    (out_dir / "metrics.json").write_text(json.dumps({"test": mt, "val": mv}, indent=2))
    with open(out_dir / "predictions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print(f"  [test] Macro Acc±1={mt['macro_acc_pm1']*100:.1f}%  MAE={mt['macro_class_mae']:.3f}  n={mt['n_images']}")
    print(f"  Saved → {out_dir}")


if __name__ == "__main__":
    main()
