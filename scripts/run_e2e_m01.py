"""
Experiment E2E-M01 — y26s → M01_selector_b2b3 heuristic counting.
Requires: ml-track/predictions/y26s_inference/*.json (run run_e2e_inference.py first)
Output: reports/e2e_m01/
"""
import os, sys, json, glob, csv
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from algorithms.M01_selector_b2b3 import predict as m01_predict

INFER_DIR = os.path.join(REPO, "ml-track", "predictions", "y26s_inference")
GT_JSON_DIR = os.path.join(REPO, "Brand-New-Dataset-YOLO", "json")
MANIFEST = os.path.join(REPO, "Brand-New-Dataset-YOLO", "split_manifest.csv")
REPORT_DIR = os.path.join(REPO, "reports", "e2e_m01")
os.makedirs(REPORT_DIR, exist_ok=True)

CLASSES = ["B1", "B2", "B3", "B4"]


def load_split_manifest():
    splits = {}
    with open(MANIFEST, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            splits[row["tree_id"]] = row["new_split"]
    return splits


def inference_json_to_dets(data: dict) -> list:
    dets = []
    for side_label, sd in data["images"].items():
        try:
            si = int(side_label.replace("sisi_", "")) - 1
        except ValueError:
            si = 0
        for ann in sd.get("annotations", []):
            cx, cy, w, h = ann["bbox_yolo"]
            dets.append({
                "class": ann["class_name"],
                "x_norm": cx,
                "y_norm": cy,
                "side_index": si,
                "area_norm": w * h,
                "aspect_ratio": w / h if h > 0 else 1.0,
                "side": side_label,
            })
    return dets


def load_gt(tree_name: str) -> dict:
    gt_path = os.path.join(GT_JSON_DIR, f"{tree_name}.json")
    if not os.path.exists(gt_path):
        return None
    data = json.loads(open(gt_path, encoding="utf-8-sig").read())
    return {c: data["summary"]["by_class"].get(c, 0) for c in CLASSES}


def compute_metrics(rows, split_label):
    filtered = [r for r in rows if r["split"] == split_label]
    if not filtered:
        return {}
    metrics = {}
    for c in CLASSES:
        errs = [abs(r[f"pred_{c}"] - r[f"gt_{c}"]) for r in filtered]
        biases = [r[f"pred_{c}"] - r[f"gt_{c}"] for r in filtered]
        within1 = [abs(e) <= 1 for e in errs]
        metrics[f"MAE_{c}"] = float(np.mean(errs))
        metrics[f"bias_{c}"] = float(np.mean(biases))
        metrics[f"acc_pm1_{c}"] = float(np.mean(within1))
    metrics["macro_class_mae"] = float(np.mean([metrics[f"MAE_{c}"] for c in CLASSES]))
    metrics["macro_acc_pm1"] = float(np.mean([metrics[f"acc_pm1_{c}"] for c in CLASSES]))
    total_errs = [abs(sum(r[f"pred_{c}"] for c in CLASSES) - sum(r[f"gt_{c}"] for c in CLASSES))
                  for r in filtered]
    metrics["total_count_mae"] = float(np.mean(total_errs))
    metrics["total_pm1_acc"] = float(np.mean([e <= 1 for e in total_errs]))
    exact = [all(abs(r[f"pred_{c}"] - r[f"gt_{c}"]) == 0 for c in CLASSES) for r in filtered]
    metrics["exact_profile_acc"] = float(np.mean(exact))
    metrics["n_trees"] = len(filtered)
    metrics["split"] = split_label
    return metrics


def main():
    splits = load_split_manifest()
    infer_files = sorted(glob.glob(os.path.join(INFER_DIR, "*.json")))
    print(f"Found {len(infer_files)} inference files.")

    rows = []
    missing_gt = 0
    for fp in infer_files:
        tree_name = os.path.splitext(os.path.basename(fp))[0]
        data = json.loads(open(fp, encoding="utf-8").read())
        dets = inference_json_to_dets(data)
        pred = m01_predict(dets)
        gt = load_gt(tree_name)
        if gt is None:
            missing_gt += 1
            continue
        split = splits.get(tree_name, data.get("split", "unknown"))
        row = {"tree_name": tree_name, "split": split}
        for c in CLASSES:
            row[f"pred_{c}"] = pred.get(c, 0)
            row[f"gt_{c}"] = gt.get(c, 0)
        rows.append(row)

    if missing_gt:
        print(f"Warning: {missing_gt} trees missing GT, skipped.")

    # Save predictions CSV
    pred_path = os.path.join(REPORT_DIR, "predictions.csv")
    with open(pred_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {len(rows)} predictions → {pred_path}")

    # Compute and save metrics
    test_metrics = compute_metrics(rows, "test")
    val_metrics = compute_metrics(rows, "val")
    all_metrics = compute_metrics(rows, None) if False else {}

    # Also compute overall
    overall = compute_metrics([{**r, "split": "all"} for r in rows], "all")

    out = {}
    if test_metrics:
        out["test"] = test_metrics
    if val_metrics:
        out["val"] = val_metrics
    out["overall"] = overall

    metrics_path = os.path.join(REPORT_DIR, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved metrics → {metrics_path}")

    print("\n=== RESULTS ===")
    for split_k, m in out.items():
        if not m:
            continue
        print(f"\n[{split_k}] n={m.get('n_trees',0)}")
        for c in CLASSES:
            print(f"  {c}: MAE={m.get(f'MAE_{c}',0):.3f}  Acc±1={m.get(f'acc_pm1_{c}',0)*100:.1f}%")
        print(f"  Macro MAE={m.get('macro_class_mae',0):.3f}  Macro Acc±1={m.get('macro_acc_pm1',0)*100:.1f}%")
        print(f"  Exact={m.get('exact_profile_acc',0)*100:.1f}%  Total MAE={m.get('total_count_mae',0):.3f}")


if __name__ == "__main__":
    main()
