"""
Experiment #8 — End-to-end y26s → SVM counting.
Requires: predictions/y26s_inference/*.json (run run_e2e_inference.py first)
Output: reports/e2e_svm/
"""
import os, sys, json, glob
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from build_counting_features import extract_features_from_json, load_split_manifest, FEATURE_NAMES, CLASSES

from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INFER_DIR = os.path.join(REPO, "predictions", "y26s_inference")
GT_JSON_DIR = os.path.join(REPO, "Tested-Brand-New-Dataset-YOLO", "json")
MANIFEST = os.path.join(REPO, "Tested-Brand-New-Dataset-YOLO", "split_manifest.csv")
REPORT_DIR = os.path.join(REPO, "reports", "e2e_svm")
os.makedirs(REPORT_DIR, exist_ok=True)


def compute_metrics(y_true, y_pred, tree_ids):
    y_pred_r = np.clip(np.round(y_pred), 0, None).astype(int)
    y_true_i = y_true.astype(int)
    rows = []
    for i, tid in enumerate(tree_ids):
        rows.append(dict(
            tree_id=tid,
            **{f"pred_{c}": y_pred_r[i, j] for j, c in enumerate(CLASSES)},
            **{f"gt_{c}": y_true_i[i, j] for j, c in enumerate(CLASSES)},
        ))
    df = pd.DataFrame(rows)
    metrics = {}
    for j, c in enumerate(CLASSES):
        err = y_pred_r[:, j] - y_true_i[:, j]
        metrics[f"MAE_{c}"] = float(np.mean(np.abs(err)))
        metrics[f"bias_{c}"] = float(np.mean(err))
        metrics[f"acc_pm1_{c}"] = float(np.mean(np.abs(err) <= 1))
    metrics["macro_class_mae"] = float(np.mean([metrics[f"MAE_{c}"] for c in CLASSES]))
    exact = np.all(y_pred_r == y_true_i, axis=1)
    metrics["exact_profile_acc"] = float(np.mean(exact))
    total_pred = y_pred_r.sum(axis=1)
    total_gt = y_true_i.sum(axis=1)
    total_err = total_pred - total_gt
    metrics["total_count_mae"] = float(np.mean(np.abs(total_err)))
    metrics["total_pm1_acc"] = float(np.mean(np.abs(total_err) <= 1))
    return metrics, df


def load_e2e_dataset():
    splits = load_split_manifest(MANIFEST)
    # Load GT labels from GT JSON
    gt_labels = {}
    for fp in glob.glob(os.path.join(GT_JSON_DIR, "*.json")):
        with open(fp, encoding="utf-8-sig") as f:
            d = json.load(f)
        tree_name = d.get("tree_name") or d.get("tree_id") or os.path.splitext(os.path.basename(fp))[0]
        gt_labels[tree_name] = np.array([d["summary"]["by_class"].get(c, 0) for c in CLASSES], dtype=np.float32)

    X, y, tree_ids, tree_splits = [], [], [], []
    for fp in sorted(glob.glob(os.path.join(INFER_DIR, "*.json"))):
        with open(fp) as f:
            data = json.load(f)
        tree_name = data.get("tree_name") or os.path.splitext(os.path.basename(fp))[0]
        if tree_name not in gt_labels:
            continue
        feats = extract_features_from_json(data)
        split = splits.get(tree_name, data.get("split", "train"))
        X.append(feats)
        y.append(gt_labels[tree_name])
        tree_ids.append(tree_name)
        tree_splits.append(split)
    return np.array(X), np.array(y), tree_ids, np.array(tree_splits)


def main():
    infer_count = len(glob.glob(os.path.join(INFER_DIR, "*.json")))
    print(f"Found {infer_count} inference JSONs in {INFER_DIR}")
    if infer_count == 0:
        print("ERROR: Run run_e2e_inference.py first!")
        sys.exit(1)

    print("Loading e2e dataset from detector features...")
    X, y, tree_ids, splits = load_e2e_dataset()
    train_mask = splits == "train"
    val_mask = splits == "val"
    test_mask = splits == "test"
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    ids_test = [tree_ids[i] for i in np.where(test_mask)[0]]
    ids_val = [tree_ids[i] for i in np.where(val_mask)[0]]
    print(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", MultiOutputRegressor(SVR(kernel="rbf"))),
    ])
    grid = {
        "svr__estimator__C": [0.1, 1, 10],
        "svr__estimator__gamma": ["scale", 0.01, 0.1],
    }
    print("Running GridSearchCV...")
    gs = GridSearchCV(pipe, grid, cv=3, scoring="neg_mean_absolute_error", n_jobs=-1, verbose=1)
    gs.fit(X_train, y_train)
    print(f"Best params: {gs.best_params_} | CV MAE: {-gs.best_score_:.4f}")

    y_pred_test = gs.predict(X_test)
    metrics_test, df_test = compute_metrics(y_test, y_pred_test, ids_test)
    metrics_test["best_params"] = str(gs.best_params_)
    metrics_test["cv_mae"] = float(-gs.best_score_)
    metrics_test["split"] = "test"

    y_pred_val = gs.predict(X_val)
    metrics_val, _ = compute_metrics(y_val, y_pred_val, ids_val)
    metrics_val["split"] = "val"

    print("\n=== TEST METRICS (E2E SVM) ===")
    for k, v in metrics_test.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")

    with open(os.path.join(REPORT_DIR, "metrics.json"), "w") as f:
        json.dump({"test": metrics_test, "val": metrics_val}, f, indent=2)
    df_test.to_csv(os.path.join(REPORT_DIR, "predictions.csv"), index=False)
    per_class = pd.DataFrame([
        {"class": c, "MAE": metrics_test[f"MAE_{c}"], "bias": metrics_test[f"bias_{c}"], "acc_pm1": metrics_test[f"acc_pm1_{c}"]}
        for c in CLASSES
    ])
    per_class.to_csv(os.path.join(REPORT_DIR, "per_class_mae.csv"), index=False)
    print(f"\nOutputs saved to {REPORT_DIR}")


if __name__ == "__main__":
    main()
