"""Linear Regression counting dari GT features — baseline regressor (permintaan dosen)."""
import json, sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_counting_features import load_dataset, FEATURE_NAMES

CLASSES = ["B1", "B2", "B3", "B4"]
OUT_DIR = Path(__file__).resolve().parent.parent / "reports" / "counting_lr"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("Loading dataset...")
    X, y, tree_ids, tree_splits = load_dataset()
    splits = np.array(tree_splits)

    tr = X[splits == "train"]; yt = y[splits == "train"]
    va = X[splits == "val"];   yv = y[splits == "val"]
    te = X[splits == "test"];  yte = y[splits == "test"]
    ids_te = [tree_ids[i] for i in np.where(splits == "test")[0]]
    ids_va = [tree_ids[i] for i in np.where(splits == "val")[0]]
    print(f"Train: {len(tr)} | Val: {len(va)} | Test: {len(te)}")

    model = MultiOutputRegressor(LinearRegression())
    model.fit(tr, yt)

    def metrics(X_, y_, ids_):
        pred = np.clip(np.round(model.predict(X_)), 0, None).astype(int)
        gt   = y_.astype(int)
        m = {}
        for j, c in enumerate(CLASSES):
            err = pred[:, j] - gt[:, j]
            m[f"MAE_{c}"]    = float(np.mean(np.abs(err)))
            m[f"bias_{c}"]   = float(np.mean(err))
            m[f"acc_pm1_{c}"] = float(np.mean(np.abs(err) <= 1))
        m["macro_class_mae"]  = float(np.mean([m[f"MAE_{c}"] for c in CLASSES]))
        m["macro_acc_pm1"]    = float(np.mean([m[f"acc_pm1_{c}"] for c in CLASSES]))
        total_err = pred.sum(1) - gt.sum(1)
        m["total_count_mae"]  = float(np.mean(np.abs(total_err)))
        m["total_pm1_acc"]    = float(np.mean(np.abs(total_err) <= 1))
        m["exact_profile_acc"] = float(np.mean(np.all(pred == gt, axis=1)))
        rows = [{"tree_id": tid, **{f"pred_{c}": pred[i,j] for j,c in enumerate(CLASSES)},
                 **{f"gt_{c}": gt[i,j] for j,c in enumerate(CLASSES)}}
                for i, tid in enumerate(ids_)]
        return m, pd.DataFrame(rows)

    mt, dt = metrics(te, yte, ids_te)
    mv, _  = metrics(va, yv,  ids_va)
    mt["split"] = "test"; mv["split"] = "val"

    (OUT_DIR / "metrics.json").write_text(json.dumps({"test": mt, "val": mv}, indent=2))
    dt.to_csv(OUT_DIR / "predictions.csv", index=False)

    print(f"\n=== TEST METRICS ===")
    for c in CLASSES:
        print(f"  MAE_{c}: {mt[f'MAE_{c}']:.4f}  acc_pm1: {mt[f'acc_pm1_{c}']:.4f}")
    print(f"  macro_class_mae: {mt['macro_class_mae']:.4f}")
    print(f"  macro_acc_pm1:   {mt['macro_acc_pm1']:.4f} ({mt['macro_acc_pm1']*100:.1f}%)")
    print(f"  exact_profile_acc: {mt['exact_profile_acc']:.4f}")
    print(f"  total_count_mae: {mt['total_count_mae']:.4f}")
    print(f"Saved to {OUT_DIR}")

if __name__ == "__main__":
    main()
