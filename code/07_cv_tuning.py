"""Group cross-validation, hyperparameter tuning and threshold selection (train data only).

The test set is used once at the end, to evaluate the already-tuned models.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score,
    roc_auc_score, average_precision_score, precision_recall_curve,
)
from sklearn.model_selection import (
    GroupKFold, StratifiedGroupKFold, RandomizedSearchCV, cross_val_predict, cross_validate,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
RANDOM_STATE = 42
N_ITER = 8
SEARCH_ROWS = 30000  # the hyperparameter search runs on a stratified, patient-grouped subset of train
STATE_DIR = PREDICTIONS_DIR / "_state"  # one json per finished model, makes the script resumable
KNN_SUBSAMPLE = 20000  # KNN is slow on ~700 one-hot features: tuning/CV use a random subset of train.

# Reuse the preprocessing and model definitions of 03_train_models.py (file name starts with a digit).
_spec = importlib.util.spec_from_file_location("train_models", Path(__file__).with_name("03_train_models.py"))
tm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tm)

PARAM_GRIDS = {
    "Logistic Regression": {"model__C": [0.001, 0.003, 0.01, 0.1, 1]},
    "Decision Tree": {"model__max_depth": [5, 8, 12], "model__min_samples_leaf": [10, 50, 200]},
    "Random Forest": {
        "model__n_estimators": [100, 200],
        "model__max_depth": [8, 12, 18],
        "model__min_samples_leaf": [5, 20, 50],
        "model__max_features": ["sqrt", 0.1],
    },
    "KNN": {"model__n_neighbors": [15, 51, 101], "model__weights": ["uniform", "distance"]},
    "Hist Gradient Boosting": {
        "model__learning_rate": [0.03, 0.05, 0.1, 0.2],
        "model__max_iter": [100, 200, 300],
        "model__max_leaf_nodes": [15, 31, 63],
        "model__l2_regularization": [0.0, 0.1, 1.0, 10.0],
    },
}


def make_models():
    models = tm.make_models()
    # Parallelism is done over CV folds/candidates (n_jobs=-1); avoid nested forest parallelism.
    models["Random Forest"].set_params(n_jobs=1)
    return models


def get_train_groups(y_train):
    """patient_nbr of each X_train row, saved by 02_preprocessing.py (alignment check only)."""
    groups = pd.read_csv(RESULTS_DIR / "train_groups.csv")["patient_nbr"]
    assert len(groups) == len(y_train), "train_groups.csv is not aligned with y_train.csv; rerun 02_preprocessing.py"
    return groups


def fingerprint():
    """sha256 of the code and labels a saved result depends on; a changed fingerprint invalidates it."""
    h = hashlib.sha256()
    for path in [Path(__file__).with_name("03_train_models.py"), Path(__file__),
                 RESULTS_DIR / "y_train.csv", RESULTS_DIR / "y_test.csv"]:
        h.update(path.read_bytes())
    return h.hexdigest()


def write_atomic(path, text):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def is_current(path, fp, force):
    """True if a saved state exists and was produced with the same fingerprint."""
    if force or not path.exists():
        return False
    if json.loads(path.read_text()).get("fingerprint") == fp:
        return True
    print(f"  {path.name}: saved with different code/data (fingerprint mismatch), recomputing")
    return False


def best_f1_threshold(y, proba):
    precision, recall, thresholds = precision_recall_curve(y, proba)
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(thresholds[np.argmax(f1)])


def test_metrics(y, proba, threshold):
    pred = (proba >= threshold).astype(int)
    return {
        "Accuracy": accuracy_score(y, pred),
        "Precision (class 1)": precision_score(y, pred, zero_division=0),
        "Recall (class 1)": recall_score(y, pred, zero_division=0),
        "F1-score (class 1)": f1_score(y, pred, zero_division=0),
        "Balanced accuracy": balanced_accuracy_score(y, pred),
        "ROC-AUC": roc_auc_score(y, proba),
        "PR-AUC": average_precision_score(y, proba),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="recompute everything, ignoring saved results")
    force = parser.parse_args().force
    start = time.time()
    if not (RESULTS_DIR / "train_groups.csv").exists():
        print("Run python code/02_preprocessing.py first.")
        return
    fp = fingerprint()
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = tm.load_train_test()
    groups = get_train_groups(y_train)
    cv = GroupKFold(n_splits=5)

    # Random subset used for KNN only (X_train/y_train/groups share the same positional index).
    knn_rows = np.random.RandomState(RANDOM_STATE).choice(len(X_train), KNN_SUBSAMPLE, replace=False)

    def train_view(name):
        if name == "KNN":
            return X_train.iloc[knn_rows], y_train.iloc[knn_rows], groups.iloc[knn_rows]
        return X_train, y_train, groups

    # (a) Cross-validation of the base models (skipped if already saved).
    scoring = {"F1": "f1", "ROC-AUC": "roc_auc", "PR-AUC": "average_precision",
               "Recall": "recall", "Precision": "precision"}
    cv_path = RESULTS_DIR / "cv_results.csv"
    cv_state = STATE_DIR / "cv.json"
    if force or not cv_path.exists() or not is_current(cv_state, fp, force):
        cv_rows = []
        for name, model in make_models().items():
            print(f"[CV] {name}", flush=True)
            X, y, g = train_view(name)
            scores = cross_validate(tm.build_pipeline(name, model, X), X, y, groups=g, cv=cv,
                                    scoring=scoring, n_jobs=-1)
            row = {"Model": name, "Train rows": len(X)}
            for metric in scoring:
                row[f"{metric} mean"] = scores[f"test_{metric}"].mean()
                row[f"{metric} std"] = scores[f"test_{metric}"].std()
            cv_rows.append(row)
        cv_tmp = cv_path.with_name(cv_path.name + ".tmp")
        pd.DataFrame(cv_rows).to_csv(cv_tmp, index=False)
        os.replace(cv_tmp, cv_path)
        write_atomic(cv_state, json.dumps({"fingerprint": fp}))
    else:
        print("[CV] cv_results.csv up to date, skipped")

    # (b) Randomized search, (c) out-of-fold threshold, (d) test evaluation. Train data only until (d).
    for name, model in make_models().items():
        state_path = STATE_DIR / f"{tm.safe_name(name)}.json"
        pred_path = PREDICTIONS_DIR / f"{tm.safe_name(name)}_tuned_predictions.csv"
        if pred_path.exists() and is_current(state_path, fp, force):
            print(f"[Tuning] {name} already done, skipped")
            continue
        t0 = time.time()
        print(f"[Tuning] {name}", flush=True)
        X, y, g = train_view(name)

        # Stratified, grouped subset of train for the search (first fold of a StratifiedGroupKFold).
        if len(X) > SEARCH_ROWS:
            n_splits = round(len(X) / SEARCH_ROWS)
            _, sub = next(StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
                          .split(X, y, groups=g))
        else:
            sub = np.arange(len(X))
        grid = PARAM_GRIDS[name]
        grid_size = int(np.prod([len(v) for v in grid.values()]))
        search = RandomizedSearchCV(
            tm.build_pipeline(name, model, X), grid, n_iter=min(N_ITER, grid_size), cv=cv,
            scoring="average_precision", random_state=RANDOM_STATE, n_jobs=-1, refit=False,
        )
        search.fit(X.iloc[sub], y.iloc[sub], groups=g.iloc[sub])
        params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}

        tuned = tm.build_pipeline(name, clone(model).set_params(**params), X)
        oof = cross_val_predict(tuned, X, y, groups=g, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
        threshold = best_f1_threshold(y, oof)

        # Only now the test set is used, with parameters and threshold already frozen.
        if name == "Random Forest":
            tuned["model"].set_params(n_jobs=-1)
        tuned.fit(X, y)  # refit on the whole (KNN: subsampled) train
        proba = tuned.predict_proba(X_test)[:, 1]
        pd.DataFrame({
            "y_true": y_test,
            "y_proba": proba,
            "y_pred_default": (proba >= 0.5).astype(int),
            "y_pred_optimized": (proba >= threshold).astype(int),
        }).to_csv(pred_path.with_name(pred_path.name + ".tmp"), index=False)
        os.replace(pred_path.with_name(pred_path.name + ".tmp"), pred_path)
        rows = [{"Model": name, "Train rows": len(X), "Threshold type": label, "Threshold": thr,
                 **test_metrics(y_test, proba, thr)}
                for label, thr in [("0.5", 0.5), ("optimized", threshold)]]
        write_atomic(state_path, json.dumps({"fingerprint": fp,
            "params": {**params, "search_average_precision": float(search.best_score_),
                       "search_rows": len(sub)},
            "threshold": threshold, "test_rows": rows}, default=str))
        print(f"  done in {(time.time() - t0) / 60:.1f} min", flush=True)

    states = {n: json.loads((STATE_DIR / f"{tm.safe_name(n)}.json").read_text()) for n in tm.make_models()}
    (RESULTS_DIR / "best_params.json").write_text(json.dumps({n: s["params"] for n, s in states.items()}, indent=2))
    (RESULTS_DIR / "thresholds.json").write_text(json.dumps({n: s["threshold"] for n, s in states.items()}, indent=2))
    # Base KNN (k=5, full train) at threshold 0.5, to compare with the tuned KNN trained on 20k rows.
    knn_base = pd.read_csv(PREDICTIONS_DIR / "knn_predictions.csv")
    knn_row = {"Model": "KNN (base, k=5)", "Train rows": len(X_train), "Threshold type": "0.5", "Threshold": 0.5,
               **test_metrics(knn_base["y_true"], knn_base["y_proba"], 0.5)}
    tuned_df = pd.DataFrame([r for s in states.values() for r in s["test_rows"]] + [knn_row])
    tuned_df.to_csv(RESULTS_DIR / "tuned_comparison.csv", index=False)
    print(tuned_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nTotal time this run: {(time.time() - start) / 60:.1f} min")
    print("Next, run: python code/04_evaluation.py and python code/05_visualize_results.py")


if __name__ == "__main__":
    main()
