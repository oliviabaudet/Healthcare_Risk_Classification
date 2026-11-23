
"""Create comparison charts, ROC/PR curves, CV comparison and feature importance plots."""
import importlib.util
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
FIGURES_DIR = RESULTS_DIR / "figures"

# Reuse model and preprocessing definitions of 03_train_models.py (file name starts with a digit).
_spec = importlib.util.spec_from_file_location("train_models", Path(__file__).with_name("03_train_models.py"))
tm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tm)


def plot_curves():
    """Overlay ROC and PR curves of base (solid) and tuned (dashed) models on the test set."""
    for kind, display_class in [("roc", RocCurveDisplay), ("pr", PrecisionRecallDisplay)]:
        fig, ax = plt.subplots(figsize=(8, 6))
        for i, name in enumerate(tm.make_models()):
            for suffix, style in [("predictions", "-"), ("tuned_predictions", "--")]:
                path = PREDICTIONS_DIR / f"{tm.safe_name(name)}_{suffix}.csv"
                if not path.exists():
                    continue
                df = pd.read_csv(path)
                label = name
                if suffix == "tuned_predictions":
                    label += " (tuned, 20k rows)" if name == "KNN" else " (tuned)"
                display_class.from_predictions(df["y_true"], df["y_proba"], ax=ax, name=label,
                                               curve_kwargs={"color": f"C{i}", "linestyle": style})
        if kind == "roc":
            ax.plot([0, 1], [0, 1], color="grey", linestyle=":")
        ax.set_title(("ROC" if kind == "roc" else "Precision-Recall") + " curves on the test set")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"{kind}_curves_test.png", dpi=150)
        plt.close(fig)


def plot_cv():
    """Bar charts of cross-validation mean +/- std (from 07_cv_tuning.py)."""
    cv_path = RESULTS_DIR / "cv_results.csv"
    if not cv_path.exists():
        print("cv_results.csv not found: run python code/07_cv_tuning.py for the CV figure.")
        return
    cv = pd.read_csv(cv_path)
    metrics = ["F1", "ROC-AUC", "PR-AUC", "Recall", "Precision"]
    cv = cv.set_index("Model")
    mean = cv[[f"{m} mean" for m in metrics]].set_axis(metrics, axis=1)
    std = cv[[f"{m} std" for m in metrics]].set_axis(metrics, axis=1)
    ax = mean.plot.bar(yerr=std, figsize=(10, 5), capsize=3)
    ax.set_title("5-fold GroupKFold cross-validation on train (mean +/- std)")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Model (KNN: CV on 20,000 rows only)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "cv_comparison.png", dpi=150)
    plt.close()


def plot_feature_importance():
    """Random Forest importances and Logistic Regression coefficients (base models refit on train)."""
    X_train, _, y_train, _ = tm.load_train_test()
    models = tm.make_models()
    fitted = {}
    for name in ["Random Forest", "Logistic Regression"]:
        fitted[name] = tm.build_pipeline(name, models[name], X_train).fit(X_train, y_train)
    names = fitted["Random Forest"]["preprocessing"].get_feature_names_out()

    rf = pd.Series(fitted["Random Forest"]["model"].feature_importances_, index=names).nlargest(15)
    rf.sort_values().plot.barh(figsize=(9, 6), title="Random Forest - top 15 feature importances")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_importance_random_forest.png", dpi=150)
    plt.close()

    coef = pd.Series(fitted["Logistic Regression"]["model"].coef_[0], index=names).sort_values()
    top = pd.concat([coef.head(15), coef.tail(15)])
    top.plot.barh(figsize=(9, 8), color=["tab:blue"] * 15 + ["tab:red"] * 15,
                  title="Logistic regression: top 15 positive / negative coefficients\n(numeric features standardized, categorical one-hot)", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_importance_logistic_regression.png", dpi=150)
    plt.close()


def main():
    results_path = RESULTS_DIR / "model_comparison.csv"

    if not results_path.exists():
        print("Run python code/04_evaluation.py first.")
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(results_path)

    metrics = [
        "Accuracy",
        "Precision (class 1)",
        "Recall (class 1)",
        "F1-score (class 1)",
    ]

    for metric in metrics:
        plt.figure(figsize=(9, 5))
        plt.bar(results["Model"], results[metric])
        plt.title(f"Model Comparison - {metric}")
        plt.ylabel(metric)
        plt.ylim(0, 1)
        plt.xticks(rotation=15)
        plt.tight_layout()

        filename = metric.lower().replace(" ", "_").replace("(", "").replace(")", "")
        plt.savefig(FIGURES_DIR / f"comparison_{filename}.png", dpi=150)
        plt.close()

    if (PREDICTIONS_DIR / "random_forest_predictions.csv").exists():
        plot_curves()
        plot_cv()
        plot_feature_importance()

    print("Comparison charts saved in results/figures/.")


if __name__ == "__main__":
    main()
