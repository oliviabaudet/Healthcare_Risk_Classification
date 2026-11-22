"""Evaluate the four models on the shared held-out test set."""

import argparse
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained classifiers")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "results" / "models",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    y_test = pd.read_csv(args.input / "y_test.csv").iloc[:, 0]
    model_names = [
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "knn",
    ]

    metrics = {}
    for name in model_names:
        prediction_file = args.input / f"{name}_predictions.csv"
        if not prediction_file.exists():
            raise FileNotFoundError(
                f"Missing predictions: {prediction_file}. Run 03_train_models.py first."
            )
        predictions = pd.read_csv(prediction_file).iloc[:, 0]

        metrics[name] = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(
                precision_score(y_test, predictions, zero_division=0)
            ),
            "recall": float(recall_score(y_test, predictions, zero_division=0)),
            "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        }

        matrix = confusion_matrix(y_test, predictions)
        plt.figure(figsize=(5, 4))
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix: {name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        figures_dir = args.output / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(figures_dir / f"{name}_confusion_matrix.png", dpi=150)
        plt.close()

    args.output.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    pd.DataFrame(metrics).T.to_csv(args.output / "metrics.csv")
    print(pd.DataFrame(metrics).T.to_string())
    print("Saved metrics to:", metrics_path)


if __name__ == "__main__":
    main()
