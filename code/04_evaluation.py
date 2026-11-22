"""Compare model predictions and save metrics and confusion matrices."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
FIGURES_DIR = RESULTS_DIR / "figures"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression_predictions.csv",
    "Decision Tree": "decision_tree_predictions.csv",
    "Random Forest": "random_forest_predictions.csv",
    "KNN": "knn_predictions.csv",
}


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    metrics_rows = []

    for model_name, filename in MODEL_FILES.items():
        prediction_path = PREDICTIONS_DIR / filename
        if not prediction_path.exists():
            print(f"Missing predictions for {model_name}: {prediction_path}")
            print("Run these scripts in order:")
            print("python code/02_preprocessing.py")
            print("python code/03_train_models.py")
            return

        predictions = pd.read_csv(prediction_path)
        y_true = predictions["y_true"]
        y_pred = predictions["y_pred"]

        metrics_rows.append({
            "Model": model_name,
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision (class 1)": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
            "Recall (class 1)": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
            "F1-score (class 1)": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        })

        matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
        display = ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=["No readmission within 30 days (0)", "Readmission within 30 days (1)"],
        )
        display.plot(values_format="d", xticks_rotation=20)
        plt.title(f"Confusion Matrix - {model_name}")
        plt.tight_layout()
        figure_name = model_name.lower().replace(" ", "_") + "_confusion_matrix.png"
        plt.savefig(FIGURES_DIR / figure_name, dpi=150)
        plt.close()

    results = pd.DataFrame(metrics_rows).sort_values(by="F1-score (class 1)", ascending=False)
    output_path = RESULTS_DIR / "model_comparison.csv"
    results.to_csv(output_path, index=False)

    print("\nMODEL COMPARISON")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nSaved metrics table to: {output_path}")
    print(f"Saved confusion matrices to: {FIGURES_DIR}")

    print("\nInterpretation reminders:")
    print("- Class 1 means readmission within 30 days.")
    print("- Accuracy alone may be misleading if the classes are imbalanced.")
    print("- Compare recall and F1-score for class 1 as well as accuracy.")
    print("- Use the actual results above; do not write conclusions before inspecting them.")


if __name__ == "__main__":
    main()
