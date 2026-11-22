
"""Create comparison charts for the four classification models."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


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

    print("Comparison charts saved in results/figures/.")


if __name__ == "__main__":
    main()
