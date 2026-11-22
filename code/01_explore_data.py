"""Begin exploratory data analysis for the readmission dataset.

This script intentionally stops with a clear message if the local dataset is
missing. Download the UCI data manually and place it in data/diabetic_data.csv.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Explore diabetic_data.csv")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "diabetic_data.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(
            f"Dataset not found: {args.input}. Download diabetic_data.csv from UCI "
            "and place it in the data directory."
        )

    figures_dir = args.output / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Treat '?' from the source file as missing values.
    data = pd.read_csv(args.input, na_values=["?"])
    print("Dataset shape:", data.shape)
    print("\nColumns:")
    print(data.dtypes.to_string())

    if "readmitted" not in data.columns:
        raise ValueError("The required 'readmitted' target column is missing.")

    print("\nReadmitted value counts:")
    print(data["readmitted"].value_counts(dropna=False).to_string())

    data["binary_readmitted"] = data["readmitted"].map(
        {"<30": 1, ">30": 0, "NO": 0}
    )
    print("\nBinary target counts:")
    print(data["binary_readmitted"].value_counts(dropna=False).to_string())

    missing = data.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    print("\nMissing values:")
    print(missing.to_string() if not missing.empty else "No missing values found.")

    if "encounter_id" in data.columns:
        print("\nEncounter count:", data["encounter_id"].nunique())
    if "patient_nbr" in data.columns:
        print("Unique patient count:", data["patient_nbr"].nunique())
        repeated = data["patient_nbr"].duplicated(keep=False).sum()
        print("Rows belonging to repeated patients:", repeated)

    if not missing.empty:
        plt.figure(figsize=(10, 6))
        missing.plot(kind="bar")
        plt.title("Missing values by column")
        plt.ylabel("Number of missing values")
        plt.xlabel("Column")
        plt.xticks(rotation=75, ha="right")
        plt.tight_layout()
        plt.savefig(figures_dir / "missing_values.png", dpi=150)
        plt.close()

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(6, 4))
    sns.countplot(data=data, x="binary_readmitted")
    plt.title("Binary 30-day readmission target")
    plt.xlabel("Readmitted within 30 days (0 = No, 1 = Yes)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(figures_dir / "target_distribution.png", dpi=150)
    plt.close()

    print("\nSaved EDA plots to:", figures_dir)
    print(
        "\nTODO: Document data-quality decisions, repeated-patient considerations,"
        " categorical cardinality, and potential leakage before final modelling."
    )


if __name__ == "__main__":
    main()
