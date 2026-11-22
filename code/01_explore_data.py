"""Explore the UCI diabetes readmission dataset and save basic plots."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "diabetic_data.csv"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"


def main():
    if not DATA_PATH.exists():
        print("Dataset not found.")
        print(f"Download diabetic_data.csv from the UCI dataset page and place it here:\n{DATA_PATH}")
        print("https://archive.ics.uci.edu/dataset/296/diabetes+130+us+hospitals+for+years+1999-2008")
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH, na_values=["?"])

    print("\nFIRST FIVE ROWS")
    print(df.head())
    print("\nDATASET SHAPE (rows, columns)")
    print(df.shape)
    print("\nCOLUMN NAMES")
    print(df.columns.tolist())
    print("\nDATA TYPES")
    print(df.dtypes)
    print("\nMISSING VALUES (top 20 columns)")
    print(df.isnull().sum().sort_values(ascending=False).head(20))
    print("\nREADMISSION COUNTS")
    print(df["readmitted"].value_counts(dropna=False))
    print("\nREADMISSION PERCENTAGES")
    print((df["readmitted"].value_counts(normalize=True, dropna=False) * 100).round(2))

    df["readmitted"].value_counts(dropna=False).plot(kind="bar")
    plt.title("Original Readmission Categories")
    plt.xlabel("readmitted")
    plt.ylabel("Number of hospital stays")
    plt.xticks(rotation=0)
    plt.tight_layout()
    output_path = FIGURES_DIR / "readmission_distribution.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\nSaved chart to: {output_path}")


if __name__ == "__main__":
    main()
