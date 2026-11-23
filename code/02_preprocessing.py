"""Create a patient-level train/test split. Preprocessing is fitted later in model pipelines."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "diabetic_data.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
RANDOM_STATE = 42
TEST_SIZE = 0.20
# Discharge to death / hospice: readmission is impossible or not meaningful, so these stays are excluded.
EXCLUDED_DISCHARGE_IDS = [11, 13, 14, 19, 20, 21]


def main():
    if not DATA_PATH.exists():
        print("Dataset not found. Place diabetic_data.csv in data/ (see README.md).")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH, na_values=["?"])

    n_before = len(df)
    df = df[~df["discharge_disposition_id"].isin(EXCLUDED_DISCHARGE_IDS)].copy()
    print(f"Excluded {n_before - len(df):,} stays discharged to death/hospice "
          f"(discharge_disposition_id in {EXCLUDED_DISCHARGE_IDS}); {len(df):,} rows kept.")

    # Convert the original target into a binary target:
    # 1 = readmitted within 30 days; 0 = not readmitted within 30 days.
    target_mapping = {"<30": 1, ">30": 0, "NO": 0}
    df["target"] = df["readmitted"].map(target_mapping)

    # Remove rows without one of the expected target values, if any exist.
    df = df.dropna(subset=["target"]).copy()
    df["target"] = df["target"].astype(int)

    # patient_nbr is used only to keep a patient's visits together during splitting.
    # It is not included as a predictive feature.
    groups = df["patient_nbr"]
    y = df["target"]
    X = df.drop(columns=["readmitted", "target", "encounter_id", "patient_nbr"], errors="ignore")

    # GroupShuffleSplit prevents a patient's records from being in both train and test.
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_indices, test_indices = next(splitter.split(X, y, groups=groups))

    X_train = X.iloc[train_indices].copy()
    X_test = X.iloc[test_indices].copy()
    y_train = y.iloc[train_indices].copy()
    y_test = y.iloc[test_indices].copy()

    # Save raw feature tables. Imputation, encoding, and scaling happen only
    # after this split, inside the scikit-learn pipelines in 03_train_models.py.
    X_train.to_csv(RESULTS_DIR / "X_train.csv", index=False)
    X_test.to_csv(RESULTS_DIR / "X_test.csv", index=False)
    y_train.to_frame(name="target").to_csv(RESULTS_DIR / "y_train.csv", index=False)
    y_test.to_frame(name="target").to_csv(RESULTS_DIR / "y_test.csv", index=False)
    # Patient ids of the training rows (same order as X_train), used for grouped CV in 07_cv_tuning.py.
    groups.iloc[train_indices].to_frame(name="patient_nbr").to_csv(RESULTS_DIR / "train_groups.csv", index=False)

    print("Preprocessing split complete.")
    print(f"Training rows: {len(X_train):,}")
    print(f"Test rows: {len(X_test):,}")
    print("\nTraining target distribution:")
    print(y_train.value_counts(normalize=True).sort_index().rename(index={0: "No readmission within 30 days", 1: "Readmission within 30 days"}))
    print("\nTest target distribution:")
    print(y_test.value_counts(normalize=True).sort_index().rename(index={0: "No readmission within 30 days", 1: "Readmission within 30 days"}))
    print("\nSaved X_train.csv, X_test.csv, y_train.csv, and y_test.csv in results/.")


if __name__ == "__main__":
    main()
