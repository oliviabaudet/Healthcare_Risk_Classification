"""Create reproducible training/test splits and preprocessing pipelines."""

import argparse
from pathlib import Path

import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET = "binary_readmitted"
IDENTIFIERS = ["encounter_id", "patient_nbr"]


def parse_args():
    parser = argparse.ArgumentParser(description="Split and prepare the dataset")
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
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def make_binary_target(data):
    """Convert the UCI readmitted column into the required binary target."""
    if "readmitted" not in data.columns:
        raise ValueError("The required 'readmitted' column is missing.")
    target = data["readmitted"].map({"<30": 1, ">30": 0, "NO": 0})
    if target.isna().any():
        unexpected = sorted(data.loc[target.isna(), "readmitted"].dropna().unique())
        raise ValueError(f"Unexpected readmitted values found: {unexpected}")
    return target.astype(int)


def patient_level_split(data, test_size, random_state):
    """Use patient numbers so that each patient belongs to only one split."""
    if "patient_nbr" not in data.columns:
        raise ValueError("patient_nbr is required for a patient-level split.")

    split = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    train_index, test_index = next(
        split.split(data, data[TARGET], groups=data["patient_nbr"])
    )
    return train_index, test_index


def build_preprocessor(X):
    """Build a preprocessing pipeline without fitting it on test data."""
    # TODO: Remove post-discharge or otherwise unavailable features after inspection.
    # TODO: Add any feature-subset decision supported by EDA and evidence.
    numerical_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numerical_steps = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    # Standard scaling is fitted here and used only for pipelines that need it.
    numerical_scaled = Pipeline(
        numerical_steps + [("scaler", StandardScaler())]
    )
    numerical_passthrough = Pipeline(numerical_steps)

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("numeric_scaled", numerical_scaled, numerical_features),
        ("numeric_passthrough", numerical_passthrough, numerical_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])


def main():
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(
            f"Dataset not found: {args.input}. Download diabetic_data.csv from UCI "
            "and place it in the data directory."
        )

    data = pd.read_csv(args.input, na_values=["?"])
    data[TARGET] = make_binary_target(data)

    train_index, test_index = patient_level_split(
        data, args.test_size, args.random_state
    )
    train_data = data.iloc[train_index].copy()
    test_data = data.iloc[test_index].copy()

    # Training and test patients must not overlap.
    overlap = set(train_data["patient_nbr"]).intersection(test_data["patient_nbr"])
    if overlap:
        raise RuntimeError("Patient overlap detected between train and test sets.")

    feature_columns = [
        column
        for column in data.columns
        if column not
        in set(IDENTIFIERS + [TARGET, "readmitted"])
    ]
    X_train = train_data[feature_columns]
    X_test = test_data[feature_columns]
    y_train = train_data[TARGET]
    y_test = test_data[TARGET]

    args.output.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(args.output / "X_train.csv", index=False)
    X_test.to_csv(args.output / "X_test.csv", index=False)
    y_train.to_csv(args.output / "y_train.csv", index=False)
    y_test.to_csv(args.output / "y_test.csv", index=False)

    joblib.dump(build_preprocessor(X_train), args.output / "preprocessor.joblib")

    print("Training rows:", len(X_train))
    print("Test rows:", len(X_test))
    print("Unique training patients:", train_data["patient_nbr"].nunique())
    print("Unique test patients:", test_data["patient_nbr"].nunique())
    print("Features excluded: encounter_id, patient_nbr, readmitted")
    print(
        "TODO: Inspect feature availability and define the final feature-selection "
        "decision before modelling."
    )


if __name__ == "__main__":
    main()
