"""Train the four planned classifiers using one shared test set."""

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Train and compare classifiers")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "results",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "models",
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    X_train = pd.read_csv(args.input / "X_train.csv")
    X_test = pd.read_csv(args.input / "X_test.csv")
    y_train = pd.read_csv(args.input / "y_train.csv").iloc[:, 0]
    y_test = pd.read_csv(args.input / "y_test.csv").iloc[:, 0]

    preprocessor = joblib.load(args.input / "preprocessor.joblib")

    # TODO: Address class imbalance only after inspecting the target distribution.
    # TODO: Tune hyperparameters and document model-selection choices later.
    models = {
        "logistic_regression": LogisticRegression(
            random_state=args.random_state,
            max_iter=1000,
        ),
        "decision_tree": DecisionTreeClassifier(
            random_state=args.random_state
        ),
        "random_forest": RandomForestClassifier(
            random_state=args.random_state,
            n_estimators=100,
        ),
        "knn": KNeighborsClassifier(n_neighbors=5),
    }

    args.output.mkdir(parents=True, exist_ok=True)
    for name, classifier in models.items():
        pipeline = Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("classifier", classifier),
        ])
        pipeline.fit(X_train, y_train)
        joblib.dump(pipeline, args.output / f"{name}.joblib")

        predictions = pipeline.predict(X_test)
        pd.Series(predictions, name="prediction").to_csv(
            args.output / f"{name}_predictions.csv",
            index=False,
        )
        print(f"Trained {name}")

    # Keep the shared test set available for the evaluation script.
    y_test.to_csv(args.output / "y_test.csv", index=False)
    print("Saved models and test predictions to:", args.output)


if __name__ == "__main__":
    main()
