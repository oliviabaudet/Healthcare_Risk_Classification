"""Train five classifiers using the same patient-level split and preprocessing pipelines."""
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
RANDOM_STATE = 42

# These columns contain numeric codes for categories, not continuous measurements.
NUMERIC_CODE_CATEGORIES = [
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
]


def build_preprocessor(X, dense=False):
    categorical_columns = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    for column in NUMERIC_CODE_CATEGORIES:
        if column in X.columns and column not in categorical_columns:
            categorical_columns.append(column)

    numerical_columns = [column for column in X.columns if column not in categorical_columns]

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=not dense)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("numeric", numeric_pipeline, numerical_columns),
        ("categorical", categorical_pipeline, categorical_columns),
    ])

    return preprocessor


def safe_name(model_name):
    return model_name.lower().replace(" ", "_").replace("-", "_")


def make_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced", random_state=RANDOM_STATE, max_depth=12
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=RANDOM_STATE,
            n_jobs=-1, max_depth=18
        ),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(
            class_weight="balanced", random_state=RANDOM_STATE,
            early_stopping=False,  # so that max_iter is a real hyperparameter
        ),
    }


def build_pipeline(model_name, model, X):
    # HistGradientBoosting does not accept sparse input, so it gets dense one-hot features.
    preprocessor = build_preprocessor(X, dense=model_name == "Hist Gradient Boosting")
    return Pipeline(steps=[("preprocessing", preprocessor), ("model", model)])


def load_train_test():
    X_train = pd.read_csv(RESULTS_DIR / "X_train.csv")
    X_test = pd.read_csv(RESULTS_DIR / "X_test.csv")
    y_train = pd.read_csv(RESULTS_DIR / "y_train.csv")["target"]
    y_test = pd.read_csv(RESULTS_DIR / "y_test.csv")["target"]
    # Convert code-like category columns to strings so they are treated as categories.
    for column in NUMERIC_CODE_CATEGORIES:
        if column in X_train.columns:
            X_train[column] = X_train[column].astype("string")
            X_test[column] = X_test[column].astype("string")
    return X_train, X_test, y_train, y_test


def main():
    required_files = [
        RESULTS_DIR / "X_train.csv",
        RESULTS_DIR / "X_test.csv",
        RESULTS_DIR / "y_train.csv",
        RESULTS_DIR / "y_test.csv",
    ]
    missing_files = [path for path in required_files if not path.exists()]
    if missing_files:
        print("Training data not found. Run this command first:")
        print("python code/02_preprocessing.py")
        return

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = load_train_test()

    for model_name, model in make_models().items():
        print(f"\nTraining {model_name}...")
        pipeline = build_pipeline(model_name, model, X_train)
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]

        output = pd.DataFrame({
            "y_true": y_test,
            "y_pred": predictions,
            "y_proba": probabilities,
        })
        output.to_csv(PREDICTIONS_DIR / f"{safe_name(model_name)}_predictions.csv", index=False)
        print(f"Saved predictions to results/predictions/{safe_name(model_name)}_predictions.csv")

    print("\nAll models have been trained and predictions saved.")
    print("Next, run: python code/04_evaluation.py")


if __name__ == "__main__":
    main()
