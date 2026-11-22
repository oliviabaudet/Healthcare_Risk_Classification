# Healthcare Risk Classification

## Project overview

This project investigates whether machine learning models can predict whether a
diabetic patient will be readmitted to hospital within 30 days after discharge.
It is a university group project for Olivia, Paolo and Garance.

The work is intended as an exploratory academic exercise. No model has been
trained yet, and no performance results are claimed in this repository.

## Research question

Can machine learning algorithms predict whether a diabetic patient will be
readmitted within 30 days after discharge?

The binary target is derived from `readmitted`:

- `1`: original value `<30`
- `0`: original values `>30` or `NO`

This is a prediction of the 30-day readmission outcome, not a prediction of
whether a patient has diabetes.

## Objectives

- Perform exploratory data analysis (EDA).
- Clean the data and handle missing values.
- Encode categorical variables and scale numerical variables where appropriate.
- Compare feature-selection options and justify the selected features.
- Split data without fitting preprocessing transformations on the test set.
- Train Logistic Regression, Decision Tree, Random Forest and K-Nearest
  Neighbors classifiers.
- Evaluate models with accuracy, precision, recall, F1-score and confusion
  matrices.
- Discuss findings, limitations and possible future improvements.

## Dataset

The project uses the UCI **Diabetes 130-US hospitals for years 1999-2008**
dataset:

https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

Download `diabetic_data.csv` from UCI and place it at:

```text
data/diabetic_data.csv
```

Do not commit the dataset unless its licence and the repository rules permit it.
The repository `.gitignore` excludes CSV data files by default.

## Repository structure

```text
Healthcare_Risk_Classification/
├── data/                  # Local datasets only
├── code/
│   ├── 01_explore_data.py
│   ├── 02_preprocessing.py
│   ├── 03_train_models.py
│   └── 04_evaluation.py
├── results/
│   └── figures/
├── report/
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate the environment with:

```bash
.venv\Scripts\activate
```

## Running the project

The scripts are beginner-friendly and should be run in order from the repository
root:

```bash
python code/01_explore_data.py
python code/02_preprocessing.py
python code/03_train_models.py
python code/04_evaluation.py
```

Optional custom paths can be passed to the scripts with `--input` or `--output`,
depending on the script. Run `python code/<script>.py --help` for details.

The scripts currently contain TODO markers for decisions that require inspection
of the actual dataset. They are designed to support, rather than replace, the
group's documented methodological work.

## Algorithms

The project compares four scikit-learn classification algorithms:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. K-Nearest Neighbors (KNN)

Preprocessing is implemented with scikit-learn pipelines so that the
transformations are fitted only on training data. Numerical features are scaled
for Logistic Regression and KNN where appropriate. Categorical features are
encoded with missing-value handling.

The `patient_nbr` field should be investigated for patient-level splitting. The
same patient can have more than one hospital encounter, so a patient-level split
is preferred when it provides enough separation for reliable evaluation.
`encounter_id` and `patient_nbr` are excluded from model features.

## Evaluation

All models use the same test set. Planned metrics are:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrices

The group should consider class imbalance and cannot rely on accuracy alone.
Precision, recall and F1-score are especially important for the minority
readmission group. The selected approach to class imbalance must be documented
before final modelling.

## Limitations and ethical considerations

- The dataset covers hospital stays from 1999 to 2008 and may not reflect modern
  clinical practice or patient populations.
- It may contain repeated encounters for the same patient.
- Missing values, coding changes and measurement differences require careful
  review.
- Features that are not available at the intended prediction time must be
  excluded to avoid data leakage.
- The available variables may contain bias and may not represent every
  population fairly.
- This project is a historical, educational modelling exercise. It is **not**
  clinically validated and must not be used to make decisions about patient care
  or to replace professional judgement.
- A predictive warning should not be used to discriminate against or unfairly
  penalize patients or groups.
