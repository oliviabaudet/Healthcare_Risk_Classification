# Healthcare Risk Classification

## Project objective
This educational machine-learning project investigates whether a hospital stay for a diabetic patient is followed by readmission within 30 days after discharge.

**Research question:** Can information available about a hospital stay help predict whether the patient will be readmitted within 30 days?

This project is for learning and research only. It is not a clinically validated medical tool.

## Dataset
We use the public UCI dataset **Diabetes 130-US Hospitals for Years 1999–2008**:
https://archive.ics.uci.edu/dataset/296/diabetes+130+us+hospitals+for+years+1999-2008

Download `diabetic_data.csv` and place it in the `data/` folder. The scripts treat `?` as a missing value. Do not commit the raw dataset to GitHub unless you have checked the dataset's terms and your course requirements.

## Target
The original `readmitted` column is converted to a binary target:
- `<30` -> `1` (readmitted within 30 days)
- `>30` -> `0` (not readmitted within 30 days)
- `NO` -> `0` (no recorded readmission)

## Project structure
```text
Healthcare_Risk_Classification/
├── data/
│   └── diabetic_data.csv   # download manually; not included
├── code/
│   ├── 01_explore_data.py
│   ├── 02_preprocessing.py
│   ├── 03_train_models.py
│   ├── 04_evaluation.py
│   ├── 05_visualize_results.py
│   ├── 06_pca_kmeans.py
│   └── 07_cv_tuning.py
├── results/
│   ├── figures/
│   └── predictions/
├── report/
│   └── README.md
├── requirements.txt
└── .gitignore
```

## Setup on Windows PowerShell
Run these commands from the project root in the PyCharm terminal:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, you can skip activation and use:
```powershell
py -m pip install -r requirements.txt
```

## Run the scripts in this order
Run each command from the project root:

```powershell
python code/01_explore_data.py
python code/02_preprocessing.py
python code/03_train_models.py
python code/04_evaluation.py
python code/05_visualize_results.py
python code/06_pca_kmeans.py
python code/07_cv_tuning.py
```

Run `04` and `05` again after `07` to add the tuned models to the confusion matrices, ROC/PR curves and cross-validation figure. `07` took about 10 minutes in total on a 22-core machine (measured, including the cross-validation). The hyperparameter search runs on a stratified, patient-grouped third of the training set (about 26,500 rows; exact size in `search_rows` of `results/best_params.json`) and the best model is refitted on the whole training set. KNN is cross-validated, tuned and fitted on a 20,000-row random subset of the training set (`Train rows` column of `results/cv_results.csv` and `results/tuned_comparison.csv`); the base KNN (k=5, full training set) is added to `tuned_comparison.csv` for comparison. Each model's result is saved in `results/predictions/_state/` with a sha256 fingerprint of the code and of the labels y_train/y_test (feature tables are not checked: use `--force` after changing 02), so an interrupted run resumes and outdated results are recomputed (`--force` recomputes everything).

## Models
- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- Hist Gradient Boosting (`HistGradientBoostingClassifier`, `class_weight='balanced'`, dense one-hot features)

`07_cv_tuning.py` also tunes each model with `RandomizedSearchCV` (`GroupKFold`, 5 folds grouped by `patient_nbr`, scoring `average_precision`, training data only). Best parameters are saved in `results/best_params.json`.

Note: Logistic Regression, Decision Tree, Random Forest and Hist Gradient Boosting use `class_weight`; KNN does not, because scikit-learn's KNN does not support it.

## Evaluation
We report accuracy, precision, recall, F1-score, balanced accuracy, ROC-AUC, PR-AUC (average precision), and confusion matrices, plus ROC and precision-recall curves. Class `1` (readmission within 30 days) is the positive class. Accuracy alone can be misleading when classes are imbalanced.

Cross-validation (`results/cv_results.csv`) reports the mean and standard deviation of F1, ROC-AUC, PR-AUC, recall and precision over 5 patient-grouped folds of the training set. The decision threshold of each tuned model is chosen to maximize F1 on out-of-fold training predictions (`results/thresholds.json`), never on the test set; the tuned models are then evaluated on the test set at threshold 0.5 and at the optimized threshold (`results/tuned_comparison.csv`). Feature importance plots (Random Forest, Logistic Regression coefficients) are saved in `results/figures/`.

## Important methodological choices
- `encounter_id` and `patient_nbr` are excluded from predictive features because they are identifiers.
- Stays whose `discharge_disposition_id` is 11, 13, 14, 19, 20 or 21 (discharged to death or hospice, so readmission is impossible or not meaningful) are removed in `02_preprocessing.py`, which prints the number of rows removed. 01 is exploratory and uses the full file (all 101,766 stays); 06 uses the same filtered population as the classifiers; neither feeds the classifiers.
- The train/test split is made by patient using `patient_nbr`, so one patient's stays do not appear in both sets.
- Imputation, encoding, and scaling are fitted only on training data inside scikit-learn pipelines.
- We must inspect the available features and discuss whether each would be known at the intended prediction time. Results are not proof of causation or clinical usefulness.

## Software versions
Results were produced with Python 3.14, pandas 3.1.1, numpy 2.5.5, matplotlib 3.11.2 and scikit-learn 1.9.1. `requirements.txt` gives minimum versions; scikit-learn >= 1.9 is needed for `PrecisionRecallDisplay.from_predictions(curve_kwargs=...)` (`sparse_output` needs >= 1.2).

## Team
- Olivia Baudet
- Paolo
- Garance

## Limitations to discuss
- Historical US hospital data from 1999–2008 may not represent current practice or other countries.
- Missing data and class imbalance can affect results.
- A model can perform differently across patient groups.
- Predictions must not be used for real clinical decisions without appropriate validation, governance, and expert review.
