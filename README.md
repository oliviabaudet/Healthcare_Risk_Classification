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
│   └── 04_evaluation.py
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
```

## Models
- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)

## Evaluation
We report accuracy, precision, recall, F1-score, and confusion matrices. Class `1` (readmission within 30 days) is the positive class. Accuracy alone can be misleading when classes are imbalanced.

## Important methodological choices
- `encounter_id` and `patient_nbr` are excluded from predictive features because they are identifiers.
- The train/test split is made by patient using `patient_nbr`, so one patient's stays do not appear in both sets.
- Imputation, encoding, and scaling are fitted only on training data inside scikit-learn pipelines.
- We must inspect the available features and discuss whether each would be known at the intended prediction time. Results are not proof of causation or clinical usefulness.

## Team
- Olivia Baudet
- Paolo
- Garance

## Limitations to discuss
- Historical US hospital data from 1999–2008 may not represent current practice or other countries.
- Missing data and class imbalance can affect results.
- A model can perform differently across patient groups.
- Predictions must not be used for real clinical decisions without appropriate validation, governance, and expert review.
