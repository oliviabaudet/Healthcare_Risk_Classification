"""Explore the UCI diabetes readmission dataset and save basic plots."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "diabetic_data.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
NUM_COLS = ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
            "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses"]
KEY_NUM = ["time_in_hospital", "num_lab_procedures", "num_medications", "number_inpatient",
           "number_outpatient", "number_emergency", "number_diagnoses"]
CAT_COLS = ["age", "race", "gender", "A1Cresult", "insulin", "change", "diabetesMed",
            "admission_type_id", "discharge_disposition_id"]
TOP_N = 10


def save(name):
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / name, dpi=150)
    plt.close()


def extra_eda(df):
    """EDA figures (eda_*.png) and results/eda_summary.txt."""
    df = df.copy()
    df["target"] = (df["readmitted"] == "<30").astype(int)
    rate = df["target"].mean()
    out = ["Exploratory, full file (all 101,766 stays, including the 2,423 death/hospice stays excluded from classification)",
           f"Rows: {len(df)}", f"Global <30 readmission rate: {rate:.4f} ({df['target'].sum()} rows)"]

    # (a) missing values
    miss = (df.isnull().mean() * 100).sort_values(ascending=False)
    miss = miss[miss > 0]
    ax = miss.sort_values().plot(kind="barh", figsize=(8, 5))
    ax.set_title("Missing values per column (columns with > 0 %)")
    ax.set_xlabel("Missing (%)")
    ax.set_ylabel("Column")
    save("eda_missing_values.png")
    out += ["", "MISSING % (columns > 0 %)", miss.round(2).to_string()]

    # (b) numeric distributions by class
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, col in zip(axes.ravel(), KEY_NUM):
        hi = df[col].quantile(0.99)
        bins = range(0, int(hi) + 2) if hi < 30 else 30
        for t, lab in [(0, "not <30"), (1, "<30")]:
            ax.hist(df.loc[(df.target == t) & (df[col] <= hi), col], bins=bins, density=True, alpha=0.5, label=lab)
        ax.set_title(col)
        ax.set_xlabel(f"{col} (clipped at 99th pct)")
        ax.set_ylabel("Density")
        ax.legend()
    axes.ravel()[-1].axis("off")
    fig.suptitle("Key numeric variables by readmission class")
    save("eda_numeric_by_target.png")

    # (c) readmission rate by category
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    out += ["", f"READMISSION <30 RATE BY MODALITY (top {TOP_N} by count)"]
    for ax, col in zip(axes.ravel(), CAT_COLS):
        g = df.groupby(df[col].fillna("Missing"))["target"].agg(["mean", "size"])
        g = g.sort_values("size", ascending=False).head(TOP_N).sort_values("mean")
        ax.barh(g.index.astype(str), g["mean"] * 100)
        for i, (m, n) in enumerate(zip(g["mean"], g["size"])):
            ax.text(m * 100, i, f" n={n}", va="center", fontsize=8)
        ax.axvline(rate * 100, color="red", ls="--", label="global rate")
        ax.set_title(col)
        ax.set_xlabel("Readmission <30 days (%)")
        ax.set_ylabel(col)
        ax.legend(fontsize=8)
        out += ["", col, g.assign(rate_pct=(g["mean"] * 100).round(2))[["size", "rate_pct"]].to_string()]
    fig.suptitle("Readmission <30 days rate by category (n = number of stays)")
    save("eda_rate_by_category.png")

    # (d) correlation heatmap
    corr = df[NUM_COLS + ["target"]].corr()
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, label="Pearson correlation")
    ax.set_title("Correlation of numeric variables (incl. binary target)")
    save("eda_correlation_heatmap.png")
    ct = corr["target"].drop("target")
    out += ["", "CORRELATION WITH TARGET (sorted by |r|)",
            ct.reindex(ct.abs().sort_values(ascending=False).index).round(4).to_string()]

    # patients vs encounters
    vc = df["patient_nbr"].value_counts()
    out += ["", "PATIENTS", f"Encounters: {len(df)}", f"Unique patients: {len(vc)}",
            f"Patients with >1 encounter: {(vc > 1).sum()}",
            f"Rows whose patient_nbr appears more than once: {int(vc[vc > 1].sum())}"]
    (RESULTS_DIR / "eda_summary.txt").write_text("\n".join(out) + "\n")
    print("\n".join(out[1:3]), f"\nSaved eda_* figures and {RESULTS_DIR / 'eda_summary.txt'}")


def main():
    if not DATA_PATH.exists():
        print("Dataset not found.")
        print(f"Download diabetic_data.csv from the UCI dataset page and place it here:\n{DATA_PATH}")
        print("https://archive.ics.uci.edu/dataset/296/diabetes+130+us+hospitals+for+years+1999-2008")
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH, na_values=["?"], low_memory=False)

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
    extra_eda(df)


if __name__ == "__main__":
    main()
