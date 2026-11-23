"""PCA and K-Means on standardized numeric variables (exploratory, filtered population; target used only for interpretation)."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "diabetic_data.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
RANDOM_STATE = 42
# Same filter as 02_preprocessing.py (death/hospice discharges), so the population is the classifiers' one.
EXCLUDED_DISCHARGE_IDS = [11, 13, 14, 19, 20, 21]
NUM_COLS = ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
            "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses"]
# Heavily right-skewed count variables: log1p before standardizing so a few extreme values do not dominate.
LOG_COLS = ["num_procedures", "number_outpatient", "number_emergency", "number_inpatient"]
RELATIVE_RISK_THRESHOLD = 1.25  # max/min cluster readmission rate above which clusters count as differing
LOW_SILHOUETTE = 0.25


def main():
    if not DATA_PATH.exists():
        print("Dataset not found. Place diabetic_data.csv in data/ (see README.md).")
        return
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH, na_values=["?"], low_memory=False)
    n_before = len(df)
    df = df[~df["discharge_disposition_id"].isin(EXCLUDED_DISCHARGE_IDS)].reset_index(drop=True)
    print(f"Kept {len(df):,} of {n_before:,} stays after excluding discharge_disposition_id in {EXCLUDED_DISCHARGE_IDS}.")
    y = (df["readmitted"] == "<30").astype(int).to_numpy()
    features = df[NUM_COLS].copy()
    features[LOG_COLS] = np.log1p(features[LOG_COLS])
    X = StandardScaler().fit_transform(features)

    # PCA
    pca = PCA(random_state=RANDOM_STATE).fit(X)
    var = pca.explained_variance_ratio_
    comps = np.arange(1, len(var) + 1)
    plt.figure(figsize=(7, 4.5))
    plt.bar(comps, var * 100, label="Per component")
    plt.plot(comps, np.cumsum(var) * 100, "o-", color="red", label="Cumulative")
    plt.title("PCA explained variance")
    plt.xlabel("Principal component")
    plt.ylabel("Explained variance (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pca_explained_variance.png", dpi=150)
    plt.close()

    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(len(X), 10000, replace=False)  # fixed subsample for plots and silhouette
    Z = pca.transform(X[idx])[:, :2]
    plt.figure(figsize=(7, 6))
    for t, lab, c in [(0, "not <30", "tab:blue"), (1, "<30", "tab:red")]:
        m = y[idx] == t
        plt.scatter(Z[m, 0], Z[m, 1], s=5, alpha=0.4, c=c, label=lab)
    plt.title("PCA projection (10,000-row sample) by readmission <30")
    plt.xlabel(f"PC1 ({var[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({var[1]*100:.1f}%)")
    plt.legend(markerscale=3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pca_projection_2d.png", dpi=150)
    plt.close()

    # K-Means, k = 2..8
    ks = list(range(2, 9))
    inertia, sil, models = [], [], {}
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
        models[k] = km
        inertia.append(km.inertia_)
        sil.append(silhouette_score(X[idx], km.labels_[idx]))
    for name, vals, yl, title in [("elbow", inertia, "Inertia", "K-Means elbow curve"),
                                  ("silhouette", sil, "Silhouette score", "K-Means silhouette (10,000-row sample)")]:
        plt.figure(figsize=(6, 4.5))
        plt.plot(ks, vals, "o-")
        plt.title(title)
        plt.xlabel("Number of clusters k")
        plt.ylabel(yl)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"kmeans_{name}.png", dpi=150)
        plt.close()

    # Retained k: best silhouette (elbow curve inspected alongside, see summary table)
    k = ks[int(np.argmax(sil))]
    labels = models[k].labels_
    plt.figure(figsize=(7, 6))
    plt.scatter(Z[:, 0], Z[:, 1], s=5, alpha=0.5, c=labels[idx], cmap="tab10")
    plt.title(f"K-Means clusters (k={k}) on PCA projection")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "kmeans_clusters_pca.png", dpi=150)
    plt.close()

    d = df[NUM_COLS].assign(cluster=labels, target=y)
    summ = d.groupby("cluster").agg(size=("target", "size"), readmission_rate=("target", "mean"),
                                    **{c: (c, "mean") for c in NUM_COLS}).round(4)
    summ.to_csv(RESULTS_DIR / "clustering_summary.csv")
    rates = summ["readmission_rate"]
    relative_risk = rates.max() / rates.min() if rates.min() > 0 else float("inf")
    best_sil = max(sil)
    verdict = (f"clusters differ in readmission rate (max/min = {relative_risk:.2f} > {RELATIVE_RISK_THRESHOLD})"
               if relative_risk > RELATIVE_RISK_THRESHOLD
               else f"readmission rates are similar across clusters (max/min = {relative_risk:.2f} <= {RELATIVE_RISK_THRESHOLD})")
    sil_note = (f"silhouette {best_sil:.3f} is low (< {LOW_SILHOUETTE}): clusters are weakly separated"
                if best_sil < LOW_SILHOUETTE else f"silhouette {best_sil:.3f} (>= {LOW_SILHOUETTE})")
    elbow_drops = ", ".join(f"{a}->{b}: {100*(i1-i2)/i1:.1f}%" for a, b, i1, i2 in zip(ks, ks[1:], inertia, inertia[1:]))
    text = [f"EXPLORATORY analysis on the filtered population of the classifiers ({len(df):,} stays, train and test rows included, death/hospice discharges excluded); it does not feed the classifiers.",
            f"Features: {NUM_COLS}; log1p applied to {LOG_COLS} before StandardScaler.",
            f"PCA variance per component (%): {np.round(var*100, 1).tolist()}",
            f"PCA cumulative (%): {np.round(np.cumsum(var)*100, 1).tolist()}",
            "Silhouette on a 10,000-row sample; k=2 nearly tied with k=3 when it is. Silhouette by k: " + ", ".join(f"k={a}: {s:.3f}" for a, s in zip(ks, sil)),
            f"Inertia relative drop: {elbow_drops}",
            f"Retained k = {k} (highest silhouette = {best_sil:.3f}; inspect the elbow drops above)",
            f"Global readmission rate: {y.mean():.4f}",
            "Readmission rate per cluster: " + ", ".join(f"{c}: {r:.4f} (n={n})" for c, r, n in zip(summ.index, rates, summ["size"])),
            f"Relative risk (max/min cluster rate): {relative_risk:.2f} (descriptive only: no confidence interval, and stays are not independent since patients can have several stays)",
            f"Result: {verdict}; {sil_note}."]
    (RESULTS_DIR / "clustering_summary.txt").write_text("\n".join(text) + "\n")
    print("\n".join(text))
    print(summ.to_string())


if __name__ == "__main__":
    main()
