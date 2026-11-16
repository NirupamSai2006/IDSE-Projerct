"""Reproducible AI4I predictive-maintenance course project.

Run: python predictive_maintenance_project.py
Downloads the UCI dataset on first run; writes figures and results to ./project_results/.
"""

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, average_precision_score,
                             classification_report, confusion_matrix, f1_score,
                             precision_score, precision_recall_curve, recall_score,
                             roc_auc_score)
from sklearn.model_selection import (StratifiedKFold, cross_validate, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SEED = 42
OUT = Path("project_results")
TARGET = "Machine failure"
NUMERIC = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]",
           "Torque [Nm]", "Tool wear [min]"]
FEATURES = ["Type", *NUMERIC]


def load_data():
    """Fetch the original UCI data or load local cache; keep download explicit for reproducibility."""
    local_csv = Path("ai4i2020.csv")
    if local_csv.exists():
        frame = pd.read_csv(local_csv)
    else:
        try:
            from ucimlrepo import fetch_ucirepo
        except ImportError as exc:
            raise SystemExit("Install requirements first: pip install -r requirements.txt") from exc
        dataset = fetch_ucirepo(id=601)
        frame = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
    
    # Normalize column whitespace to protect against CSV/API formatting differences.
    frame.columns = [str(c).strip() for c in frame.columns]
    # Standardize column names with unit brackets
    rename_map = {}
    for col in frame.columns:
        c_lower = col.lower()
        if "air temp" in c_lower:
            rename_map[col] = "Air temperature [K]"
        elif "process temp" in c_lower:
            rename_map[col] = "Process temperature [K]"
        elif "rotational speed" in c_lower:
            rename_map[col] = "Rotational speed [rpm]"
        elif "torque" in c_lower:
            rename_map[col] = "Torque [Nm]"
        elif "tool wear" in c_lower:
            rename_map[col] = "Tool wear [min]"
        elif "machine failure" in c_lower:
            rename_map[col] = "Machine failure"
    frame = frame.rename(columns=rename_map)
    return frame


def make_pipeline(estimator, scale=False):
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps)
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])
    prep = ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC),
        ("categorical", categorical_pipe, ["Type"]),
    ])
    return Pipeline([("prepare", prep), ("model", estimator)])


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    OUT.mkdir(exist_ok=True)
    data = load_data()
    required = FEATURES + [TARGET]
    missing_cols = sorted(set(required) - set(data.columns))
    if missing_cols:
        raise ValueError(f"Dataset is missing expected columns: {missing_cols}")

    print("\nQUESTION")
    print("Can machine type and operating measurements identify failure cases on unseen records?")
    print("Goal: compare models, emphasizing failure recall and precision-recall performance.")
    print("\nDATA AND PREPARATION")
    print(f"Rows: {len(data):,}; columns: {data.shape[1]}; source: UCI AI4I 2020 (synthetic industrial data).")
    print("Excluded UID/Product ID identifiers and TWF/HDF/PWF/OSF/RNF failure subtype labels to prevent leakage.")
    print("Target is the overall Machine failure label. Missing values and duplicates are checked below.")
    print("Missing values by selected column:\n", data[required].isna().sum().to_string())
    duplicate_count = int(data.duplicated().sum())
    print(f"Exact duplicate rows: {duplicate_count}")
    if duplicate_count:
        data = data.drop_duplicates().copy()
        print("Removed exact duplicate rows.")
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    data = data.dropna(subset=[TARGET]).copy()
    data[TARGET] = data[TARGET].astype(int)
    data = data[data[TARGET].isin([0, 1])].copy()
    X, y = data[FEATURES], data[TARGET]
    counts = y.value_counts().sort_index()
    print("Class counts (0=no failure, 1=failure):", counts.to_dict())
    print(f"Failure prevalence: {y.mean():.2%}; this is imbalanced, so accuracy alone can mislead.")

    # EDA: target balance, sensor distributions, and IQR-based outlier counts.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    counts.rename(index={0: "No failure", 1: "Failure"}).plot.bar(ax=axes[0], color=["#4c78a8", "#f58518"])
    axes[0].set_title("Failure labels are imbalanced")
    axes[0].set_ylabel("Number of records")
    data[NUMERIC].boxplot(ax=axes[1], rot=35)
    axes[1].set_title("Sensor distributions (points beyond whiskers are candidates, not automatic errors)")
    fig.tight_layout(); fig.savefig(OUT / "eda_overview.png", dpi=160); plt.close(fig)
    print("EDA insight: the failure class is rare; the overview plot shows class balance and sensor spread.")

    outlier_rows = []
    for col in NUMERIC:
        q1, q3 = X[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n = int(((X[col] < low) | (X[col] > high)).sum())
        outlier_rows.append({"sensor": col, "iqr_candidate_count": n, "lower_fence": low, "upper_fence": high})
    pd.DataFrame(outlier_rows).to_csv(OUT / "outlier_audit.csv", index=False)
    print("Outlier handling: kept plausible extremes; IQR candidates are recorded in outlier_audit.csv for review.")
    data[FEATURES + [TARGET]].describe(include="all").to_csv(OUT / "descriptive_statistics.csv")

    # A stratified held-out test set is used only at final evaluation. No failure subtype columns enter X.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y)
    models = {
        "Majority baseline": (make_pipeline(DummyClassifier(strategy="most_frequent")), "f1"),
        "Logistic regression": (make_pipeline(LogisticRegression(max_iter=2000, class_weight="balanced"), scale=True), "f1"),
        "Random forest": (make_pipeline(RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                               class_weight="balanced", random_state=SEED, n_jobs=-1)), "f1"),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scoring = {"f1": "f1", "recall": "recall", "precision": "precision", "ap": "average_precision"}
    rows = []
    print("\nTRAINING COMPARISON (five-fold CV on training partition)")
    for name, (pipe, _) in models.items():
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        row = {"model": name}
        for metric in scoring:
            row[f"cv_{metric}_mean"] = scores[f"test_{metric}"].mean()
            row[f"cv_{metric}_std"] = scores[f"test_{metric}"].std()
        rows.append(row)
        print(f"{name}: CV F1={row['cv_f1_mean']:.3f}±{row['cv_f1_std']:.3f}; "
              f"recall={row['cv_recall_mean']:.3f}; precision={row['cv_precision_mean']:.3f}; "
              f"average precision={row['cv_ap_mean']:.3f}")
    cv_table = pd.DataFrame(rows).sort_values("cv_f1_mean", ascending=False)
    cv_table.to_csv(OUT / "cross_validation_comparison.csv", index=False)

    # Select using cross-validated F1, then fit once and evaluate on untouched test data.
    chosen_name = cv_table.iloc[0]["model"]
    chosen = models[chosen_name][0]
    chosen.fit(X_train, y_train)
    y_pred = chosen.predict(X_test)
    y_score = chosen.predict_proba(X_test)[:, 1]
    metrics = {
        "chosen_model": chosen_name,
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_precision": precision_score(y_test, y_pred, zero_division=0),
        "test_recall": recall_score(y_test, y_pred, zero_division=0),
        "test_f1": f1_score(y_test, y_pred, zero_division=0),
        "test_average_precision": average_precision_score(y_test, y_score),
        "test_roc_auc": roc_auc_score(y_test, y_score),
        "seed": SEED,
        "n_test": len(y_test),
    }
    pd.DataFrame([metrics]).to_csv(OUT / "final_test_metrics.csv", index=False)
    print("\nFINAL HELD-OUT TEST (used once after model selection)")
    for key, value in metrics.items():
        print(f"{key}: {value:.3f}" if isinstance(value, (float, np.floating)) else f"{key}: {value}")
    print("\nError analysis (rows=true, columns=predicted):\n", confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=["No failure", "Failure"], zero_division=0))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["No failure", "Failure"],
                                            cmap="Blues", ax=axes[0], colorbar=False)
    axes[0].set_title(f"Held-out errors: {chosen_name}")
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    axes[1].plot(recall, precision, label=f"{chosen_name} (AP={metrics['test_average_precision']:.3f})")
    axes[1].axhline(y_test.mean(), linestyle="--", color="gray", label="Prevalence baseline")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall trade-off", xlim=(0, 1), ylim=(0, 1.02))
    axes[1].legend()
    fig.tight_layout(); fig.savefig(OUT / "test_error_analysis.png", dpi=160); plt.close(fig)

    # Permutation importance on held-out data measures performance drop when a feature is shuffled.
    importance = permutation_importance(chosen, X_test, y_test, n_repeats=8, random_state=SEED,
                                        scoring="average_precision", n_jobs=-1)
    imp = pd.DataFrame({"feature": FEATURES, "importance_mean": importance.importances_mean,
                        "importance_std": importance.importances_std}).sort_values("importance_mean", ascending=False)
    imp.to_csv(OUT / "permutation_importance.csv", index=False)
    ax = imp.sort_values("importance_mean").plot.barh(x="feature", y="importance_mean", xerr="importance_std",
                                                      legend=False, figsize=(8, 4), color="#4c78a8")
    ax.set(xlabel="Decrease in average precision after shuffling", ylabel="", title="Held-out permutation importance")
    plt.tight_layout(); plt.savefig(OUT / "feature_importance.png", dpi=160); plt.close()
    print("Interpretation: prioritize failure recall when missed failures are costly, while tracking precision to estimate inspection burden.")
    print("Limitations: synthetic records may not represent a plant; random split may overstate performance for future time periods.")
    print("Next steps: validate on real time-ordered machine data, tune the decision threshold using maintenance costs, and investigate false negatives.")
    print(f"\nSaved charts and CSV evidence to: {OUT.resolve()}")


if __name__ == "__main__":
    main()
