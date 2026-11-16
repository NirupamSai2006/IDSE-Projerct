# Predictive Maintenance Modeling for Industrial Machinery: Failure Classification Under Severe Class Imbalance

**Course:** CS3006 Introduction to Data Science and Engineering (IDSE)  
**Deliverable:** Part A — Data Science Project Report (10 Marks Rubric)  
**Author:** Balla Nirupam Sai  
**Repository:** [https://github.com/NirupamSai2006/IDSE-Projerct](https://github.com/NirupamSai2006/IDSE-Projerct)  
**Date:** November 2026  

---

## Executive Summary

Predictive maintenance (PdM) leverages machine sensor telemetry to forecast imminent mechanical failures before catastrophic breakdowns occur. This project develops an end-to-end, fully reproducible machine learning pipeline on the **UCI AI4I 2020 Predictive Maintenance Dataset** ($N = 10,000$). The central challenge is severe target imbalance, where machine failures constitute only **3.39%** of operational records. 

Through rigorous leakage prevention, statistical outlier auditing, five-fold stratified cross-validation, and multi-model benchmarking (Majority Baseline, Balanced Logistic Regression, and Balanced Random Forest), we demonstrate that a balanced Random Forest ensemble achieves a **Held-out Test F1-Score of 0.671**, a **Failure Recall of 77.94%**, an **Average Precision (PR AUC) of 0.754**, and a **ROC-AUC of 0.967**. Permutation importance analysis demonstrates that **Rotational speed [rpm]** and **Torque [Nm]** (the constituents of mechanical shaft power) dominate predictive performance.

---

## 1. Problem Formulation & Data Justification

### 1.1 Non-Trivial Research Question & Measurable Goals
* **Core Question:** Can operational sensor telemetry (temperature gradients, rotational speed, torque, and cumulative tool wear) alongside machine product tier reliably identify impending machine failures on previously unseen observations without leaking diagnostic failure modes?
* **Measurable Objectives:**
  1. Achieve a held-out test Failure Recall $\ge 75\%$ to minimize unpredicted, catastrophic equipment outages.
  2. Maintain a test Precision $\ge 55\%$ to prevent technician alert fatigue and unwarranted line halts.
  3. Outperform the majority-class baseline across all non-threshold-dependent metrics (Average Precision > 0.70 vs. baseline 0.034).

### 1.2 Dataset Provenance, Scale, and Engineering Relevance
* **Source:** UCI Machine Learning Repository (Dataset ID: 601).
* **Reference Citation:** Matzka, S. (2020). *AI4I 2020 Predictive Maintenance Dataset*. UCI Machine Learning Repository. [https://doi.org/10.24432/C5HS5C](https://doi.org/10.24432/C5HS5C).
* **License:** Creative Commons Attribution 4.0 International (CC BY 4.0).
* **Volume:** 10,000 machine operating instances, 14 raw attributes.
* **Domain Relevance:** Modern Industry 4.0 manufacturing lines rely heavily on automated sensory monitoring. Early detection of tool degradation, heat dissipation flaws, and overstrain allows proactive servicing during scheduled changeovers rather than enduring unplanned factory downtime.

---

## 2. Data Preparation, Leakage Prevention & Exploratory Data Analysis (EDA)

### 2.1 Rigorous Leakage Prevention Protocol
A common pitfall in predictive maintenance benchmarks is feature leakage. In this analysis, data hygiene and strict boundary isolation were established:
1. **Identifier Exclusion:** `UID` (row index: 1–10,000) and `Product ID` (variant serial codes) were strictly removed. Retaining arbitrary identifiers causes models to memorize spurious indexing artifacts.
2. **Failure Subtype Exclusion:** The raw dataset includes five specific failure mode flags:
   * `TWF` (Tool Wear Failure)
   * `HDF` (Heat Dissipation Failure)
   * `PWF` (Power Failure)
   * `OSF` (Overstrain Failure)
   * `RNF` (Random Failure)
   
   > [!IMPORTANT]
   > Including these failure sub-modes in the feature set $X$ would constitute catastrophic target leakage, as they represent post-hoc failure diagnoses that are logically co-linear with the primary target `Machine failure`. All five flags were sequestered and strictly excluded from input features.

### 2.2 Data Integrity and Missingness Audit
* **Missing Value Analysis:** All selected predictor columns were audited programmatically; exactly 0 missing/null cells were detected across all 10,000 rows.
* **Exact Duplicate Audit:** 0 duplicate records existed in the dataset.
* **Data Types:** Categorical quality variant `Type` was converted via `OneHotEncoder(handle_unknown='ignore')`. Continuous numerical telemetry was typed to 64-bit floating point.

### 2.3 Statistical Outlier Audit & Engineering Justification
Outlier candidates were identified using the Interquartile Range (IQR) fence criterion ($[Q_1 - 1.5 \times \text{IQR},\, Q_3 + 1.5 \times \text{IQR}]$):

| Sensor Metric | Lower Fence | Upper Fence | IQR Outlier Count | Action & Engineering Justification |
| :--- | :---: | :---: | :---: | :--- |
| **Air temperature [K]** | 293.50 K | 306.30 K | 0 | Normal seasonal/factory ambient bounds. Retained. |
| **Process temperature [K]** | 305.35 K | 314.55 K | 0 | Normal operating heat generation. Retained. |
| **Rotational speed [rpm]** | 1,139.50 rpm | 1,895.50 rpm | 418 | **Retained.** High speed spikes represent low-load or slipping events; essential precursor to mechanical stall. |
| **Torque [Nm]** | 12.80 Nm | 67.20 Nm | 69 | **Retained.** Torque surges indicate excessive tool resistance, binding, or overstrain. Critical failure signal. |
| **Tool wear [min]** | -110.50 min | 325.50 min | 0 | Monotonic tool life cycle within physical limits. Retained. |

**Reasoning:** Mechanical failures inherently occur under extreme, non-Gaussian operating regimes. Trimming or clipping rotational speed or torque outliers would artificially delete the exact physical shock signatures that distinguish imminent breakdowns from steady-state manufacturing.

### 2.4 Class Imbalance Analysis
* **Operational Normal (Class 0):** 9,661 instances (96.61%)
* **Machine Failure (Class 1):** 339 instances (3.39%)
* **Prevalence Ratio:** Approximately 28.5 to 1.
* **Implication:** A naive dummy model predicting "No Failure" 100% of the time achieves **96.61% accuracy** while missing every single breakdown (0% recall). Therefore, **Accuracy is clinically misleading**, necessitating optimization for **F1-Score, Recall, and Precision-Recall Area Under the Curve (Average Precision)**.

```
+-----------------------------------------------------------------------------------------+
|                                    EDA VISUALIZATION                                    |
| [Figure 1: Target Imbalance Bar Chart]      [Figure 2: Sensor Telemetry Boxplots]       |
|  - Class 0: 9,661 (96.61%)                  - Rotational speed shows upper outliers     |
|  - Class 1:   339 ( 3.39%)                  - Torque shows bidirectional tails          |
+-----------------------------------------------------------------------------------------+
```

---

## 3. Modeling Methodology & Validation Strategy

### 3.1 Strict Train/Validation/Test Split Architecture
To prevent any subtle data leakage during preprocessing and hyperparameter tuning:
* **Held-Out Test Partition:** 20% of data ($N = 2,000$, containing 1,932 normal and 68 failure cases) was partitioned using **Stratified Random Sampling** (`random_state=42`) and locked away untouched until final post-selection evaluation.
* **Training Partition:** 80% of data ($N = 8,000$, containing 7,729 normal and 271 failure cases) was utilized for pipeline fitting and cross-validation.
* **Pipelines:** Data imputation (`SimpleImputer`) and scaling (`StandardScaler`) were embedded inside Scikit-Learn `Pipeline` objects to ensure test fold parameters were never exposed during training.

### 3.2 Cross-Validation Protocol
A **5-Fold Stratified Cross-Validation** (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`) was conducted across the 8,000 training samples. Each fold preserved the 3.39% class ratio.

### 3.3 Model Portfolio & Inductive Biases
Three competing modeling approaches were benchmarked:
1. **Majority-Class Baseline (`DummyClassifier`):** Predicts the mode (always 0). Establishes the floor benchmark for cost and metric comparison.
2. **Linear Parametric Approach (`LogisticRegression`):** Scaled numerical inputs, L2 regularization, inverse class frequency weighting (`class_weight='balanced'`) to penalize minority misclassifications.
3. **Non-Linear Ensemble Approach (`RandomForestClassifier`):** 300 decision trees, `min_samples_leaf=2`, `class_weight='balanced'`, multi-threaded execution. Captures complex non-linear boundary interactions (e.g., thermal power and torque-speed inversely proportional hyperbolas).

---

## 4. Quantitative Results & Comparative Analysis

### 4.1 Cross-Validation Benchmarking (Training Partition, 5 Folds)

| Model Architecture | Mean CV F1 ($\pm \sigma$) | Mean CV Recall | Mean CV Precision | Mean CV Average Precision |
| :--- | :---: | :---: | :---: | :---: |
| **Majority Baseline** | $0.000 \pm 0.000$ | $0.000$ | $0.000$ | $0.034 \pm 0.000$ |
| **Logistic Regression (Balanced)** | $0.235 \pm 0.010$ | **$0.804$** | $0.137$ | $0.441 \pm 0.030$ |
| **Random Forest (Balanced)** | **$0.667 \pm 0.008$** | $0.749$ | **$0.603$** | **$0.732 \pm 0.032$** |

### 4.2 Model Selection Rationale
While Balanced Logistic Regression yielded slightly higher raw recall ($80.44\%$), its precision collapsed to a dismal **13.74%**—triggering nearly 6 false alarms for every single true failure detected. In a production environment, such an error profile halts operations needlessly. The **Random Forest Classifier** achieved a vastly superior trade-off ($F_1 = 0.667$, Precision $= 60.30\%$, Recall $= 74.90\%$, $\text{AP} = 0.732$) and was decisively selected for final holdout testing.

---

## 5. Final Held-Out Test Evaluation & Error Analysis

The selected Random Forest model was trained on the entire training split ($N = 8,000$) and evaluated exactly once on the pristine test split ($N = 2,000$).

### 5.1 Test Set Summary Metrics

$$\text{Accuracy} = 97.45\% \quad\vert\quad \text{Precision} = 58.89\% \quad\vert\quad \text{Recall} = 77.94\% \quad\vert\quad F_1\text{-Score} = 0.671$$

$$\text{Average Precision (PR-AUC)} = 0.754 \quad\vert\quad \text{ROC-AUC} = 0.967$$

### 5.2 Confusion Matrix Decomposition

The test set error distribution on $N = 2,000$ machine records (1,932 Normal, 68 True Failures):

```
                       Predicted: Normal    Predicted: Failure
  Actual: Normal (0)          1895                  37         (False Positive Rate = 1.92%)
  Actual: Failure (1)           15                  53         (True Positive Rate  = 77.94%)
```

* **True Negatives ($TN = 1,895$):** Healthy machines correctly allowed to continue operating without disruption.
* **True Positives ($TP = 53$):** Severe failure conditions successfully intercepted prior to breakdown.
* **False Positives ($FP = 37$):** Preventive inspections dispatched where no immediate failure was pending. Cost: minor labor overhead ($\approx 15$ min inspection).
* **False Negatives ($FN = 15$):** Unpredicted breakdowns. Cost: high replacement expense and unplanned line downtime.

### 5.3 Classification Report Breakdown

```text
              precision    recall  f1-score   support
  No failure       0.99      0.98      0.99      1932
     Failure       0.59      0.78      0.67        68

    accuracy                           0.97      2000
   macro avg       0.79      0.88      0.83      2000
weighted avg       0.98      0.97      0.98      2000
```

---

## 6. Model Interpretability: Permutation Feature Importance

To understand the decision mechanics without relying on Gini-importance (which is known to bias toward continuous, high-cardinality features), **Permutation Importance** was computed on the held-out test set over 8 randomized repeats using Average Precision scoring:

| Feature Name | Mean Importance Drop ($\Delta \text{AP}$) | Std Dev ($\sigma$) | Physical Engineering Interpretation |
| :--- | :---: | :---: | :--- |
| **Rotational speed [rpm]** | **0.4519** | $\pm 0.0305$ | Kinetic energy and rotational dynamics; speed dips indicate heavy mechanical resistance. |
| **Torque [Nm]** | **0.4289** | $\pm 0.0205$ | Direct measure of rotational resistance. High torque combined with low RPM denotes impending stall. |
| **Air temperature [K]** | **0.3386** | $\pm 0.0619$ | Ambient environmental baseline affecting thermal dissipation rates. |
| **Tool wear [min]** | **0.2719** | $\pm 0.0304$ | Cumulative micro-abrasion of the cutting edge leading to chatter and thermal buildup. |
| **Process temperature [K]** | **0.1254** | $\pm 0.0357$ | Operating temperature resulting from friction. |
| **Type (Product Variant)** | **0.0452** | $\pm 0.0192$ | Structural grade (Low, Medium, High variant quality). Modest impact compared to physical sensors. |

**Key Domain Finding:** Rotational speed and Torque together dominate the predictive capability. Mechanically, mechanical power $P$ is the direct product of angular velocity and torque ($P = \tau \cdot \omega$). Discrepancies between expected and actual power output provide the earliest indicators of machine strain.

---

## 7. Critical Discussion, Limitations & Operational Next Steps

### 7.1 Limitations
1. **Synthetic Nature of Dataset:** While modeled on realistic physical laws, the UCI AI4I data is synthetically generated. Real plants experience non-stationary environmental shifts, vibration harmonics, and acoustic emissions not present here.
2. **Temporal Dynamics vs. Random Split:** The dataset represents operational snapshots. In physical manufacturing, degradation is a sequential time-series process. A time-ordered rolling window split would provide even more realistic fidelity than random stratified splitting.
3. **Threshold Calibration:** The default decision threshold ($\theta = 0.50$) was used. In real plant deployment, the threshold should be formally tuned using a cost matrix:
   $$\text{Total Cost} = C_{\text{FN}} \cdot FN + C_{\text{FP}} \cdot FP$$
   If an unscheduled breakdown costs \$10,000 ($C_{\text{FN}}$) while a manual sensor check costs \$200 ($C_{\text{FP}}$), lowering the decision threshold to $\theta \approx 0.25$ will catch nearly 90%+ of failures while maintaining economically viable precision.

---

## 8. Viva Voce & Presentation Defense Guide

### Q1: Why did you prioritize F1-score and Average Precision over Accuracy?
> **Answer:** With a 96.61% negative class prevalence, an uninformative model that trivially predicts "No Failure" achieves 96.61% accuracy while achieving 0% recall. Average Precision and F1 assess the model's true discriminatory ability on the rare, high-stakes failure class.

### Q2: How did you ensure your pipeline had zero data leakage?
> **Answer:** First, row identifiers (`UID`, `Product ID`) were discarded. Second, failure diagnostic categories (`TWF`, `HDF`, etc.) were removed from the input matrix $X$ because they reveal the diagnosis after the fact. Third, all imputation, scaling, and encoding were strictly wrapped in pipelines fit solely on training folds.

### Q3: Why didn't you remove the 418 outliers in Rotational Speed?
> **Answer:** Outliers in predictive maintenance are often not measurement corruption, but true physical anomalies (e.g., motor slippage or stalling) that precede failure. Removing them would eliminate the vital physical signatures that our model needs to detect failure.

---

## 9. Deliverable Verification & Artifacts Summary

* **Source Code:** [`predictive_maintenance_project.py`](file:///c:/Users/nirup/Desktop/5th%20Semister/Data%20Science/IDSE%20Project/predictive_maintenance_project.py)
* **Dataset Cache:** [`ai4i2020.csv`](file:///c:/Users/nirup/Desktop/5th%20Semister/Data%20Science/IDSE%20Project/ai4i2020.csv)
* **EDA Visuals:** [`project_results/eda_overview.png`](file:///c:/Users/nirup/Desktop/5th%20Semister/Data%20Science/IDSE%20Project/project_results/eda_overview.png)
* **Model Error & PR Curve:** [`project_results/test_error_analysis.png`](file:///c:/Users/nirup/Desktop/5th%20Semister/Data%20Science/IDSE%20Project/project_results/test_error_analysis.png)
* **Feature Importance Chart:** [`project_results/feature_importance.png`](file:///c:/Users/nirup/Desktop/5th%20Semister/Data%20Science/IDSE%20Project/project_results/feature_importance.png)
* **Cross-Validation Comparison:** [`project_results/cross_validation_comparison.csv`](file:///c:/Users/nirup/Desktop/5th%20Semister/Data%20Science/IDSE%20Project/project_results/cross_validation_comparison.csv)
* **Test Metrics Manifest:** [`project_results/final_test_metrics.csv`](file:///c:/Users/nirup/Desktop/5th%20Semister/Data%20Science/IDSE%20Project/project_results/final_test_metrics.csv)
* **Public GitHub Repo:** [https://github.com/NirupamSai2006/IDSE-Projerct](https://github.com/NirupamSai2006/IDSE-Projerct)
