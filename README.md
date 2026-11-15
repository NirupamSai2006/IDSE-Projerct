# IDSE data science project and learning game

Two runnable Python deliverables for the course rubric. The project analyzes machine failure risk; the game lets players discover the bias-variance trade-off by trying model settings and observing computed test results.

## Install and run

Use Python 3.10 or newer. In a terminal opened in this folder:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python predictive_maintenance_project.py
streamlit run data_science_game.py
```

The project downloads the dataset through the UCI repository package on its first run, so an internet connection is needed. Results and plots are written to `project_results/`.

## Part A: question and data

**Question:** Can machine type and operating measurements identify failure cases on unseen records?

The [UCI AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i) has 10,000 records and is relevant to engineering maintenance. UCI states it is synthetic data designed to reflect industrial predictive-maintenance data; it is not direct evidence about a particular factory. The dataset is CC BY 4.0. Cite it as: Matzka, S. (2020). *AI4I 2020 Predictive Maintenance Dataset*. UCI Machine Learning Repository. https://doi.org/10.24432/C5HS5C.

The analysis excludes identifiers (`UDI`, `Product ID`) and failure subtype labels (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) from the model inputs. Including subtype labels would leak information about the overall `Machine failure` target. It checks missingness, exact duplicates, and IQR outlier candidates. Plausible extremes are retained; outlier candidates are documented for review rather than automatically removed.

## Analysis and evaluation

The script creates a fixed, stratified 80/20 train/test split. It compares a majority-class baseline, balanced logistic regression, and balanced random forest with five-fold stratified cross-validation on the training split. It selects the model by mean CV F1, then evaluates that model once on the held-out test split. The report includes precision, recall, F1, average precision, ROC AUC, a confusion matrix, and a precision-recall plot. Average precision and recall matter because failures are uncommon and missing a failure can be costly. Permutation importance summarizes which available features mattered most to the selected model; it does not establish causation.

Generated evidence in `project_results/`:

- `eda_overview.png`: class balance and sensor distributions
- `outlier_audit.csv`, `descriptive_statistics.csv`: preparation and EDA evidence
- `cross_validation_comparison.csv`, `final_test_metrics.csv`: model comparison and held-out metrics
- `test_error_analysis.png`: confusion matrix and precision-recall curve
- `permutation_importance.csv`, `feature_importance.png`: feature importance

**Limitations to discuss:** data are synthetic; a random split does not measure future-time performance; costs of false alarms and missed failures are not specified; feature importance is not causal. A stronger follow-up would test on real, time-ordered plant data and choose a probability threshold from maintenance costs.

## Part B: how to play and learning goals

Run `streamlit run data_science_game.py`. Over four rounds, pick the number of training measurements and polynomial degree, then press **Run this round**. A separate noisy test sample measures generalization. Points are computed from test MSE relative to a simple mean-prediction baseline. Feedback compares train/test error and highlights possible underfitting or overfitting. The underlying data, models, predictions, metrics, and plots are computed each round; the answer is not a hard-coded quiz key.

**Learning goals:** explain underfitting and overfitting; distinguish training performance from generalization; describe how sample size and model complexity affect results. The curve and noise are simulated for learning, not production engineering data.

### Peer playtest evidence to complete

Run the game with at least a few classmates and record genuine results here before submission:

| Playtester (initials only) | Choices tried | What they noticed or learned | Feedback / change made |
|---|---|---|---|
| 1 |  |  |  |
| 2 |  |  |  |
| 3 |  |  |  |

Do not claim playtest results until the game has actually been tried by peers.
