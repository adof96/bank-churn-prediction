# Bank Churn Prediction

## Business problem
Beta Bank is losing customers month over month. Retaining an existing customer is
cheaper than acquiring a new one, so the bank wants to know, in advance, which
customers are likely to leave (churn) so retention efforts can be targeted at them.

## Data
`datasets/Churn.csv` — 10,000 bank customers with:
- Demographics: `Geography`, `Gender`, `Age`
- Account info: `CreditScore`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`
- Target: `Exited` (1 = churned) — **20.4% positive class**, a moderate imbalance

## Methods tried
1. Preprocessing: dropped rows with missing `Tenure`, one-hot encoded categorical
   features, scaled numeric features with `StandardScaler`.
2. Baseline `DecisionTreeClassifier` — used as a sanity check against a
   constant-prediction baseline given the class imbalance.
3. Class imbalance handling: `class_weight='balanced'`, and manual
   upsampling/downsampling of the training set.
4. Models compared: `LogisticRegression` and `RandomForestClassifier`.
5. Final model: `RandomForestClassifier(n_estimators=150, class_weight='balanced', min_samples_leaf=3)`
   trained on the upsampled training set.

## Result
- **F1 = 0.60** on the test set (target was ≥ 0.59)
- **AUC-ROC = 0.86**

ROC curve and class-balance plots are in [`Figuras/`](Figuras/).

## How to run
```bash
pip install -r requirements.txt
```
`Proyecto_betabank.py` is a Jupytext "percent-format" script — open it in Jupyter or
VS Code to run it cell by cell, or execute it directly from the repo root:
```bash
python Proyecto_betabank.py
```
