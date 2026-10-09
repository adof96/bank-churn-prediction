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
1. Preprocessing: dropped rows with missing `Tenure` and the identifier columns
   (`RowNumber`, `CustomerId`, `Surname`), one-hot encoded categorical features,
   and scaled numeric features with `StandardScaler` fitted on the training set only.
2. Stratified train / validation / test split (60 / 20 / 20).
3. Baseline `DecisionTreeClassifier` — used as a sanity check against a
   constant-prediction baseline given the class imbalance.
4. Class imbalance handling: `class_weight='balanced'`, and manual
   upsampling/downsampling of the training set.
5. Models compared on the validation set: decision tree (plain and balanced),
   `LogisticRegression` and `RandomForestClassifier`, each trained on the original,
   upsampled and downsampled training sets.
6. Final model: `RandomForestClassifier(n_estimators=150, class_weight='balanced', min_samples_leaf=3)`
   trained on the upsampled training set, evaluated once on the test set.

## Result
Test set metrics of the final model (target was F1 ≥ 0.59):

| F1 | AUC-ROC | Recall | Precision |
|---|---|---|---|
| **0.635** | **0.855** | 0.650 | 0.621 |

ROC curve and class-balance plots are in [`Figuras/`](Figuras/).

## Project structure
```
src/betabank/
    config.py       # paths, random seed, columns, F1 goal
    data.py         # loading and cleaning
    features.py     # one-hot encoding, train/valid/test split, scaling
    sampling.py     # upsampling and downsampling
    models.py       # candidate models
    evaluation.py   # F1, AUC-ROC, recall, precision and model comparison
    plots.py        # class balance and ROC curve
main.py               # full pipeline from the command line
Proyecto_betabank.py  # step-by-step analysis (percent-format notebook)
tests/                # pytest tests
```

## How to run
```bash
python -m venv venvpbb
venvpbb\Scripts\activate          # Windows (source venvpbb/bin/activate on Linux/macOS)
pip install -r requirements.txt
pip install -e .

python main.py   # compares models, evaluates the best one on test, saves Figuras/Roc_curve.png
pytest           # runs the tests
```
`Proyecto_betabank.py` is a Jupytext "percent-format" script — open it in Jupyter or
VS Code (with the `venvpbb` interpreter selected) to run it cell by cell.
