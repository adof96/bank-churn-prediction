import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from betabank.config import DROP_COLS, TARGET
from betabank.data import clean, load_data
from betabank.evaluation import evaluate
from betabank.features import prepare
from betabank.sampling import downsample, upsample


@pytest.fixture(scope='module')
def prepared():
    return prepare(clean(load_data()))


@pytest.fixture
def toy():
    features = pd.DataFrame({'x': range(10)})
    target = pd.Series([0] * 8 + [1] * 2)
    return features, target


def test_clean_drops_ids_and_nans():
    df = clean(load_data())
    assert not set(DROP_COLS) & set(df.columns)
    assert df['Tenure'].notna().all()


def test_splits_are_disjoint_and_keep_class_ratio(prepared):
    features_train, features_valid, features_test, target_train, _, target_test = prepared
    assert not set(features_train.index) & set(features_valid.index)
    assert not set(features_train.index) & set(features_test.index)
    assert abs(target_train.mean() - target_test.mean()) < 0.01
    assert TARGET not in features_train.columns


def test_upsample(toy):
    features, target = upsample(*toy, repeat=4)
    assert len(features) == len(target) == 8 + 2 * 4
    assert (target == 1).sum() == 8


def test_downsample(toy):
    features, target = downsample(*toy, fraction=0.5)
    assert len(features) == len(target) == 4 + 2
    assert (target == 1).sum() == 2


def test_evaluate_returns_metrics(toy):
    features, target = toy
    model = DecisionTreeClassifier(random_state=0).fit(features, target)
    metrics = evaluate(model, features, target)
    assert {'f1', 'auc_roc', 'recall', 'precision'} <= metrics.keys()
    assert all(0 <= value <= 1 for value in metrics.values())
