import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from betabank.config import NUMERIC, RANDOM_STATE, TARGET, TEST_SIZE, VALID_SIZE


def encode(df):
    return pd.get_dummies(df, drop_first=True)


def split(df):
    """Divide en train / valid / test estratificando por la clase objetivo."""
    target = df[TARGET]
    features = df.drop(TARGET, axis=1)

    features_rest, features_test, target_rest, target_test = train_test_split(
        features, target, test_size=TEST_SIZE,
        stratify=target, random_state=RANDOM_STATE
    )
    features_train, features_valid, target_train, target_valid = train_test_split(
        features_rest, target_rest, test_size=VALID_SIZE / (1 - TEST_SIZE),
        stratify=target_rest, random_state=RANDOM_STATE
    )
    return (features_train, features_valid, features_test,
            target_train, target_valid, target_test)


def scale(features_train, *others):
    """Ajusta el escalador solo con train y lo aplica a todos los conjuntos."""
    scaler = StandardScaler()
    scaler.fit(features_train[NUMERIC])

    scaled = []
    for features in (features_train, *others):
        features = features.copy()
        features[NUMERIC] = scaler.transform(features[NUMERIC])
        scaled.append(features)
    return scaled


def prepare(df):
    """Limpieza ya aplicada -> codificacion, division y escalado."""
    (features_train, features_valid, features_test,
     target_train, target_valid, target_test) = split(encode(df))
    features_train, features_valid, features_test = scale(
        features_train, features_valid, features_test
    )
    return (features_train, features_valid, features_test,
            target_train, target_valid, target_test)
